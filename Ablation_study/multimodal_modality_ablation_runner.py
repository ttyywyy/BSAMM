# -*- coding: utf-8 -*-
"""
multimodal_modality_ablation_runner.py

Deep-learning ablation study (中文：深度学习的消融实验 / 模态消融实验)

This single script consolidates the 8 separate training scripts you used for modality ablation:
- minus MRI / minus WSI / minus IHC / minus Clinical
- stage1 (larger augmentation) and stage2 (smaller augmentation + optional pretrained init)

✅ Privacy-safe by design:
- No hard-coded local absolute paths (e.g., /Volumes/..., hospital names)
- No patient identifiers are printed; indices are only used for alignment/splitting

------------------------------------------------------------
Expected input files (default names; can be changed via args)
------------------------------------------------------------
In --data_dir, prepare these files:

1) MRI_features.xlsx
2) wsi_features.xlsx
3) ich_features.xlsx
4) Clinical_features.xlsx
5) PFS_survival.xlsx

Optional (recommended, to reproduce your fixed cohort split):
6) TRAIN_SURVIVAL.xlsx   (only used to provide train IDs via its index)
7) VAL_SURVIVAL.xlsx     (only used to provide validation IDs via its index)

All excel files should have:
- index_col=0 as patient_id (string / int OK)
- survival file contains columns: time, event

------------------------------------------------------------
Example usage
------------------------------------------------------------
# Stage 1: ablate Clinical (i.e., keep MRI+WSI+IHC)
python multimodal_modality_ablation_runner.py \
  --data_dir /path/to/data \
  --ablate clinical \
  --stage 1 \
  --out_dir ./results_ablation \
  --n_runs 50

# Stage 2: continue training from a pretrained checkpoint
python multimodal_modality_ablation_runner.py \
  --data_dir /path/to/data \
  --ablate clinical \
  --stage 2 \
  --init_ckpt ./results_ablation/stage1_minus_clinical/run_001/best_run001.pth \
  --out_dir ./results_ablation \
  --n_runs 1

Notes:
- "ablate" is the modality removed: mri / wsi / ihc / clinical
- Output is organized as:
  out_dir/stage{stage}_minus_{ablate}/run_XXX/{best_runXXX.pth, fold_log.csv, ...}
"""

from __future__ import annotations

import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Cox proportional hazards loss
# -----------------------------
def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    risk: (B, 1) or (B,)
    time: (B,)
    event:(B,)
    """
    risk, time, event = [x.view(-1) for x in (risk, time, event)]
    # Sort by time descending (standard for Cox partial likelihood)
    idx = torch.argsort(time, descending=True)
    risk = risk[idx]
    event = event[idx]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    nll = -(event * (risk - log_cumsum)).sum()
    return nll / (event.sum() + eps)


# -----------------------------
# Model blocks
# -----------------------------
class ModalityEncoder(nn.Module):
    """Simple linear + ReLU encoder to project a modality to embed_dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SelfAttention(nn.Module):
    """
    Single-head self-attention (as used in your scripts).
    Input:  (B, T, input_dim)
    Output: (B, T, hidden_dim)
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_proj   = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)  # (B,T,T)
        attn  = torch.softmax(score, dim=-1)
        out   = torch.matmul(attn, V)  # (B,T,hidden_dim)
        return out


class SurvAttentionAblation(nn.Module):
    """
    Generic ablation model that supports:
    - keeping any subset of modalities among {mri, wsi, ihc, clinical}
    - optional MRI encoder (1342 -> 768) if MRI is included
    - attention2 kept for checkpoint compatibility (not used by default)
    """
    def __init__(
        self,
        modalities: List[str],
        input_dims: Dict[str, int],
        embed_dim: int = 768,
        attn_hidden: int = 256,
        use_attention2: bool = True,
        use_attention2_in_forward: bool = False,
    ):
        super().__init__()
        assert len(modalities) >= 2, "Need at least 2 modalities to run attention."

        self.modalities = modalities
        self.embed_dim = embed_dim
        self.attn_hidden = attn_hidden
        self.use_attention2_in_forward = use_attention2_in_forward

        # Encoders: only create if in_dim != embed_dim (MRI is usually 1342 -> 768)
        self.encoders = nn.ModuleDict()
        for m in modalities:
            in_dim = int(input_dims[m])
            if in_dim != embed_dim:
                self.encoders[m] = ModalityEncoder(in_dim, embed_dim)

        self.attention1 = SelfAttention(embed_dim, attn_hidden)
        self.attention2 = SelfAttention(attn_hidden, attn_hidden) if use_attention2 else nn.Identity()

        self.ln = nn.LayerNorm(attn_hidden, eps=1e-5, elementwise_affine=True)

        # Flattened dim = T * attn_hidden
        flat_dim = len(modalities) * attn_hidden
        self.for_head = nn.Sequential(
            nn.Linear(flat_dim, 32, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode each modality to embed_dim, then stack to tokens (B, T, embed_dim)
        tokens = []
        for m in self.modalities:
            x = batch[m]  # (B, dim_m)
            if m in self.encoders:
                x = self.encoders[m](x)
            tokens.append(x)

        combined = torch.stack(tokens, dim=1)     # (B, T, embed_dim)
        z1 = self.attention1(combined)           # (B, T, attn_hidden)

        # In your original training scripts, attention2 exists but is not used in forward.
        if self.use_attention2_in_forward and not isinstance(self.attention2, nn.Identity):
            z1 = self.attention2(z1)

        z = self.ln(z1).flatten(1)               # (B, T*attn_hidden)
        risk = self.for_head(z)                  # (B, 1)
        return risk


# -----------------------------
# Dataset
# -----------------------------
class MultimodalDataset(Dataset):
    """
    A unified Dataset that supports modality ablation and your two augmentation styles.

    Augmentation styles observed in your scripts:
    - stage1: if MRI is present, jitter is applied to MRI features only
             if MRI is absent, jitter is applied to concatenated remaining modalities
    - stage2: jitter is applied to concatenated features of all present modalities

    We expose this as:
      aug_mode = "mri_only" | "all" | "none"
    """
    def __init__(
        self,
        df_by_modality: Dict[str, pd.DataFrame],
        surv_df: pd.DataFrame,
        modalities: List[str],
        augment: bool = False,
        aug_mode: str = "none",
    ):
        super().__init__()
        self.modalities = modalities
        self.augment = augment
        self.aug_mode = aug_mode
        self.jr = 0.0
        self.gen = torch.Generator()

        # store tensors
        self.X: Dict[str, torch.Tensor] = {}
        for m in modalities:
            self.X[m] = torch.tensor(df_by_modality[m].values, dtype=torch.float32)

        # survival
        self.surv = surv_df[["time", "event"]].copy()

        # for "all" augmentation: compute slice ranges for splitting
        self._sizes = {m: self.X[m].shape[1] for m in modalities}
        self._cum_sizes = np.cumsum([self._sizes[m] for m in modalities])

    def set_jitter(self, jr: float, seed: int) -> None:
        self.jr = float(jr)
        self.gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.surv)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # copy features
        for m in self.modalities:
            out[m] = self.X[m][i].clone()

        # augmentation
        if self.augment and self.jr != 0 and self.aug_mode != "none":
            if self.aug_mode == "mri_only" and ("mri" in self.modalities):
                noise = torch.randn(out["mri"].size(), generator=self.gen) * abs(self.jr)
                out["mri"] = out["mri"] + noise
            else:
                # aug_mode == "all" or mri_only but MRI absent -> treat as all
                vec = torch.cat([out[m] for m in self.modalities], dim=0)
                noise = torch.randn(vec.size(), generator=self.gen) * abs(self.jr)
                vec = vec + noise

                # split back
                start = 0
                for m, end in zip(self.modalities, self._cum_sizes):
                    out[m] = vec[start:end]
                    start = end

        # survival
        out["time"]  = torch.tensor(float(self.surv.iloc[i]["time"]),  dtype=torch.float32)
        out["event"] = torch.tensor(float(self.surv.iloc[i]["event"]), dtype=torch.float32)
        return out


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}


# -----------------------------
# Data IO helpers
# -----------------------------
@dataclass
class DataBundle:
    df_by_modality: Dict[str, pd.DataFrame]
    surv: pd.DataFrame
    train_ids: List[str]
    val_ids: List[str]


def read_table_auto(path: str, index_col: int = 0) -> pd.DataFrame:
    """Read .xlsx or .csv with a unified API."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, index_col=index_col)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, index_col=index_col)
    raise ValueError(f"Unsupported file type: {path}")


def load_multimodal_data(
    data_dir: str,
    modalities: List[str],
    mri_file: str,
    wsi_file: str,
    ihc_file: str,
    clinical_file: str,
    survival_file: str,
    train_survival_file: Optional[str],
    val_survival_file: Optional[str],
) -> DataBundle:
    """
    Load feature tables + survival, align by intersection of patient IDs.

    If train_survival_file and val_survival_file are provided:
    - their indices define train_ids and val_ids (no label leakage; only IDs used)
    Otherwise:
    - randomly split 80/20 as train/val
    """
    # file mapping
    file_map = {
        "mri": mri_file,
        "wsi": wsi_file,
        "ihc": ihc_file,
        "clinical": clinical_file,
    }

    # read features
    df_by_modality = {}
    for m in modalities:
        p = os.path.join(data_dir, file_map[m])
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {m} feature file: {p}")
        df_by_modality[m] = read_table_auto(p, index_col=0)

    # read survival
    surv_path = os.path.join(data_dir, survival_file)
    if not os.path.exists(surv_path):
        raise FileNotFoundError(f"Missing survival file: {surv_path}")
    surv = read_table_auto(surv_path, index_col=0)
    if not {"time", "event"}.issubset(set(surv.columns)):
        raise ValueError("Survival file must contain columns: time, event")

    # align common IDs
    common = surv.index
    for m in modalities:
        common = common.intersection(df_by_modality[m].index)

    if len(common) == 0:
        raise RuntimeError("No common patient IDs across selected modalities and survival table.")

    # subset + sort for reproducibility
    common = common.sort_values()
    surv = surv.loc[common]
    for m in modalities:
        df_by_modality[m] = df_by_modality[m].loc[common]

    # get train/val ids
    if train_survival_file and val_survival_file:
        tr = read_table_auto(os.path.join(data_dir, train_survival_file), index_col=0)
        va = read_table_auto(os.path.join(data_dir, val_survival_file), index_col=0)
        train_ids = [str(x) for x in tr.index.intersection(common)]
        val_ids   = [str(x) for x in va.index.intersection(common)]
        if len(train_ids) == 0 or len(val_ids) == 0:
            raise RuntimeError("Train/Val split files have no overlap with the loaded common IDs.")
    else:
        # random 80/20 split
        ids = [str(x) for x in common]
        rng = np.random.RandomState(42)
        rng.shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_ids = ids[:cut]
        val_ids = ids[cut:]

    # cast index to str for consistent matching
    surv.index = surv.index.astype(str)
    for m in modalities:
        df_by_modality[m].index = df_by_modality[m].index.astype(str)

    return DataBundle(df_by_modality=df_by_modality, surv=surv, train_ids=train_ids, val_ids=val_ids)


def infer_input_dims(df_by_modality: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    return {m: int(df.shape[1]) for m, df in df_by_modality.items()}


# -----------------------------
# Training / evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Returns:
      (cox_loss, c_index)
    """
    model.eval()
    risks, times, events = [], [], []
    losses = []

    for batch in dl:
        # move to device
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        risk = model(batch_dev).view(-1)

        time = batch_dev["time"].view(-1)
        event = batch_dev["event"].view(-1)

        loss = cox_ph_loss(risk, time, event).item()
        losses.append(loss)

        risks.append(risk.detach().cpu().numpy())
        times.append(time.detach().cpu().numpy())
        events.append(event.detach().cpu().numpy())

    risks = np.concatenate(risks)
    times = np.concatenate(times)
    events = np.concatenate(events)

    cidx = concordance_index(times, -risks, events)  # negative risk => higher risk => shorter time
    return float(np.mean(losses)), float(cidx)


def train_one_run(
    run_id: int,
    out_dir: str,
    bundle: DataBundle,
    modalities: List[str],
    stage: int,
    init_ckpt: Optional[str],
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    jitter_range: Tuple[float, float],
    aug_mode: str,
) -> Tuple[float, str]:
    """
    Train one run and save best checkpoint.

    Returns:
      best_val_cindex, best_ckpt_path
    """
    os.makedirs(out_dir, exist_ok=True)

    # split by ids
    tr_idx = bundle.surv.index.intersection(pd.Index(bundle.train_ids))
    va_idx = bundle.surv.index.intersection(pd.Index(bundle.val_ids))

    # build dataset / loader
    df_tr = {m: bundle.df_by_modality[m].loc[tr_idx] for m in modalities}
    df_va = {m: bundle.df_by_modality[m].loc[va_idx] for m in modalities}

    ds_tr = MultimodalDataset(df_tr, bundle.surv.loc[tr_idx], modalities, augment=True, aug_mode=aug_mode)
    ds_va = MultimodalDataset(df_va, bundle.surv.loc[va_idx], modalities, augment=False, aug_mode="none")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=64,         shuffle=False, collate_fn=collate_fn)

    # model
    input_dims = infer_input_dims(bundle.df_by_modality)
    model = SurvAttentionAblation(modalities=modalities, input_dims=input_dims).to(device)

    # optional pretrained init (stage2)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device)
        # allow missing/unexpected keys if user loads from slightly different code versions
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[run {run_id:03d}] init_ckpt missing keys (ignored): {len(missing)}")
        if unexpected:
            print(f"[run {run_id:03d}] init_ckpt unexpected keys (ignored): {len(unexpected)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_c = -1.0
    best_ckpt = os.path.join(out_dir, f"best_run{run_id:03d}.pth")

    log_rows = []

    for ep in range(1, epochs + 1):
        # refresh jitter each epoch (same pattern as your scripts)
        jr = random.uniform(float(jitter_range[0]), float(jitter_range[1]))
        seed = random.randrange(10**9)
        ds_tr.set_jitter(jr, seed)

        # ---- train ----
        model.train()
        for batch in dl_tr:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()

            risk = model(batch_dev).view(-1)
            loss = cox_ph_loss(risk, batch_dev["time"].view(-1), batch_dev["event"].view(-1))
            loss.backward()
            opt.step()

        # ---- eval ----
        tr_loss, tr_c = evaluate(model, dl_tr, device)
        va_loss, va_c = evaluate(model, dl_va, device)

        if va_c > best_val_c:
            best_val_c = va_c
            torch.save(model.state_dict(), best_ckpt)

        log_rows.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_cindex": tr_c,
            "val_loss": va_loss,
            "val_cindex": va_c,
            "best_val_cindex": best_val_c,
            "jitter": jr,
        })

        if ep % 20 == 0 or ep == 1:
            print(f"[run {run_id:03d}] ep={ep:03d} | trC={tr_c:.4f} vaC={va_c:.4f} best={best_val_c:.4f}")

    # save logs
    pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, "fold_log.csv"), index=False)
    return best_val_c, best_ckpt


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal self-attention survival model - modality ablation runner")

    p.add_argument("--data_dir", type=str, required=True, help="Directory containing feature/survival files")
    p.add_argument("--out_dir", type=str, default="./ablation_results", help="Output root directory")
    p.add_argument("--ablate", type=str, required=True, choices=["mri", "wsi", "ihc", "clinical"],
                   help="Which modality to remove (ablation target)")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Training stage: 1 or 2")
    p.add_argument("--init_ckpt", type=str, default=None, help="Optional init checkpoint for stage2 (or warm-start)")

    # file names
    p.add_argument("--mri_file", type=str, default="MRI_features.xlsx")
    p.add_argument("--wsi_file", type=str, default="wsi_features.xlsx")
    p.add_argument("--ihc_file", type=str, default="ich_features.xlsx")
    p.add_argument("--clinical_file", type=str, default="Clinical_features.xlsx")
    p.add_argument("--survival_file", type=str, default="PFS_survival.xlsx")
    p.add_argument("--train_survival_file", type=str, default=None,
                   help="Optional survival file whose index defines TRAIN IDs (e.g., tt_PFS_survival.xlsx)")
    p.add_argument("--val_survival_file", type=str, default=None,
                   help="Optional survival file whose index defines VAL IDs (e.g., zd_PFS_survival.xlsx)")

    # training params
    p.add_argument("--n_runs", type=int, default=10, help="Number of repeated runs with different seeds")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)

    # stage-specific defaults (can be overridden)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--jitter_lo", type=float, default=None)
    p.add_argument("--jitter_hi", type=float, default=None)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✔ device: {device}")

    # modalities to keep
    all_mods = ["mri", "wsi", "ihc", "clinical"]
    modalities = [m for m in all_mods if m != args.ablate]
    print(f"✔ ablation: minus {args.ablate} | keep: {modalities}")

    # stage defaults (matching your scripts)
    if args.stage == 1:
        lr = args.lr if args.lr is not None else 1e-5
        wd = args.weight_decay if args.weight_decay is not None else 1e-4
        jr_lo = args.jitter_lo if args.jitter_lo is not None else -0.7
        jr_hi = args.jitter_hi if args.jitter_hi is not None else 0.7

        # stage1 behavior:
        # - if MRI exists -> jitter MRI only
        # - else -> jitter all (because MRI is absent)
        aug_mode = "mri_only" if "mri" in modalities else "all"

        init_ckpt = None  # stage1 from scratch by default
    else:
        lr = args.lr if args.lr is not None else 1e-6
        wd = args.weight_decay if args.weight_decay is not None else 1.0
        jr_lo = args.jitter_lo if args.jitter_lo is not None else -0.2
        jr_hi = args.jitter_hi if args.jitter_hi is not None else 0.2

        # stage2 behavior: jitter all modalities (as in your step2 scripts)
        aug_mode = "all"

        init_ckpt = args.init_ckpt
        if init_ckpt is None:
            raise ValueError("Stage 2 requires --init_ckpt (pretrained checkpoint path).")

    # load data
    bundle = load_multimodal_data(
        data_dir=args.data_dir,
        modalities=modalities,
        mri_file=args.mri_file,
        wsi_file=args.wsi_file,
        ihc_file=args.ihc_file,
        clinical_file=args.clinical_file,
        survival_file=args.survival_file,
        train_survival_file=args.train_survival_file,
        val_survival_file=args.val_survival_file,
    )
    print(f"✔ N_total={len(bundle.surv)} | N_train={len(bundle.train_ids)} | N_val={len(bundle.val_ids)}")

    exp_root = os.path.join(args.out_dir, f"stage{args.stage}_minus_{args.ablate}")
    os.makedirs(exp_root, exist_ok=True)

    best_overall = -1.0
    best_path = None

    summary_rows = []

    for r in range(1, args.n_runs + 1):
        # vary seed across runs
        run_seed = args.seed + r * 101
        set_seed(run_seed)

        run_dir = os.path.join(exp_root, f"run_{r:03d}")
        val_c, ckpt = train_one_run(
            run_id=r,
            out_dir=run_dir,
            bundle=bundle,
            modalities=modalities,
            stage=args.stage,
            init_ckpt=init_ckpt,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=lr,
            weight_decay=wd,
            jitter_range=(jr_lo, jr_hi),
            aug_mode=aug_mode,
        )

        summary_rows.append({
            "run": r,
            "stage": args.stage,
            "ablate": args.ablate,
            "keep_modalities": "+".join(modalities),
            "val_cindex": val_c,
            "ckpt_path": ckpt,
            "lr": lr,
            "weight_decay": wd,
            "jitter_range": f"[{jr_lo},{jr_hi}]",
            "aug_mode": aug_mode,
            "run_seed": run_seed,
        })

        if val_c > best_overall:
            best_overall = val_c
            best_path = ckpt

    pd.DataFrame(summary_rows).to_csv(os.path.join(exp_root, "run_summary.csv"), index=False)

    print("\n✅ Done.")
    print(f"Best val C-index: {best_overall:.4f}")
    print(f"Best checkpoint : {best_path}")
    print(f"Outputs saved to: {exp_root}")


if __name__ == "__main__":
    main()
