# -*- coding: utf-8 -*-
"""
extract_penultimate_features.py
-------------------------------
Load a trained multimodal survival model, run inference on a dataset,
and extract only the 32-dimensional penultimate-layer features.

Expected input files in DATA_DIR:
- MRI_features.xlsx
- wsi_features.xlsx
- ich_features.xlsx
- clinical_features.xlsx

Each file should use patient_id as the index column.

Output:
- One Excel file containing:
    1) patient_id
    2) 32 penultimate-layer features

Notes:
- The 32-dimensional features are defined as the output of:
      Linear(1024 -> 32) + ReLU
- Dropout is not applied during inference because model.eval() is used.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =======================
# 1) User configuration
# =======================
DATA_DIR = Path("path/to/data_dir")
CHECKPOINT_PATH = Path("path/to/model_checkpoint.pth")
OUTPUT_EXCEL_PATH = Path("outputs/penultimate_features_32.xlsx")
BATCH_SIZE = 256


# =======================
# 2) Model definition
# =======================
class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.hidden_dim), dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class SurvivalAttentionModel(nn.Module):
    """
    Returns:
        feat32: (B, 32)
    """
    def __init__(self):
        super().__init__()
        self.mri_encoder = ModalityEncoder(1342, 768)
        self.attention1 = SelfAttention(768, 256)

        # Kept for checkpoint compatibility, although it is not used in forward().
        self.attention2 = SelfAttention(256, 256)

        self.layer_norm = nn.LayerNorm(256, eps=1e-5, elementwise_affine=True)

        self.head = nn.Sequential(
            nn.Linear(1024, 32, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(
        self,
        mri: torch.Tensor,
        wsi: torch.Tensor,
        ich: torch.Tensor,
        text: torch.Tensor
    ) -> torch.Tensor:
        # MRI encoding
        mri_feat = self.mri_encoder(mri)  # (B, 768)

        # Stack four modalities
        combined = torch.stack([mri_feat, wsi, ich, text], dim=1)  # (B, 4, 768)

        # Self-attention
        z1 = self.attention1(combined)  # (B, 4, 256)

        # LayerNorm + flatten -> 1024-d fused representation
        fused_feat = self.layer_norm(z1).flatten(1)  # (B, 1024)

        # Extract only the 32-d penultimate-layer features
        x32 = self.head[0](fused_feat)   # Linear(1024 -> 32)
        x32 = self.head[1](x32)          # ReLU
        feat32 = x32                     # (B, 32)

        return feat32


# =======================
# 3) Dataset
# =======================
class MultimodalDataset(Dataset):
    def __init__(
        self,
        patient_ids: List[str],
        mri: pd.DataFrame,
        wsi: pd.DataFrame,
        ich: pd.DataFrame,
        text: pd.DataFrame
    ):
        self.patient_ids = list(patient_ids)

        self.mri = torch.tensor(mri.values, dtype=torch.float32)
        self.wsi = torch.tensor(wsi.values, dtype=torch.float32)
        self.ich = torch.tensor(ich.values, dtype=torch.float32)
        self.text = torch.tensor(text.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        return (
            self.patient_ids[idx],
            self.mri[idx],
            self.wsi[idx],
            self.ich[idx],
            self.text[idx]
        )


# =======================
# 4) Data loading and alignment
# =======================
def load_and_align_data(
    data_dir: Path
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mri = pd.read_excel(data_dir / "MRI_features.xlsx", index_col=0)
    wsi = pd.read_excel(data_dir / "wsi_features.xlsx", index_col=0)
    ich = pd.read_excel(data_dir / "ich_features.xlsx", index_col=0)
    text = pd.read_excel(data_dir / "clinical_features.xlsx", index_col=0)

    # Convert patient_id to string for consistent matching
    mri.index = mri.index.astype(str)
    wsi.index = wsi.index.astype(str)
    ich.index = ich.index.astype(str)
    text.index = text.index.astype(str)

    # Intersect patient IDs across all modalities
    common_ids = mri.index.intersection(wsi.index)
    common_ids = common_ids.intersection(ich.index)
    common_ids = common_ids.intersection(text.index)
    common_ids = list(common_ids)

    if len(common_ids) == 0:
        raise ValueError(
            "No common patient_id was found across the four modality files. "
            "Please check the index column in each Excel file."
        )

    # Align all tables by common patient_id
    mri = mri.loc[common_ids]
    wsi = wsi.loc[common_ids]
    ich = ich.loc[common_ids]
    text = text.loc[common_ids]

    n = len(common_ids)
    assert len(mri) == len(wsi) == len(ich) == len(text) == n

    print(f"Aligned samples: {n}")
    print(f"MRI   shape: {mri.shape}")
    print(f"WSI   shape: {wsi.shape}")
    print(f"ICH   shape: {ich.shape}")
    print(f"TEXT  shape: {text.shape}")

    return common_ids, mri, wsi, ich, text


# =======================
# 5) Checkpoint loading
# =======================
def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove "module." prefix if the model was saved with DataParallel
    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

    return model


# =======================
# 6) Feature extraction
# =======================
def main():
    if str(DATA_DIR) == "path/to/data_dir":
        raise ValueError("Please update DATA_DIR before running the script.")
    if str(CHECKPOINT_PATH) == "path/to/model_checkpoint.pth":
        raise ValueError("Please update CHECKPOINT_PATH before running the script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and align data
    patient_ids, mri, wsi, ich, text = load_and_align_data(DATA_DIR)

    dataset = MultimodalDataset(patient_ids, mri, wsi, ich, text)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = SurvivalAttentionModel().to(device)
    model = load_model_weights(model, CHECKPOINT_PATH, device)
    model.eval()
    print(f"Checkpoint loaded from: {CHECKPOINT_PATH}")

    # Extract features
    all_patient_ids = []
    all_feat32 = []

    with torch.no_grad():
        for pid_batch, mri_batch, wsi_batch, ich_batch, text_batch in dataloader:
            mri_batch = mri_batch.to(device)
            wsi_batch = wsi_batch.to(device)
            ich_batch = ich_batch.to(device)
            text_batch = text_batch.to(device)

            feat32_batch = model(mri_batch, wsi_batch, ich_batch, text_batch)

            all_patient_ids.extend(list(pid_batch))
            all_feat32.append(feat32_batch.cpu())

    all_feat32 = torch.cat(all_feat32, dim=0)  # (N, 32)
    print(f"Extracted feature shape: {tuple(all_feat32.shape)}")

    # Save results
    feat32_colnames = [f"feat32_{i + 1:02d}" for i in range(all_feat32.shape[1])]
    df_feat32 = pd.DataFrame(all_feat32.numpy(), columns=feat32_colnames)
    df_feat32.insert(0, "patient_id", all_patient_ids)

    OUTPUT_EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat32.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print(f"Saved features to: {OUTPUT_EXCEL_PATH}")


if __name__ == "__main__":
    main()
