import math
import torch
import torch.nn as nn
import os, warnings, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import random
warnings.filterwarnings("ignore")
import sys



JITTER_RANGE = (0, 0)

def cox_ph_loss(risk_score: torch.Tensor,
                time: torch.Tensor,
                event: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    risk  = risk_score.view(-1)
    time  = time.view(-1)
    event = event.view(-1)

    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cumsum  = torch.logcumsumexp(risk, dim=0)
    nll = -(event * (risk - log_cumsum)).sum()
    return nll / (event.sum() + eps)

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                     nn.ReLU())
    def forward(self, x): return self.encoder(x)


class self_att(nn.Module):
    def __init__(self,
                 input_dim:int = 768,
                 hidden_dim:int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(input_dim,hidden_dim,bias=False)
        self.key_proj = nn.Linear(input_dim, hidden_dim,bias=False)
        self.value_proj = nn.Linear(input_dim, hidden_dim,bias=False)


    def forward(self,x):
        Q = self.query_proj(x)
        #print(Q.shape)
        K = self.key_proj(x)
        #print(K.shape)
        V = self.value_proj(x)
        #print(V.shape)

        attent_value = torch.matmul(
            Q,K.transpose(-1,-2)
        )
        attent_weight = torch.softmax(
            attent_value/math.sqrt(self.hidden_dim),
            dim = -1
        )

        output = torch.matmul(attent_weight,V)
        return output


class Surv_attention(nn.Module):
    """ forward 返回 (risk, feat64) """
    def __init__(self):
        super().__init__()
        self.MRI = ModalityEncoder(1342, 768)
        self.attention1 = self_att(768,256)
        self.attention2 = self_att(256, 256)
        self.ln = nn.LayerNorm(256,eps=1e-5, elementwise_affine=True)
        self.for_head = nn.Sequential(
            nn.Linear(1024, 32,bias=False), nn.ReLU(), nn.Dropout(0.5),
            #nn.Linear(128,  32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32,  1,bias=False),
            nn.Sigmoid())

        #self.apply(self._init_w)

    @staticmethod
    def _init_w(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        # m.bias 可能是 None
            if m.bias is not None:          # ← 加这一行保护
                nn.init.zeros_(m.bias)

    def forward(self,  mri, wsi,ich,text):
        MRI_feat = self.MRI(mri)
        combined_tensor = torch.stack([MRI_feat, wsi, ich, text], dim=1)
        z1 = self.attention1(combined_tensor)
        #print(z1.shape)
        #z2 = self.attention2(z1)
        z3 = self.ln(z1).flatten(1)
        risk = self.for_head(z3)

        return risk



# ───────── Dataset ─────────
class MultimodalDataset(Dataset):
    """
    augment=True 时：
      1) concat 三模态 → vec
      2) vec += N(0, |J|)  (每 epoch 刷新 J 和随机种子)
      3) 再切回三段
    """
    def __init__(self, mri, wsi, ich, txt, surv,
                 augment=False, jr=0.0):
        self.mri  = torch.FloatTensor(mri.values)
        self.wsi  = torch.FloatTensor(wsi.values)
        self.ich = torch.FloatTensor(ich.values)
        self.txt  = torch.FloatTensor(txt.values)
        self.surv = surv
        self.aug, self.jr = augment, jr
        self.len_m, self.len_w, self.len_ii, self.len_t = self.mri.shape[1], self.wsi.shape[1], self.ich.shape[1], self.txt.shape[1]
        self.gen = torch.Generator()                       # 独立 RNG
    def set_jitter(self, jr, seed=None):
        """epoch 开头调用，更新抖动幅度 & 随机种子"""
        self.jr = jr
        if seed is not None:
            self.gen.manual_seed(int(seed))
    def __len__(self): return len(self.surv)
    def __getitem__(self, i):
        m, w, ii, t = self.mri[i], self.wsi[i], self.ich[i], self.txt[i]
        if self.aug and self.jr != 0:
            vec = torch.cat([m, w, ii, t])
            noise = torch.randn(vec.size(), generator=self.gen) * abs(self.jr)
            vec = vec + noise
            m = vec[:self.len_m]
            w = vec[self.len_m:self.len_m+self.len_w]
            ii = vec[self.len_w:self.len_w+self.len_ii]
            t = vec[-self.len_t:]
        time  = torch.tensor(self.surv.iloc[i]['time' ], dtype=torch.float32)
        event = torch.tensor(self.surv.iloc[i]['event'], dtype=torch.float32)
        return { "mri": m, "wsi": w, "ich": ii, "text": t, "time": time, "event": event}

def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

# ───────── 单折训练 ─────────
def run_fold(model_path,tr_idx, va_idx,
             mri, wsi,ich, txt, surv,
             dev, bs, epochs, lr, wd, outd):

    ds_tr = MultimodalDataset(mri.loc[tr_idx], wsi.loc[tr_idx],ich.loc[tr_idx],txt.loc[tr_idx],
                              surv.loc[tr_idx], augment=True)
    ds_va = MultimodalDataset(mri.loc[va_idx], wsi.loc[va_idx],ich.loc[va_idx],txt.loc[va_idx],
                              surv.loc[va_idx], augment=False)

    dl_tr = DataLoader(ds_tr, bs, shuffle=True,  collate_fn=collate)
    dl_va = DataLoader(ds_va, 32 , shuffle=False, collate_fn=collate)

    model = Surv_attention().to(dev)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)
    #opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    #opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    best_c, best_p = -1, f"{outd}/fold_best.pth"

    for ep in range(1, epochs+1):
        # —— ❶ 刷新抖动幅度 & 随机种子 ——
        jr   = random.uniform(*JITTER_RANGE)
        seed = random.randrange(10**9)
        ds_tr.set_jitter(jr, seed)

        # —— Train ——
        model.train(); loss_sum=0; r_tr,t_tr,e_tr=[],[],[]
        for b in dl_tr:
            for k,v in b.items():
                if torch.is_tensor(v): b[k]=v.to(dev)
            r= model(b['mri'],b['wsi'],b['ich'], b['text'])
            loss = cox_ph_loss(r,b['time'],b['event'])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum+=loss.item()
            r_tr.append(r.detach().cpu()); t_tr.append(b['time'].cpu()); e_tr.append(b['event'].cpu())
        loss_avg = loss_sum/len(dl_tr)
        train_c  = concordance_index(torch.cat(t_tr).numpy(),
                                     -torch.cat(r_tr).numpy(),
                                     torch.cat(e_tr).numpy())

        # —— Val ——
        model.eval(); r_va,t_va,e_va=[],[],[]
        with torch.no_grad():
            for b in dl_va:
                for k,v in b.items():
                    if torch.is_tensor(v): b[k]=v.to(dev)
                r_val = model( b['mri'],b['wsi'], b['ich'],b['text'])
                r_va.append(r_val)
                t_va.append(b['time']); e_va.append(b['event'])
        val_c = concordance_index(torch.cat(t_va).numpy(),
                                  -torch.cat(r_va).cpu().numpy(),
                                  torch.cat(e_va).numpy())
        if val_c>best_c:
            best_c=val_c; torch.save(model.state_dict(),best_p)

        print(f"[Fold Ep{ep:03d}] JR={jr:+.2f} "
              f"Loss={loss_avg:6.3f} Train-C={train_c:.4f} "
              f"Val-C={val_c:.4f} Best={best_c:.4f}")

    return best_c,best_p

# ───────── 主函数 ─────────
def main():
    data_dir="/Volumes/T7/WSI/features/train_new6"
    out_dir ="/Volumes/T7/WSI/features/train_new6/pth";os.makedirs(out_dir,exist_ok=True)
    model_path = '//Volumes/T7/WSI/features/train_new6/pth/best_overall.pth'
    bs,epochs,lr,wd = 128,500,1e-6,1
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(out_dir, "pretrained_log.txt")
    orig_stdout = sys.stdout
    sys.stdout = open(log_path, "w", buffering=1)

    mri = pd.read_excel(f"{data_dir}/MRI_features.xlsx",index_col=0)
    wsi = pd.read_excel(f"{data_dir}/wsi_features.xlsx",index_col=0)
    ich =  pd.read_excel(f"{data_dir}/ich_features.xlsx",index_col=0)
    txt = pd.read_excel(f"{data_dir}/Clinical_features.xlsx",index_col=0)
    surv = pd.read_excel(f"{data_dir}/PFS_survival.xlsx", index_col=0)
    surv_train= pd.read_excel(f"{data_dir}/rs63_train.xlsx",index_col=0)
    surv_val = pd.read_excel(f"{data_dir}/rs63_val.xlsx", index_col=0)
    print(f"样本量: {len(surv)} ▶ 开始 5-fold CV")

    #kf = KFold(n_splits=10, shuffle=True, random_state=63)
    best_c,best_path,fold_cs = -1,None,[]


    c,p = run_fold(model_path,surv_train.index,surv_val.index,
                   mri,wsi,ich,txt,surv,
                   dev,bs,epochs,lr,wd,out_dir)
    fold_cs.append(c)
    if c>best_c: best_c,best_path=c,p

    torch.save(torch.load(best_path),f"{out_dir}/best_overall_pretrained.pth")
    print("\n======= 5-Fold 结果 =======")
    for i,c in enumerate(fold_cs,1):
        print(f"Fold {i:2d}: Best Val-C = {c:.4f}")
    print(f"Overall best C-index = {best_c:.4f} "
          f"(权重已保存至 {out_dir}/best_overall_pretrained.pth)")


    sys.stdout.close()  # 关闭文件句柄
    sys.stdout = orig_stdout

if __name__ == "__main__":
    main()

