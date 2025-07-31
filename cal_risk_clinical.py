# evaluate_deepsurv.py
# -----------------------------------------------------------
"""
加载训练好的 best_overall.pth，在“纯外部测试集”上
  • 计算 concordance index
  • 计算平均 Cox PH loss
"""

import os, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from torch import nn
from sklearn.model_selection import KFold
# ====== >>> 把这几个路径改成你的实际位置 <<< =========================
data_dir_test = "/Volumes/T7/WSI/features/test_new"   # 外部测试集目录
model_path    = "/Volumes/T7/WSI/features/train_new3/pth/best_overall.pth"
path_risk = "/Volumes/T7/WSI/features/test_new/risk_out.xlsx"
# ======================================================================

# ----------------- 损失函数（保持一致） -----------------
def cox_ph_loss(risk, time, event, eps=1e-8):
    risk  = risk.view(-1); time = time.view(-1); event = event.view(-1)
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    return (-(event * (risk - log_cum)).sum() / (event.sum() + eps))

# ----------------- 与训练脚本相同的模型 ----------------
class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim),
                                     nn.ReLU())
    def forward(self, x): return self.encoder(x)

class DeepSurvMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_SR = ModalityEncoder(1536, 768)
        self.MRI      = ModalityEncoder(1342, 768)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1536, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(128,  64),  torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight); torch.nn.init.zeros_(m.bias)

    def forward(self, wsi, mri):
        z1 = self.patch_SR(wsi); z2 = self.MRI(mri)
        x  = torch.cat([z1, z2], dim=1)
        return self.mlp(x).squeeze(-1)

# ----------------- Dataset（无抖动） -------------------
class MultimodalDataset(Dataset):
    def __init__(self, wsi, mri, surv):
        self.wsi  = torch.tensor(wsi.values,  dtype=torch.float32)
        self.mri  = torch.tensor(mri.values,  dtype=torch.float32)
        self.time  = torch.tensor(surv["time" ].values, dtype=torch.float32)
        self.event = torch.tensor(surv["event"].values, dtype=torch.float32)

    def __len__(self): return len(self.time)

    def __getitem__(self, idx):
        return (self.wsi[idx], self.mri[idx],
                self.time[idx], self.event[idx])

# ------------------------- 主流程 -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 读取外部测试特征
    wsi  = pd.read_excel(os.path.join(data_dir_test, "zd_pathology_features.xlsx"), index_col=0)
    mri  = pd.read_excel(os.path.join(data_dir_test, "zd_MRI_features.xlsx"),       index_col=0)
    surv = pd.read_excel(os.path.join(data_dir_test, "zd_PFS_survival.xlsx"),       index_col=0)
    print(f"▶ 外部测试集样本数: {len(surv)}")

    test_ds = MultimodalDataset(wsi, mri, surv)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)

    # 2. 加载模型权重
    model = DeepSurvMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✔ 已加载权重: {model_path}")

    # 3. 推理 & 指标
    risks, times, events = [], [], []
    with torch.no_grad():
        for wsi_b, mri_b, t_b, e_b in test_loader:
            wsi_b = wsi_b.to(device); mri_b = mri_b.to(device)
            r = model(wsi_b, mri_b).cpu()
            risks.append(r);  times.append(t_b);  events.append(e_b)

    risks  = torch.cat(risks)
    times  = torch.cat(times)
    events = torch.cat(events)

    c_index_test = concordance_index(times.numpy(), risks.numpy(), events.numpy())
    avg_cox_loss = cox_ph_loss(risks, times, events).item()

    print("\n=======  外部测试集评估  =======")

    df = pd.DataFrame(risks, columns=['risk'])  # ② 转成一列 DataFrame；可改列名
    # （可选）如果想把行索引改成 1, 2, 3…而不是默认 0,1,2…
    df.index += 1  # 行号从 1 开始
    df.to_excel(path_risk, index=False)
    print(f"C-index          : {c_index_test:.4f}")
    print(f"平均 Cox-PH Loss : {avg_cox_loss:.4f}")

if __name__ == "__main__":
    main()