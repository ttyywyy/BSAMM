# -*- coding: utf-8 -*-
"""
evaluate_deepsurv_with_64feat.py
--------------------------------
加载训练好模型，推理外部测试集
  • 输出 C-index / Cox-PH loss
  • 额外保存倒数第 2 层 64 维特征 (features64.xlsx)
"""
import math
import os, torch, pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
import sys

# ===== 路径自行替换 =====
data_dir_test = "//Volumes/T7/WSI/features/melting/NO_Aug/val"
model_path    = "/Volumes/T7/WSI/features/melting/NO_Aug/train/pth/best_overall.pth"
path_risk     = "/Volumes/T7/WSI/features/melting/NO_Aug/val/risk_out_val.xlsx"
# ==========================================================

# ---------- Cox-PH loss ----------
def cox_ph_loss(risk, time, event, eps=1e-8):
    risk, time, event = [x.view(-1) for x in (risk, time, event)]
    idx   = torch.argsort(time, descending=True)
    risk  = risk[idx]; event = event[idx]
    log_c = torch.logcumsumexp(risk, 0)
    return -(event * (risk - log_c)).sum() / (event.sum() + eps)


# ---------- 与训练脚本同结构的模型 ----------
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

        # self.apply(self._init_w)

    # @staticmethod
    # def _init_w(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self,  mri, wsi,ich,text):
        MRI_feat = self.MRI(mri)
        combined_tensor = torch.stack([MRI_feat, wsi, ich, text], dim=1)
        z1 = self.attention1(combined_tensor)
        #print(z1.shape)
        #z2 = self.attention2(z1)
        z3 = self.ln(z1).flatten(1)
        risk = self.for_head(z3)

        return risk

# ---------- Dataset ----------
class MultimodalDataset(Dataset):
    def __init__(self, mri, wsi, ich, text, surv):
        self.mri  = torch.tensor(mri.values,  dtype=torch.float32)
        self.wsi  = torch.tensor(wsi.values,  dtype=torch.float32)
        self.ich = torch.tensor(ich.values, dtype=torch.float32)
        self.text = torch.tensor(text.values, dtype=torch.float32)
        self.time  = torch.tensor(surv["time" ].values, dtype=torch.float32)
        self.event = torch.tensor(surv["event"].values, dtype=torch.float32)

    def __len__(self): return len(self.time)

    def __getitem__(self, idx):
        return (self.mri[idx], self.wsi[idx], self.ich[idx],
                self.text[idx],self.time[idx], self.event[idx])


# ---------------- 主流程 ----------------
def main():
    out_dir = '/Volumes/T7/WSI/features/train_new6/pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log_path = os.path.join(out_dir, "val_log.txt")
    # orig_stdout = sys.stdout
    # sys.stdout = open(log_path, "w", buffering=1)

    # 1. 读取外部测试集
    mri  = pd.read_excel(os.path.join(data_dir_test, "MRI_features.xlsx"), index_col=0)
    wsi  = pd.read_excel(os.path.join(data_dir_test, "wsi_features.xlsx"), index_col=0)
    ich = pd.read_excel(os.path.join(data_dir_test, "ich_features.xlsx"), index_col=0)
    text = pd.read_excel(os.path.join(data_dir_test, "clinical_features.xlsx"),index_col=0)
    surv = pd.read_excel(os.path.join(data_dir_test, "rs63_val.xlsx"), index_col=0)
    print(f"▶ 外部测试集样本数: {len(surv)}")



    ds_test = MultimodalDataset(mri, wsi, ich, text, surv)
    dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

    # 2. 加载模型
    model = Surv_attention().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✔ 已加载权重: {model_path}")

    # 3. 推理
    risks, times, events, feats64 = [], [], [], []
    with torch.no_grad():
        for mri_b,  wsi_b, ich_b, txt_b, t_b, e_b in dl_test:
            mri_b = mri_b.to(device); wsi_b = wsi_b.to(device); ich_b = ich_b.to(device); txt_b = txt_b.to(device)
            r= model(mri_b, wsi_b, ich_b, txt_b)
            risks.append(r.cpu()); times.append(t_b); events.append(e_b)

    risks   = torch.cat(risks).squeeze(1)      # (N,64)
    times   = torch.cat(times)
    events  = torch.cat(events)
    #print(risks.shape)
    # 4. 评估指标
    c_index_test = concordance_index(times.numpy(), -risks.numpy(), events.numpy())
    avg_cox_loss = cox_ph_loss(risks, times, events).item()

    # 5. 保存结果
    pd.DataFrame({"risk": risks.numpy()}).to_excel(path_risk, index=False)

    print("\n=======  外部测试集评估  =======")
    print(f"C-index          : {c_index_test:.4f}")
    print(f"平均 Cox-PH Loss : {avg_cox_loss:.4f}")
    print(f"risk 已保存至    : {path_risk}")

    # sys.stdout.close()  # 关闭文件句柄
    # sys.stdout = orig_stdout

if __name__ == "__main__":
    main()
