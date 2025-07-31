import os, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from torch import nn
# ====== >>> 把这几个路径改成你的实际位置 <<< =========================
#data_dir_test = "/Volumes/T7/WSI/features/test_new"   # 外部测试集目录
model_path = '/Volumes/T7/WSI/features/melting/FC/best_deepsurv.pth'
path_risk = "/Volumes/T7/WSI/features/melting/FC/train_risk_new.xlsx"
# ======================================================================
CONFIG = dict(
    HIDDEN_SIZES = (128, 64),   # MLP 结构
    DROPOUT      = 0.5,
    LR           = 1e-4,
    WEIGHT_DECAY = 1e-1,
    EPOCHS       = 500,
    BATCH_SIZE   = 256,
    TRAIN_RATIO  = 0.8,
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu",
    SEED         = 42,
)

# ----------------- 损失函数（保持一致） -----------------
def cox_ph_loss(risk, time, event, eps=1e-8):
    risk  = risk.view(-1); time = time.view(-1); event = event.view(-1)
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    return (-(event * (risk - log_cum)).sum() / (event.sum() + eps))

# ───────────── 3. DeepSurv 网络 (MLP) ─────────────
class DeepSurvMLP(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        sizes   = CONFIG["HIDDEN_SIZES"]
        dropout = CONFIG["DROPOUT"]

        layers, in_dim = [], num_features
        for h in sizes:
            layers += [nn.Linear(in_dim, h),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]        # 输出 1 个风险分数

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)          # shape -> (B,)

# ───────────── 4. 读取 CSV 的 Dataset ─────────────
class SurvivalDataset(Dataset):
    """
    期望：
      • X.csv  行=样本，列=数值特征（已做 one-hot / 标准化）
      • y.csv  列 = [id, time, event]
    """
    def __init__(self, x_path, y_path):
        self.x = torch.tensor(pd.read_csv(x_path, index_col=0).values,
                              dtype=torch.float32)
        ydf    = pd.read_csv(y_path, index_col=0)
        self.time  = torch.tensor(ydf["time" ].values, dtype=torch.float32)
        self.event = torch.tensor(ydf["event"].values, dtype=torch.float32)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.time[idx], self.event[idx]

# ───────────── 5. 训练 & 验证循环 ─────────────
def run_deepsurv(x_csv, y_csv):
    torch.manual_seed(CONFIG["SEED"])
    # 5.1 数据准备
    test_ds  = SurvivalDataset(x_csv, y_csv)
    feat_dim = test_ds.x.shape[1]


    test_loader = DataLoader(test_ds, batch_size=256,
                              shuffle=False,  num_workers=0)


    # 5.2 建立模型与优化器
    device = torch.device(CONFIG["DEVICE"])
    model  = DeepSurvMLP(num_features=feat_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for x,t,e in test_loader:
        risk = model(x)
        df = pd.DataFrame(risk.detach().numpy(), columns=['risk'])
        # 写入 Excel 文件
        df.to_excel(path_risk, index=False)
        loss = cox_ph_loss(risk, t, e)
        c_index_test = concordance_index(t.detach().numpy(), -risk.detach().numpy(), e.detach().numpy())
        print(loss,c_index_test)


run_deepsurv("/Volumes/T7/WSI/features/melting/FC/train_all_features_new.csv",
               "/Volumes/T7/WSI/features/melting/FC/train_PFS_new.csv")