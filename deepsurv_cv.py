#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💡 DeepSurv（PyTorch 版）+ 5倍交叉验证
--------------------------------
• 用全连接 MLP 替换 Cox 回归的线性项
• 损失仍是负对数部分似然 (Cox PH loss)
• 评价指标使用 concordance index (C-index)
• 新增：5倍交叉验证功能
"""

# ───────────── 0. 依赖包 ─────────────
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ───────────── 1. 全局超参数 (一处修改，全局生效) ─────────────
CONFIG = dict(
    HIDDEN_SIZES=(128, 64),  # MLP 结构
    DROPOUT=0.4,
    LR=1e-4,
    WEIGHT_DECAY=1e-5,
    EPOCHS=100,  # 减少epochs以适应交叉验证
    BATCH_SIZE=64,
    TRAIN_RATIO=0.9,  # 在交叉验证中，这个参数用于训练集内部的验证
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",
    SEED=42,
    # 交叉验证参数
    CV_FOLDS=10,  # 5倍交叉验证
    PATIENCE=50,  # 早停耐心
    MIN_DELTA=0.001,  # 最小改善阈值
)

1
# ───────────── 2. 数值稳定的 Cox-PH 损失 ─────────────
def cox_ph_loss(risk_score: torch.Tensor,
                time: torch.Tensor,
                event: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """
    risk_score : (N,) 网络输出的对数风险
    time       : (N,) 生存 / 随访时间
    event      : (N,) 1=事件，0=删失
    """
    # 统一展平
    risk = risk_score.view(-1)
    time = time.view(-1)
    event = event.view(-1)

    # ① 先按时间降序排列  (Risk set = t_j >= t_i)
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]

    # ② log 累加求和：log Σ_{j≥i} e^{risk_j}
    log_cumsum = torch.logcumsumexp(risk, dim=0)

    # ③ 负对数似然 (只对事件样本求和)
    nll = -(event * (risk - log_cumsum)).sum()

    # ④ 用事件数归一化，防止全删失时 /0
    return nll / (event.sum() + eps)


# ───────────── 3. DeepSurv 网络 (MLP) ─────────────
class DeepSurvMLP(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        sizes = CONFIG["HIDDEN_SIZES"]
        dropout = CONFIG["DROPOUT"]

        layers, in_dim = [], num_features
        for h in sizes:
            layers += [nn.Linear(in_dim, h),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # 输出 1 个风险分数

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # shape -> (B,)


# ───────────── 4. 读取 CSV 的 Dataset ─────────────
class SurvivalDataset(Dataset):
    """
    期望：
      • X.csv  行=样本，列=数值特征（已做 one-hot / 标准化）
      • y.csv  列 = [id, time, event]
    """

    def __init__(self, x_path: str, y_path: str):
        self.x = torch.tensor(pd.read_csv(x_path, index_col=0).values,
                              dtype=torch.float32)
        ydf = pd.read_csv(y_path, index_col=0)
        self.time = torch.tensor(ydf["time"].values, dtype=torch.float32)
        self.event = torch.tensor(ydf["event"].values, dtype=torch.float32)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.time[idx], self.event[idx]


# ───────────── 5. 早停机制 ─────────────
# class EarlyStopping:
#     """早停机制"""
#
#     def __init__(self, patience=50, min_delta=0.001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_score = None
#
#     def __call__(self, val_score):
#         if self.best_score is None:
#             self.best_score = val_score
#         elif val_score < self.best_score + self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         else:
#             self.best_score = val_score
#             self.counter = 0
#         return False


# ───────────── 6. 单折训练函数 ─────────────
def train_single_fold(train_dataset: Dataset,
                      val_dataset: Dataset,
                      fold_idx: int,
                      verbose: bool = True) -> Tuple[float, Dict]:
    """
    训练单个折叠

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        fold_idx: 折叠索引
        verbose: 是否打印详细信息

    Returns:
        best_c_index: 最佳C-index
        history: 训练历史
    """

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1024,
                            shuffle=False, num_workers=0)

    # 模型和优化器
    device = torch.device(CONFIG["DEVICE"])
    feat_dim = train_dataset[0][0].shape[0]
    model = DeepSurvMLP(num_features=feat_dim).to(device)
    optim = torch.optim.Adam(model.parameters(),
                             lr=CONFIG["LR"],
                             weight_decay=CONFIG["WEIGHT_DECAY"])

    # 早停
    # early_stopping = EarlyStopping(patience=CONFIG["PATIENCE"],
    #                                min_delta=CONFIG["MIN_DELTA"])

    # 训练历史
    history = {
        'train_loss': [],
        'val_c_index': [],
        'epochs': 0
    }

    best_c = 0.0

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        # ─── Train ───
        model.train()
        train_loss = 0.
        for x, t, e in train_loader:
            x, t, e = (x.to(device), t.to(device), e.to(device))
            risk = model(x)
            loss = cox_ph_loss(risk, t, e)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ─── Validate ───
        model.eval()
        with torch.no_grad():
            risks, times, events = [], [], []
            for x, t, e in val_loader:
                r = model(x.to(device)).cpu()
                risks.append(r)
                times.append(t)
                events.append(e)
            risks = torch.cat(risks).numpy()
            times = torch.cat(times).numpy()
            events = torch.cat(events).numpy()
            c_val = concordance_index(times, -risks, events)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_c_index'].append(c_val)
        history['epochs'] = epoch

        # 保存最优
        if c_val > best_c:
            best_c = c_val
            # 保存当前折叠的最佳模型
            torch.save(model.state_dict(), f"best_deepsurv_fold_{fold_idx}.pth")

        if verbose and (epoch % 100 == 0 or epoch <= 10):
            print(f"  Fold {fold_idx} - Epoch [{epoch:03d}/{CONFIG['EPOCHS']}]  "
                  f"train loss {train_loss:6.3f} | val C-index {c_val:5.3f}")

        # 早停检查
        # if early_stopping(c_val):
        #     if verbose:
        #         print(f"  Fold {fold_idx} - 早停于第 {epoch} 轮")
        #     break

    return best_c, history


# ───────────── 7. 交叉验证主函数 ─────────────
def train_deepsurv_cv(x_csv: str, y_csv: str, output_dir: str = "./cv_results"):
    """
    5倍交叉验证训练DeepSurv

    Args:
        x_csv: 特征文件路径
        y_csv: 标签文件路径
        output_dir: 输出目录
    """

    print("=" * 60)
    print("🚀 DeepSurv 5倍交叉验证训练")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载完整数据集
    full_dataset = SurvivalDataset(x_csv, y_csv)
    n_samples = len(full_dataset)
    feat_dim = full_dataset[0][0].shape[0]

    print(f"📊 数据信息:")
    print(f"  样本数量: {n_samples}")
    print(f"  特征维度: {feat_dim}")
    print(f"  事件率: {full_dataset.event.mean():.3f}")
    print(f"  设备: {CONFIG['DEVICE']}")

    # 5倍交叉验证
    kfold = KFold(n_splits=CONFIG["CV_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])

    # 存储结果
    cv_results = {
        'fold_c_indices': [],
        'fold_histories': [],
        'fold_train_sizes': [],
        'fold_val_sizes': []
    }

    print(f"\n🔄 开始 {CONFIG['CV_FOLDS']} 倍交叉验证...")

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(n_samples))):
        print(f"\n📁 训练第 {fold_idx + 1} 折...")
        print(f"  训练集大小: {len(train_indices)}")
        print(f"  验证集大小: {len(val_indices)}")

        # 创建当前折叠的数据集
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # 训练当前折叠
        fold_c_index, fold_history = train_single_fold(
            train_dataset, val_dataset, fold_idx + 1, verbose=True
        )

        # 保存结果
        cv_results['fold_c_indices'].append(fold_c_index)
        cv_results['fold_histories'].append(fold_history)
        cv_results['fold_train_sizes'].append(len(train_indices))
        cv_results['fold_val_sizes'].append(len(val_indices))

        print(f"  ✅ 第 {fold_idx + 1} 折完成，最佳 C-index: {fold_c_index:.4f}")

    # 计算交叉验证统计
    cv_c_indices = np.array(cv_results['fold_c_indices'])
    mean_c_index = cv_c_indices.mean()
    std_c_index = cv_c_indices.std()

    print(f"\n" + "=" * 60)
    print("📈 交叉验证结果汇总")
    print("=" * 60)
    print(f"各折 C-index: {cv_c_indices}")
    print(f"平均 C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")
    print(f"最佳 C-index: {cv_c_indices.max():.4f}")
    print(f"最差 C-index: {cv_c_indices.min():.4f}")

    # 保存结果到文件
    results_df = pd.DataFrame({
        'Fold': range(1, CONFIG['CV_FOLDS'] + 1),
        'C_Index': cv_c_indices,
        'Train_Size': cv_results['fold_train_sizes'],
        'Val_Size': cv_results['fold_val_sizes'],
        'Epochs': [h['epochs'] for h in cv_results['fold_histories']]
    })
    results_df.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)

    # 绘制结果图表
    plot_cv_results(cv_results, output_dir, mean_c_index, std_c_index)

    # 保存配置
    config_df = pd.DataFrame([CONFIG]).T
    config_df.columns = ['Value']
    config_df.to_csv(os.path.join(output_dir, 'config.csv'))

    print(f"📁 结果已保存到: {output_dir}")

    return {
        'mean_c_index': mean_c_index,
        'std_c_index': std_c_index,
        'fold_c_indices': cv_c_indices,
        'fold_histories': cv_results['fold_histories'],
        'config': CONFIG
    }


# ───────────── 8. 结果可视化 ─────────────
def plot_cv_results(cv_results: Dict, output_dir: str, mean_c: float, std_c: float):
    """绘制交叉验证结果"""

    plt.figure(figsize=(20, 12))

    # 1. C-index 柱状图
    plt.subplot(2, 3, 1)
    fold_indices = range(1, len(cv_results['fold_c_indices']) + 1)
    bars = plt.bar(fold_indices, cv_results['fold_c_indices'],
                   color='skyblue', alpha=0.7, edgecolor='navy')
    plt.axhline(y=mean_c, color='red', linestyle='--',
                label=f'平均: {mean_c:.4f}±{std_c:.4f}')
    plt.xlabel('折叠')
    plt.ylabel('C-index')
    plt.title('各折 C-index 对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, c_idx in zip(bars, cv_results['fold_c_indices']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{c_idx:.3f}', ha='center', va='bottom', fontsize=10)

    # 2. 训练曲线（所有折叠）
    plt.subplot(2, 3, 2)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, history in enumerate(cv_results['fold_histories']):
        plt.plot(history['val_c_index'], color=colors[i % len(colors)],
                 alpha=0.7, label=f'Fold {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('验证 C-index')
    plt.title('训练过程 - 验证 C-index')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 损失曲线（所有折叠）
    plt.subplot(2, 3, 3)
    for i, history in enumerate(cv_results['fold_histories']):
        plt.plot(history['train_loss'], color=colors[i % len(colors)],
                 alpha=0.7, label=f'Fold {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('训练损失')
    plt.title('训练过程 - 训练损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. C-index 分布箱线图
    plt.subplot(2, 3, 4)
    plt.boxplot(cv_results['fold_c_indices'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.ylabel('C-index')
    plt.title('C-index 分布')
    plt.grid(True, alpha=0.3)

    # 5. 训练集/验证集大小
    plt.subplot(2, 3, 5)
    x = np.arange(len(fold_indices))
    width = 0.35
    plt.bar(x - width / 2, cv_results['fold_train_sizes'], width,
            label='训练集', alpha=0.7, color='lightcoral')
    plt.bar(x + width / 2, cv_results['fold_val_sizes'], width,
            label='验证集', alpha=0.7, color='lightblue')
    plt.xlabel('折叠')
    plt.ylabel('样本数量')
    plt.title('各折数据集大小')
    plt.xticks(x, fold_indices)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 统计摘要
    plt.subplot(2, 3, 6)
    stats_text = [
        f"DeepSurv 5倍交叉验证结果",
        f"",
        f"平均 C-index: {mean_c:.4f}",
        f"标准差: {std_c:.4f}",
        f"最佳: {max(cv_results['fold_c_indices']):.4f}",
        f"最差: {min(cv_results['fold_c_indices']):.4f}",
        f"",
        f"模型配置:",
        f"隐藏层: {CONFIG['HIDDEN_SIZES']}",
        f"Dropout: {CONFIG['DROPOUT']}",
        f"学习率: {CONFIG['LR']}",
        f"批次大小: {CONFIG['BATCH_SIZE']}",
        f"最大轮数: {CONFIG['EPOCHS']}",
        f"",
        f"性能评估:",
        f"{'优秀' if mean_c > 0.75 else '良好' if mean_c > 0.65 else '一般'} "
        f"(C-index {mean_c:.3f})",
        f"稳定性: {'高' if std_c < 0.05 else '中' if std_c < 0.1 else '低'} "
        f"(std {std_c:.3f})"
    ]

    plt.text(0.05, 0.95, '\n'.join(stats_text), fontsize=10,
             verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    plt.axis('off')
    plt.title('统计摘要')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_results_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 结果图表已保存: {os.path.join(output_dir, 'cv_results_analysis.png')}")


# ───────────── 9. 原始训练函数（保持兼容性） ─────────────
def train_deepsurv(x_csv: str, y_csv: str):
    """原始训练函数（无交叉验证）"""
    torch.manual_seed(CONFIG["SEED"])

    # 数据准备
    full_ds = SurvivalDataset(x_csv, y_csv)
    feat_dim = full_ds.x.shape[1]
    n_train = int(CONFIG["TRAIN_RATIO"] * len(full_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(CONFIG["SEED"])
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1024,
                            shuffle=False, num_workers=0)

    # 建立模型与优化器
    device = torch.device(CONFIG["DEVICE"])
    model = DeepSurvMLP(num_features=feat_dim).to(device)
    optim = torch.optim.Adam(model.parameters(),
                             lr=CONFIG["LR"],
                             weight_decay=CONFIG["WEIGHT_DECAY"])

    best_c = 0.0
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        # ─── Train ───
        model.train()
        train_loss = 0.
        for x, t, e in train_loader:
            x, t, e = (x.to(device), t.to(device), e.to(device))
            risk = model(x)
            loss = cox_ph_loss(risk, t, e)
            optim.zero_grad();
            loss.backward();
            optim.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ─── Validate ───
        model.eval()
        with torch.no_grad():
            risks, times, events = [], [], []
            for x, t, e in val_loader:
                r = model(x.to(device)).cpu()
                risks.append(r);
                times.append(t);
                events.append(e)
            risks = torch.cat(risks).numpy()
            times = torch.cat(times).numpy()
            events = torch.cat(events).numpy()
            c_val = concordance_index(times, -risks, events)

        # 保存最优
        if c_val > best_c:
            best_c = c_val
            torch.save(model.state_dict(), "best_deepsurv.pth")

        print(f"[{epoch:02d}/{CONFIG['EPOCHS']}]  "
              f"train loss {train_loss:6.3f} | val C-index {c_val:5.3f}")

    print("✅ Training done.  Best validation C-index =", best_c)


# ───────────── 10. 入口 ─────────────
if __name__ == "__main__":
    # 文件路径
    x_csv = "/Users/ouyouyou/Desktop/residual_PA/yanshao/new_fusion/MRI_features.csv"
    y_csv = "/Users/ouyouyou/Desktop/residual_PA/yanshao/new_fusion/PFS_survival.csv"

    print("选择训练模式:")
    print("1. 5倍交叉验证训练 (推荐)")
    print("2. 传统单次训练")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        # 5倍交叉验证训练
        results = train_deepsurv_cv(x_csv, y_csv, output_dir="./deepsurv_cv_results")
        print(f"\n🎉 交叉验证完成！")
        print(f"平均 C-index: {results['mean_c_index']:.4f} ± {results['std_c_index']:.4f}")
    elif choice == "2":
        # 传统训练
        train_deepsurv(x_csv, y_csv)
    else:
        print("无效选择，默认使用交叉验证训练")
        results = train_deepsurv_cv(x_csv, y_csv, output_dir="./deepsurv_cv_results")
        print(f"\n🎉 交叉验证完成！")
        print(f"平均 C-index: {results['mean_c_index']:.4f} ± {results['std_c_index']:.4f}")