"""
功能说明：
该脚本用于训练一个自监督模型（Transformer Encoder），用于闲时流量的序列特征建模。
训练目标为掩码预测任务（Masked Sequence Modeling, MSM），输入为 [pkt_len, iat, direction] 组成的多通道序列，
通过 MLP 投影、位置编码、Transformer 编码器进行建模，最终训练后保存模型权重，可用于后续嵌入向量提取。
输入数据来源于 uk_train.csv 中 is_behavior=0 的训练样本（即闲时样本）。
输出模型保存为：idle_seq_model.pt
输出日志包含损失信息和样本处理进度。
新增：对输入样本进行归一化处理，确保与嵌入阶段一致。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 设置只使用 GPU 1

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- 参数配置 -------------------------
csv_path = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/cicIoT2022/3_cicIoT2022_train.csv"
seq_feature_root = Path("/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_sequence_feature_matrix/cicIoT2022")
save_model_path = Path("/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_idle_seq_model.pt")

log_dir = Path("/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/train_logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "cicIoT2022_idle_seq_train.log"

batch_size = 32
num_epochs = 20
lr = 1e-3
mask_ratio = 0.15  # 15% 掩码比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- 日志函数 -------------------------
def log(msg: str):
    timestamped = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    print(timestamped)
    with open(log_file, "a") as f:
        f.write(timestamped + "\n")

# 若需要每次训练清空旧日志，可取消注释以下语句
# open(log_file, 'w').close()

# ------------------------- 特征归一化函数 -------------------------
def normalize_features(feature_matrix):
    pkt_len = feature_matrix[:, 0]
    iat = feature_matrix[:, 1]
    direction = feature_matrix[:, 2]

    # 合理归一化策略
    pkt_len = np.log1p(pkt_len) / 8.0         # 常见 MTU 上限为 ~1500，log1p 压缩再除以8可映射到[0,1)
    iat = iat / 1.0                            # 假设最大间隔在1秒内，保留原始尺度
    direction = (direction + 1) / 2.0          # 映射 [-1, +1] → [0, 1]

    return np.stack([pkt_len, iat, direction], axis=1).astype(np.float32)

# ------------------------- 数据集定义 -------------------------
class IdleMaskedDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df[df["is_behavior"] == 0].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature_matrix = np.load(row["seq_feature_path"])["feature_matrix"]
        feature_matrix = normalize_features(feature_matrix)
        return feature_matrix

# ------------------------- 模型定义 -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MaskedSeqModel(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, trans_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x, mask):
        x_embed = self.mlp(x)
        x_pos = self.pos_encoder(x_embed)
        x_trans = self.transformer(x_pos)
        x_masked = x_trans[mask]
        out = self.decoder(x_masked)
        return out

# ------------------------- 掩码函数 -------------------------
def random_mask(x, mask_ratio=0.15):
    B, L, D = x.shape
    mask = torch.rand(B, L) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask

# ------------------------- 训练主流程 -------------------------
def train():
    dataset = IdleMaskedDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MaskedSeqModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    log(f"📚 开始训练，总样本数: {len(dataset)}, 批大小: {batch_size}, 总轮数: {num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            x_masked, mask = random_mask(batch, mask_ratio=mask_ratio)
            pred = model(x_masked, mask)
            target = batch[mask]
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        log(f"📉 Epoch {epoch+1} 平均损失: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_model_path)
    log(f"✅ 模型训练完成，权重已保存: {save_model_path}")

if __name__ == "__main__":
    train()
