# train_idle_rawbyte_masked_model.py

"""
🔧 训练数据：
来自 uk_train.csv 中 is_behavior==0 的样本；每个样本为一个原始字节特征矩阵 [128, 128]，表示 128 个数据包，每包截取前 128 字节，值为 0~255 的整数。
📐 模型结构：
ByteEmbedding（词嵌入层）
将每个字节（0~255 及 [MASK]=256）映射。
PositionEmbedding（新增）
加入位置编码，增强 Transformer 对字节顺序的感知。
ByteCNN
卷积提取每个包中的局部字段组合特征（例如协议头部特征）。
Transformer Encoder
对包级特征建模包与包之间的上下文关系。
Decoder
对 Transformer 输出的每个字节位置，预测其原始字节 ID（0~255）。
学习率调度器（StepLR）
每 5 个 epoch 学习率减半，提升训练稳定性。
🧠 训练目标：
通过掩码机制遮掩部分字节，使用交叉熵损失预测原始字节值；
学习能够理解跨字节和跨包上下文的表示模型。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ------------------------ 配置参数 ------------------------
csv_path = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/cicIoT2022/3_cicIoT2022_train.csv"
raw_feature_root = Path("/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_rawByte_feature_matrix/cicIoT2022")
save_model_path = Path("/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_idle_rawbyte_model.pt")
log_file = Path("/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/train_logs/cicIoT2022_idle_rawbyte_train.log")

batch_size = 16
num_epochs = 20
lr = 1e-3
mask_ratio = 0.15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ 日志函数 ------------------------
def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - {msg}\n")

# ------------------------ 数据集类 ------------------------
class IdleRawByteDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df[df["is_behavior"] == 0].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        data = np.load(row["raw_feature_path"])
        matrix = data["raw_matrix"].astype(np.int64)
        return torch.tensor(matrix, dtype=torch.long)  # [128, 128]

# ------------------------ 掩码策略 ------------------------
def byte_mask(batch, mask_ratio=0.15, vocab_size=256):
    B, N, L = batch.shape
    mask = torch.rand(B, N, L) < mask_ratio
    masked = batch.clone()
    rand = torch.rand(B, N, L)
    masked[mask & (rand < 0.8)] = vocab_size       # [MASK] token = 256
    masked[mask & (rand >= 0.8) & (rand < 0.9)] = torch.randint(0, vocab_size, size=(1,)).item()
    return masked, mask

# ------------------------ Position Embedding ------------------------
class BytePositionEmbedding(nn.Module):
    def __init__(self, max_len=128, d_model=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)  # [1, L, d]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ------------------------ 模型定义 ------------------------
class ByteModel(nn.Module):
    def __init__(self, vocab_size=257, embed_dim=64, cnn_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = BytePositionEmbedding(max_len=128, d_model=embed_dim)
        self.cnn = nn.Conv1d(embed_dim, cnn_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(cnn_dim, vocab_size - 1)

    def forward(self, x):  # x: [B, N, L]
        B, N, L = x.shape
        x = self.byte_embed(x)               # [B, N, L, d]
        x = self.pos_embed(x)                # 加入位置编码
        x = x.view(B * N, L, -1).transpose(1, 2)  # [B*N, d, L]
        x = self.relu(self.cnn(x))           # [B*N, c, L]
        x = x.transpose(1, 2)                # [B*N, L, c]
        x = self.transformer(x)              # [B*N, L, c]
        logits = self.decoder(x)             # [B*N, L, 256]
        return logits.view(B, N, L, -1)      # [B, N, L, 256]

# ------------------------ 训练主函数 ------------------------
def train():
    dataset = IdleRawByteDataset(csv_path)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = ByteModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    log(f"📚 开始训练：总样本 {len(dataset)}，训练集 {train_size}，验证集 {val_size}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            masked, mask = byte_mask(batch, mask_ratio)
            masked, mask = masked.to(device), mask.to(device)
            logits = model(masked)

            pred = logits[mask]         # [M, 256]
            target = batch[mask]        # [M]

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        log(f"📉 Epoch {epoch+1} 训练损失: {avg_train_loss:.6f}")

        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                masked, mask = byte_mask(batch, mask_ratio)
                masked, mask = masked.to(device), mask.to(device)
                logits = model(masked)
                pred = logits[mask]
                target = batch[mask]
                val_loss += criterion(pred, target).item()
            avg_val_loss = val_loss / len(val_loader)
            log(f"🧪 Epoch {epoch+1} 验证损失: {avg_val_loss:.6f}")

        scheduler.step()

    torch.save(model.state_dict(), save_model_path)
    log(f"✅ 模型训练完成，权重已保存至: {save_model_path}")

if __name__ == "__main__":
    train()
