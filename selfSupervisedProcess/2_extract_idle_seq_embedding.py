"""
功能说明：
该脚本使用已训练好的 MaskedSeqModel（用于掩码预测），
对 uk_train.csv 中的闲时序列样本提取 Transformer 输出的全局平均池化作为嵌入向量，
每个样本得到一个 128 维嵌入 z_seq，保存为 _seq_embed.npy。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ===================== 归一化函数（与训练阶段一致） =====================
def normalize_features(feature_matrix):
    """
    对输入的 [pkt_len, iat, direction] 进行归一化处理：
    - pkt_len: log1p
    - iat: log1p
    - direction: [-1, 1] → [0, 1]
    """
    x = feature_matrix.copy()
    x[:, 0] = np.log1p(x[:, 0])  # pkt_len
    x[:, 1] = np.log1p(x[:, 1])  # iat
    x[:, 2] = (x[:, 2] + 1) / 2  # direction
    return x

# ------------------------- 模型结构（保持与训练阶段一致） -------------------------
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
        return x + self.pe[:, :x.size(1)]

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
        self.decoder = nn.Sequential(  # 注意：这里只是加载模型结构，不使用
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):  # 嵌入提取只用 transformer
        x = self.mlp(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x  # shape: [B, L, d]
    
# ===================== 数据加载类 =====================
class IdleSequenceDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df[df['is_behavior'] == 0].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        npz_path = row['seq_feature_path']
        feature_matrix = np.load(npz_path)['feature_matrix'].astype(np.float32)
        feature_matrix = normalize_features(feature_matrix)
        return feature_matrix, npz_path

# ------------------------- 嵌入保存 -------------------------
def save_embedding(embedding, npz_path_str, input_root: Path, output_root: Path):
    npz_path = Path(npz_path_str)
    relative_path = npz_path.relative_to(input_root)
    embed_name = relative_path.name.replace('_seq.npz', '_seq_embed.npy')
    save_path = output_root / relative_path.parent / embed_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embedding.cpu().numpy())

# ------------------------- 主函数 -------------------------
def main():
    csv_path = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/cicIoT2022/3_cicIoT2022_unknown.csv"    # 这里根据数据集修改train/test/unknown
    input_root = Path("/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_sequence_feature_matrix/cicIoT2022")
    output_root = Path("/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_sequence_embeddings/unknown/cicIoT2022")   # 这里也一样
    model_path = Path("/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_idle_seq_model.pt")

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型权重（只用前向 transformer）
    model = MaskedSeqModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = IdleSequenceDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"📦 共有闲时训练样本数: {len(dataset)}")

    with torch.no_grad():
        for feature_matrix, npz_path in tqdm(dataloader, desc="🧠 提取嵌入"):
            feature_matrix = feature_matrix.to(device)  # [1, L, 3]
            z = model(feature_matrix)                  # [1, L, d]
            z_seq = z.mean(dim=1).squeeze(0)           # [128]
            save_embedding(z_seq, npz_path[0], input_root, output_root)

    print("✅ 所有嵌入向量提取完成:", output_root)


if __name__ == "__main__":
    main()
