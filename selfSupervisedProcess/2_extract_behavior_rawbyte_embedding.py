# extract_behavior_rawbyte_embedding.py

"""
📌 功能说明：
该脚本使用训练好的 ByteModel，提取每个行为样本的全局嵌入向量 z_raw ∈ ℝ^128，
方法为：输入原始字节特征矩阵 [256, 128]，经过 ByteEmbedding → CNN → Transformer 后，
对输出进行 mean pooling 得到样本级嵌入向量。
结果保存为 xxx_raw_embed.npy，与原始样本路径结构一致。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---------------- 模型结构 ----------------
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

class ByteModel(nn.Module):
    def __init__(self, vocab_size=257, embed_dim=64, cnn_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = BytePositionEmbedding(max_len=128, d_model=embed_dim)
        self.cnn = nn.Conv1d(embed_dim, cnn_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(cnn_dim, vocab_size - 1)  # ⚠️结构保持一致但不使用

    def forward(self, x):  # [B, N, L]
        B, N, L = x.shape
        x = self.byte_embed(x)                  # [B, N, L, d]
        x = self.pos_embed(x)                   # 加位置编码
        x = x.view(B * N, L, -1).transpose(1, 2) # [B*N, d, L]
        x = self.relu(self.cnn(x))              # [B*N, c, L]
        x = x.transpose(1, 2)                   # [B*N, L, c]
        x = self.transformer(x)                 # [B*N, L, c]
        x = x.mean(dim=1)                       # 每个包→全局
        return x.view(B, N, -1).mean(dim=1)     # 所有包聚合 → [B, 128]

# ---------------- 数据集 ----------------
class RawByteFeatureDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df[df["is_behavior"] == 1].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw = np.load(row["raw_feature_path"])["raw_matrix"].astype(np.int64)
        return torch.tensor(raw, dtype=torch.long), row["raw_feature_path"]

# ---------------- 嵌入保存 ----------------
def save_embedding(embedding, raw_path, input_root: Path, output_root: Path):
    relative_path = Path(raw_path).relative_to(input_root)
    embed_name = relative_path.name.replace("_raw.npz", "_raw_embed.npy")
    save_path = output_root / relative_path.parent / embed_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embedding.cpu().numpy())

# ---------------- 主函数 ----------------
def main():
    csv_path = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/cicIoT2022/3_cicIoT2022_unknown.csv"    # 这里根据数据集修改train/test/unknown
    input_root = Path("/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_rawByte_feature_matrix/cicIoT2022")
    output_root = Path("/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_rawbyte_embeddings/unknown/cicIoT2022")    # 这里也一样
    model_path = "/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_behavior_rawbyte_model.pt"

    dataset = RawByteFeatureDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ByteModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"🔍 开始提取行为流量嵌入向量，总样本数: {len(dataset)}")

    with torch.no_grad():
        for raw_tensor, raw_path in tqdm(dataloader, desc="提取嵌入"):
            raw_tensor = raw_tensor.to(device)         # [1, 256, 128]
            z_raw = model(raw_tensor).squeeze(0)       # [128]
            save_embedding(z_raw, raw_path[0], input_root, output_root)

    print(f"✅ 行为样本嵌入提取完成，保存在: {output_root}")

if __name__ == "__main__":
    main()
