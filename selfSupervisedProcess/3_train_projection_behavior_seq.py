# train_projection_behavior_seq.py

"""
📌 功能说明：
本脚本对 UK 数据集中“行为流量的序列嵌入向量（128维）”进行对比学习训练，
训练一个投影模型（MLP），将其映射到更具判别力的64维嵌入空间，并保存为 .pt 文件。

特点：
- 输入数据：每台设备的行为流量序列嵌入（128维 .npy），路径结构保持不变。
- 输出模型：Projection MLP 的权重文件（.pt）。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ========================== 配置参数 ==========================
INPUT_ROOT = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_sequence_embeddings/train/cicIoT2022"
SAVE_MODEL_PATH = "/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_projection_behavior_seq.pt"
EMBEDDING_DIM = 128
PROJECTED_DIM = 64
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== Step 1: 数据集定义 ==========================
class TripletActivitySeqDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.device2files = {}
        for device in os.listdir(root):
            act_dir = os.path.join(root, device, "activity")
            if not os.path.isdir(act_dir): continue
            for behavior in os.listdir(act_dir):
                beh_dir = os.path.join(act_dir, behavior)
                for file in os.listdir(beh_dir):
                    if file.endswith("_seq_embed.npy"):
                        path = os.path.join(beh_dir, file)
                        self.samples.append((path, device))
                        self.device2files.setdefault(device, []).append(path)
        print(f"✅ 加载行为序列嵌入样本: {len(self.samples)}，设备数: {len(self.device2files)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_dev = self.samples[idx]
        anchor = torch.tensor(np.load(anchor_path), dtype=torch.float32)

        # positive（同设备）
        pos_path = anchor_path
        while pos_path == anchor_path:
            pos_path = random.choice(self.device2files[anchor_dev])
        pos = torch.tensor(np.load(pos_path), dtype=torch.float32)

        # negative（不同设备）
        neg_dev = random.choice([d for d in self.device2files if d != anchor_dev])
        neg_path = random.choice(self.device2files[neg_dev])
        neg = torch.tensor(np.load(neg_path), dtype=torch.float32)

        return anchor, pos, neg

# ========================== Step 2: 投影网络定义 ==========================
class ProjectionMLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ========================== Step 3: 对比损失函数 ==========================
def nt_xent_loss(z1, z2, z3, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    z3 = nn.functional.normalize(z3, dim=1)
    sim_pos = torch.sum(z1 * z2, dim=1) / temperature
    sim_neg = torch.sum(z1 * z3, dim=1) / temperature
    logits = torch.stack([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(z1.size(0), dtype=torch.long).to(z1.device)
    return nn.CrossEntropyLoss()(logits, labels)

# ========================== Step 4: 训练主流程 ==========================
def train():
    dataset = TripletActivitySeqDataset(INPUT_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = ProjectionMLP(EMBEDDING_DIM, PROJECTED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("🚀 开始训练投影模型（行为序列嵌入）...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for anchor, pos, neg in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            z1 = model(anchor)
            z2 = model(pos)
            z3 = model(neg)
            loss = nt_xent_loss(z1, z2, z3, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"🎉 模型保存成功: {SAVE_MODEL_PATH}")

# ========================== 主程序入口 ==========================
if __name__ == "__main__":
    train()
