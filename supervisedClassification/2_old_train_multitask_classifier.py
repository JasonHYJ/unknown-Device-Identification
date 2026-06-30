# train_multitask_classifier.py
# 📆 多任务IoT设备识别经典培训脚本

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from multimodal_dataset import MultiModalIoTDataset
from tqdm import tqdm

"""
（该脚本为老脚本，暂不使用，统计特征没有进行归一化，最老的版本）
脚本功能说明：
读取阶段一提取的统计特征、阶段二对比学习后的64维嵌入向量（序列+原始字节，区分闲时/行为）；
拼接成 287维 特征向量，进行门控融合（128维闲时 vs. 128维行为）；
输入 Transformer 结构进行学习；
输出三个分类器预测：
设备类型；
厂商；
型号；
使用多任务加权损失进行联合训练；
最终将训练好的模型保存到：/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/multitask_model.pt

模型结构说明：
输入向量：     [stat(31), idle_embed(128), behavior_embed(128)]  → 共287维
门控机制：     gate = sigmoid(W * is_behavior + b)
融合嵌入：     weighted_embed = gate * behavior + (1 - gate) * idle
拼接输入：     [stats, weighted_embed] → 输入 Transformer
Transformer： 编码行为/闲时流量的深层语义信息
输出层：       三个独立的 softmax 头 → type, brand, device
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --------------------------- 参数配置 ---------------------------
csv_path = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk/uk_train.csv"
root_stat = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature/uk"
root_seq = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings/uk"
root_raw = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings/uk"
label_dict_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk"

output_model_path = "/home/hyj/unknownDeviceIdentification/supervisedClassification/multitask_model.pt"

batch_size = 64
num_epochs = 40
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- 模型定义 ---------------------------
class MultiTaskClassifier(nn.Module):
    """
    多任务分类模型结构：
    - 输入为拼接后的287维特征（统计+闲时嵌入+行为嵌入）
    - 通过门控网络根据 is_behavior 控制行为/闲时特征加权融合（得128维）
    - 与统计特征拼接后（共159维），通过Transformer编码器提取全局表示
    - 并行输出三个预测结果：设备类型、厂商、具体型号
    """
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        # 门控机制：用于根据 is_behavior 值动态选择使用行为或闲时的嵌入特征
        self.gate = nn.Sequential(
            nn.Linear(1, 1),  # is_behavior --> gate value
            nn.Sigmoid()
        )
        self.fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier_type = nn.Linear(hidden_dim, num_type)
        self.classifier_brand = nn.Linear(hidden_dim, num_brand)
        self.classifier_device = nn.Linear(hidden_dim, num_device)

    def forward(self, x):
        # x: (B, 287)
        stats = x[:, :31]           # (B, 31)，统计特征，包括 is_behavior
        idle_beh = x[:, 31:]        # (B, 256)，拼接的闲时+行为嵌入向量
        idle_embed = idle_beh[:, :128]      # 闲时嵌入特征（64+64）
        behavior_embed = idle_beh[:, 128:]  # 行为嵌入特征（64+64）
        is_behavior = stats[:, 0:1]         # (B, 1)，is_behavior 标志位用于控制门控输出

        gate = self.gate(is_behavior)  # 根据 is_behavior 生成门控权重（接近0或1）
        weighted_embed = gate * behavior_embed + (1 - gate) * idle_embed  # (B, 128)

        # 拼接统计特征和加权后的嵌入向量，作为 Transformer 输入
        # unsqueeze(1) 添加时间维度，使其符合 Transformer 的 batch_first 输入格式 (B, T=1, D=159)
        combined = torch.cat([stats, weighted_embed], dim=1).unsqueeze(1)  # (B, 1, 159)

        x = self.fc(combined)  # (B, 1, hidden)
        x = self.encoder(x).squeeze(1)  # (B, hidden)

        out_type = self.classifier_type(x)
        out_brand = self.classifier_brand(x)
        out_device = self.classifier_device(x)
        return out_type, out_brand, out_device

# --------------------------- 训练逻辑 ---------------------------
def train():
    dataset = MultiModalIoTDataset(csv_path, root_stat, root_seq, root_raw, label_dict_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiTaskClassifier(input_dim=159, hidden_dim=256,
                                num_type=len(dataset.type2idx),
                                num_brand=len(dataset.brand2idx),
                                num_device=len(dataset.device2idx)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 可调节的任务损失权重（用于控制每个任务的训练重点）
    alpha_type = 1.0
    alpha_brand = 1.0
    alpha_device = 1.0

    print("🚀 开始多任务训练...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y_type, y_brand, y_device in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device)
            y_type = y_type.to(device)
            y_brand = y_brand.to(device)
            y_device = y_device.to(device)

            pred_type, pred_brand, pred_device = model(x)
            # 多任务损失组合，可添加权重以调节训练重点
            loss = alpha_type * loss_fn(pred_type, y_type.argmax(1)) \
                 + alpha_brand * loss_fn(pred_brand, y_brand.argmax(1)) \
                 + alpha_device * loss_fn(pred_device, y_device.argmax(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), output_model_path)
    print(f"✔️ 模型已保存至: {output_model_path}")

if __name__ == "__main__":
    train()
