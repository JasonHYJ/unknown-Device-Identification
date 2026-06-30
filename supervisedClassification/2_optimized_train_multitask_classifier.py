# train_multitask_classifier.py
# 📆 多任务IoT设备识别经典培训脚本

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from optimized_multimodal_dataset import MultiModalIoTDataset
from tqdm import tqdm

"""
(当前主要使用的用于训练模型的脚本，包含验证集，每次epoch的各项标签的效果)
脚本功能说明：
读取阶段一提取的统计特征、阶段二对比学习后的64维嵌入向量（序列+原始字节，区分闲时/行为）；
拼接成 287维 特征向量，进行门控融合（128维闲时 vs. 128维行为）；
输入 Transformer 结构进行学习；
输出三个分类器预测：
设备类型；
厂商；
型号；
使用多任务加权损失进行联合训练；
最终将训练好的模型保存到：/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/optimized_multitask_model.pt

模型结构说明：
输入向量：     [stat(31), idle_embed(128), behavior_embed(128)]  → 共287维
门控机制：     gate = sigmoid(W * is_behavior + b)
融合嵌入：     weighted_embed = gate * behavior + (1 - gate) * idle
拼接输入：     [stats, weighted_embed] → 输入 Transformer
Transformer： 编码行为/闲时流量的深层语义信息
输出层：       三个独立的 softmax 头 → type, brand, device
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置使用GPU 0

# --------------------------- 参数配置 ---------------------------
train_csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/3_uk_train.csv"  # 训练集CSV路径
test_csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/3_uk_test.csv"  # 验证集CSV路径
root_stat = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings"  # 统计特征目录
root_seq = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings"  # 序列嵌入目录
root_raw = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings"  # 原始字节嵌入目录
label_dict_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk"  # 标签字典目录

output_model_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/3_optimized_multitask_model.pt"  # 模型保存路径

batch_size = 64  # 批次大小
num_epochs = 40  # 训练轮数
lr = 1e-3  # 学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备（优先GPU）

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
            nn.Linear(1, 1),  # 输入is_behavior（1维），输出门控权重
            nn.Sigmoid()  # 将权重压缩到[0,1]
        )
        self.fc = nn.Linear(input_dim, hidden_dim)  # 全连接层：159维（31+128）转256维
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)  # Transformer编码层，4个注意力头
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 2层Transformer编码器
        self.classifier_type = nn.Linear(hidden_dim, num_type)  # 类型分类头
        self.classifier_brand = nn.Linear(hidden_dim, num_brand)  # 品牌分类头
        self.classifier_device = nn.Linear(hidden_dim, num_device)  # 型号分类头

    def forward(self, x):
        # 输入x: (batch_size, 287)，包含统计特征和嵌入
        stats = x[:, :31]  # 提取统计特征（31维，包括is_behavior）
        idle_beh = x[:, 31:]  # 提取嵌入特征（256维：128闲时+128行为）
        idle_embed = idle_beh[:, :128]  # 闲时嵌入（128维：64序列+64原始字节）
        behavior_embed = idle_beh[:, 128:]  # 行为嵌入（128维：64序列+64原始字节）
        is_behavior = stats[:, 30:31]  # 提取is_behavior标志（stat_vec[30]，1维）

        gate = self.gate(is_behavior)  # 计算门控权重（0~1），决定闲时/行为嵌入的贡献
        weighted_embed = gate * behavior_embed + (1 - gate) * idle_embed  # 融合嵌入：128维

        # 拼接统计特征和加权后的嵌入向量，作为 Transformer 输入
        # unsqueeze(1) 添加时间维度，使其符合 Transformer 的 batch_first 输入格式 (B, T=1, D=159)
        combined = torch.cat([stats, weighted_embed], dim=1).unsqueeze(1)  # (batch_size, 1, 159)

        x = self.fc(combined)  # 全连接层转换：(batch_size, 1, 256)
        x = self.encoder(x).squeeze(1)  # Transformer编码并移除序列维度：(batch_size, 256)

        out_type = self.classifier_type(x)  # 类型预测：(batch_size, num_type)
        out_brand = self.classifier_brand(x)  # 品牌预测：(batch_size, num_brand)
        out_device = self.classifier_device(x)  # 型号预测：(batch_size, num_device)
        return out_type, out_brand, out_device

# --------------------------- 验证逻辑 ---------------------------
def evaluate(model, dataloader, loss_fn, device):
    """在验证集上评估模型，计算损失和准确率"""
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    type_correct = 0
    brand_correct = 0
    device_correct = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for x, y_type, y_brand, y_device in dataloader:
            # 跳过无效样本（如特征文件缺失）
            if x is None:
                continue
            x = x.to(device)  # 将输入移到GPU/CPU
            y_type = y_type.to(device)  # 类型标签（one-hot）
            y_brand = y_brand.to(device)  # 品牌标签（one-hot）
            y_device = y_device.to(device)  # 型号标签（one-hot）

            pred_type, pred_brand, pred_device = model(x)  # 模型前向传播
            # 计算多任务损失：类型、品牌、型号的交叉熵损失之和
            loss = loss_fn(pred_type, y_type.argmax(1)) + \
                   loss_fn(pred_brand, y_brand.argmax(1)) + \
                   loss_fn(pred_device, y_device.argmax(1))

            total_loss += loss.item()  # 累加批次损失
            total_samples += x.size(0)  # 累加样本数

            # 计算准确率
            type_correct += (pred_type.argmax(1) == y_type.argmax(1)).sum().item()  # 类型正确预测数
            brand_correct += (pred_brand.argmax(1) == y_brand.argmax(1)).sum().item()  # 品牌正确预测数
            device_correct += (pred_device.argmax(1) == y_device.argmax(1)).sum().item()  # 型号正确预测数

    # 计算平均损失和准确率
    avg_loss = total_loss / len(dataloader) if total_samples > 0 else float('inf')  # 平均损失
    type_acc = type_correct / total_samples if total_samples > 0 else 0.0  # 类型准确率
    brand_acc = brand_correct / total_samples if total_samples > 0 else 0.0  # 品牌准确率
    device_acc = device_correct / total_samples if total_samples > 0 else 0.0  # 型号准确率

    return avg_loss, type_acc, brand_acc, device_acc

# --------------------------- 训练逻辑 ---------------------------
def train():
    # 加载训练和验证数据集
    train_dataset = MultiModalIoTDataset(train_csv_path, root_stat, root_seq, root_raw, label_dict_dir)  # 训练数据集
    test_dataset = MultiModalIoTDataset(test_csv_path, root_stat, root_seq, root_raw, label_dict_dir)  # 验证数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器，随机打乱
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 验证数据加载器，保持顺序

    # 初始化模型
    model = MultiTaskClassifier(
        input_dim=159,  # 输入维度：31（统计）+128（加权嵌入）
        hidden_dim=256,  # 隐藏层维度
        num_type=len(train_dataset.type2idx),  # 类型类别数
        num_brand=len(train_dataset.brand2idx),  # 品牌类别数
        num_device=len(train_dataset.device2idx)  # 型号类别数
    ).to(device)  # 移到GPU/CPU

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 多任务损失权重，可调节任务重要性
    alpha_type = 1.0  # 类型任务权重
    alpha_brand = 1.0  # 品牌任务权重
    alpha_device = 1.0  # 型号任务权重

    print("🚀 开始多任务训练...")

    # 梯度裁剪，防止梯度爆炸
    max_grad_norm = 2

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0
        total_samples = 0

        for x, y_type, y_brand, y_device in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 跳过无效样本（如特征文件缺失）
            if x is None:
                continue
            x = x.to(device)  # 将输入移到GPU/CPU
            y_type = y_type.to(device)  # 类型标签
            y_brand = y_brand.to(device)  # 品牌标签
            y_device = y_device.to(device)  # 型号标签

            pred_type, pred_brand, pred_device = model(x)  # 模型前向传播
            # 计算加权多任务损失
            loss = alpha_type * loss_fn(pred_type, y_type.argmax(1)) + \
                   alpha_brand * loss_fn(pred_brand, y_brand.argmax(1)) + \
                   alpha_device * loss_fn(pred_device, y_device.argmax(1))

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪
            optimizer.step()  # 更新参数

            total_loss += loss.item() * x.size(0)  # 累加批次损失
            total_samples += x.size(0)  # 累加样本数

        # 计算平均训练损失
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"✅ Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        # 在验证集上评估
        val_loss, type_acc, brand_acc, device_acc = evaluate(model, test_dataloader, loss_fn, device)
        print(f"✅ Epoch {epoch+1}, Val Loss: {val_loss:.4f}, "
              f"Type Acc: {type_acc:.4f}, Brand Acc: {brand_acc:.4f}, Device Acc: {device_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), output_model_path)
    print(f"✔️ 模型已保存至: {output_model_path}")

if __name__ == "__main__":
    train()