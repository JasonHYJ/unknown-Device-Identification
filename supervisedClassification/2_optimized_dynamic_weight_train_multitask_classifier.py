# 2_optimized_dynamic_weight_train_multitask_classifier.py
# 📆 多任务IoT设备识别经典培训脚本

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from optimized_multimodal_dataset import MultiModalIoTDataset
from tqdm import tqdm
import numpy as np
import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/optimized_dynamic_weight_multitask_model_train_log.txt')
    ]
)
logger = logging.getLogger(__name__)

"""
（当前暂未使用的脚本，最后一个版本，使用动态权重，最后选择效果最好的组的模型保存）
脚本功能说明：
读取阶段一提取的统计特征、阶段二对比学习后的64维嵌入向量（序列+原始字节，区分闲时/行为）；
拼接成 287维 特征向量，进行门控融合（128维闲时 vs. 128维行为）；
输入 Transformer 结构进行学习；
输出三个分类器预测：设备类型、厂商、型号；
使用多任务加权损失进行联合训练，动态调整任务权重；
记录梯度范数以监控训练稳定性；
最终将训练好的模型保存到：/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/optimized_dynamic_weight_multitask_model.pt
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --------------------------- 参数配置 ---------------------------
train_csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/uk_train.csv"
test_csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/uk_test.csv"
root_stat = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings"
root_seq = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings"
root_raw = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings"
label_dict_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk"

output_model_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/optimized_dynamic_weight_multitask_model.pt"

batch_size = 64
num_epochs = 40
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 任务权重初始值
alpha_type = 1.0
alpha_brand = 1.0
alpha_device = 1.0
weight_adjust_rate = 0.1
min_weight = 0.5
max_weight = 2.0

# --------------------------- 模型定义 ---------------------------
class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier_type = nn.Linear(hidden_dim, num_type)
        self.classifier_brand = nn.Linear(hidden_dim, num_brand)
        self.classifier_device = nn.Linear(hidden_dim, num_device)

    def forward(self, x):
        stats = x[:, :31]
        idle_beh = x[:, 31:]
        idle_embed = idle_beh[:, :128]
        behavior_embed = idle_beh[:, 128:]
        is_behavior = stats[:, 0:1]

        gate = self.gate(is_behavior)
        weighted_embed = gate * behavior_embed + (1 - gate) * idle_embed

        combined = torch.cat([stats, weighted_embed], dim=1).unsqueeze(1)
        x = self.fc(combined)
        x = self.encoder(x).squeeze(1)

        out_type = self.classifier_type(x)
        out_brand = self.classifier_brand(x)
        out_device = self.classifier_device(x)
        return out_type, out_brand, out_device

# --------------------------- 验证逻辑 ---------------------------
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    type_correct = 0
    brand_correct = 0
    device_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y_type, y_brand, y_device in dataloader:
            if x is None:
                continue
            x = x.to(device)
            y_type = y_type.to(device)
            y_brand = y_brand.to(device)
            y_device = y_device.to(device)

            pred_type, pred_brand, pred_device = model(x)
            loss = loss_fn(pred_type, y_type.argmax(1)) + \
                   loss_fn(pred_brand, y_brand.argmax(1)) + \
                   loss_fn(pred_device, y_device.argmax(1))

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            type_correct += (pred_type.argmax(1) == y_type.argmax(1)).sum().item()
            brand_correct += (pred_brand.argmax(1) == y_brand.argmax(1)).sum().item()
            device_correct += (pred_device.argmax(1) == y_device.argmax(1)).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    type_acc = type_correct / total_samples if total_samples > 0 else 0.0
    brand_acc = brand_correct / total_samples if total_samples > 0 else 0.0
    device_acc = device_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, type_acc, brand_acc, device_acc

# --------------------------- 梯度信息记录 ---------------------------
def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)

# --------------------------- 训练逻辑 ---------------------------
def train():
    logger.info("Starting multi-task training...")
    logger.info(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")

    # 检查文件存在性
    for path in [train_csv_path, test_csv_path, label_dict_dir]:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            sys.exit(1)
    # 检查标签字典文件
    for json_file in ["type2idx.json", "brand2idx.json", "device2idx.json"]:
        json_path = os.path.join(label_dict_dir, json_file)
        if not os.path.exists(json_path):
            logger.error(f"Label dictionary not found: {json_path}")
            sys.exit(1)

    # 加载训练和验证数据集
    try:
        train_dataset = MultiModalIoTDataset(train_csv_path, root_stat, root_seq, root_raw, label_dict_dir)
        test_dataset = MultiModalIoTDataset(test_csv_path, root_stat, root_seq, root_raw, label_dict_dir)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    if len(train_dataset) == 0:
        logger.error("Train dataset is empty. Check CSV file or feature files.")
        sys.exit(1)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 检查 DataLoader 是否为空
    try:
        next(iter(train_dataloader))
        logger.info("Train DataLoader initialized successfully.")
    except StopIteration:
        logger.error("Train DataLoader is empty. No valid samples found.")
        sys.exit(1)

    # 初始化模型
    try:
        model = MultiTaskClassifier(input_dim=159, hidden_dim=256,
                                   num_type=len(train_dataset.type2idx),
                                   num_brand=len(train_dataset.brand2idx),
                                   num_device=len(train_dataset.device2idx)).to(device)
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    global alpha_type, alpha_brand, alpha_device
    best_val_acc = 0.0

    max_grad_norm = 2

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        total_grad_norm = 0
        grad_batches = 0

        for x, y_type, y_brand, y_device in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout):
            if x is None:
                continue
            x = x.to(device)
            y_type = y_type.to(device)
            y_brand = y_brand.to(device)
            y_device = y_device.to(device)

            pred_type, pred_brand, pred_device = model(x)
            loss_type = loss_fn(pred_type, y_type.argmax(1))
            loss_brand = loss_fn(pred_brand, y_brand.argmax(1))
            loss_device = loss_fn(pred_device, y_device.argmax(1))
            loss = alpha_type * loss_type + alpha_brand * loss_brand + alpha_device * loss_device

            optimizer.zero_grad()
            loss.backward()
            grad_norm = compute_gradient_norm(model)
            total_grad_norm += grad_norm
            grad_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        avg_grad_norm = total_grad_norm / grad_batches if grad_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")

        # 验证集评估
        try:
            val_loss, type_acc, brand_acc, device_acc = evaluate(model, test_dataloader, loss_fn, device)
            avg_val_acc = (type_acc + brand_acc + device_acc) / 3
            logger.info(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, "
                        f"Type Acc: {type_acc:.4f}, Brand Acc: {brand_acc:.4f}, Device Acc: {device_acc:.4f}, "
                        f"Avg Val Acc: {avg_val_acc:.4f}")
        except Exception as e:
            logger.error(f"Validation failed at epoch {epoch+1}: {e}")
            continue

        # 动态调整任务权重
        if epoch > 5:
            max_acc = max(type_acc, brand_acc, device_acc)
            if max_acc > 0:
                if type_acc < max_acc * 0.9:
                    alpha_type = min(alpha_type + weight_adjust_rate, max_weight)
                else:
                    alpha_type = max(alpha_type - weight_adjust_rate, min_weight)
                if brand_acc < max_acc * 0.9:
                    alpha_brand = min(alpha_brand + weight_adjust_rate, max_weight)
                else:
                    alpha_brand = max(alpha_brand - weight_adjust_rate, min_weight)
                if device_acc < max_acc * 0.9:
                    alpha_device = min(alpha_device + weight_adjust_rate, max_weight)
                else:
                    alpha_device = max(alpha_device - weight_adjust_rate, min_weight)
            logger.info(f"Epoch {epoch+1}, Updated Weights: Type={alpha_type:.2f}, "
                        f"Brand={alpha_brand:.2f}, Device={alpha_device:.2f}")

        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            try:
                torch.save(model.state_dict(), output_model_path)
                logger.info(f"Best model saved with Avg Val Acc: {best_val_acc:.4f}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    logger.info(f"Training completed. Final model saved to: {output_model_path}")

if __name__ == "__main__":
    try:
        logger.info("Script started.")
        train()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)