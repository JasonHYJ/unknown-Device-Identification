import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from optimized_multimodal_dataset import MultiModalIoTDataset  # 假设你用的是之前定义的 dataset 类
import json  # 添加 json 导入

# ------------------------ 配置参数 ------------------------
csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/uk_test.csv"
root_stat = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings/test/uk"
root_seq = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings/test/uk"
root_raw = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings/test/uk"
label_dict_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk"

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ 加载模型 ------------------------
class MultiTaskClassifier(torch.nn.Module):
    # 请确保这个类与你训练时的模型类一致
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            torch.nn.Sigmoid()
        )
        self.fc = torch.nn.Linear(input_dim, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier_type = torch.nn.Linear(hidden_dim, num_type)
        self.classifier_brand = torch.nn.Linear(hidden_dim, num_brand)
        self.classifier_device = torch.nn.Linear(hidden_dim, num_device)

    def forward(self, x):
        stats = x[:, :31]           # 31维统计特征
        idle_beh = x[:, 31:]        # 拼接的闲时+行为嵌入向量
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

# 加载训练好的模型
model_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/multitask_model.pt"
model = MultiTaskClassifier(input_dim=159, hidden_dim=256,
                            num_type=3, num_brand=3, num_device=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ------------------------ 数据集定义 ------------------------
dataset = MultiModalIoTDataset(csv_path, root_stat, root_seq, root_raw, label_dict_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ------------------------ 评估函数 ------------------------
def evaluate():
    all_preds_type = []
    all_preds_brand = []
    all_preds_device = []
    all_true_type = []
    all_true_brand = []
    all_true_device = []

    with torch.no_grad():
        for x, y_type, y_brand, y_device in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            y_type = y_type.to(device)
            y_brand = y_brand.to(device)
            y_device = y_device.to(device)

            pred_type, pred_brand, pred_device = model(x)

            all_preds_type.append(pred_type.argmax(dim=1).cpu().numpy())
            all_preds_brand.append(pred_brand.argmax(dim=1).cpu().numpy())
            all_preds_device.append(pred_device.argmax(dim=1).cpu().numpy())
            
            all_true_type.append(y_type.argmax(dim=1).cpu().numpy())
            all_true_brand.append(y_brand.argmax(dim=1).cpu().numpy())
            all_true_device.append(y_device.argmax(dim=1).cpu().numpy())

    # Flatten lists
    all_preds_type = np.concatenate(all_preds_type)
    all_preds_brand = np.concatenate(all_preds_brand)
    all_preds_device = np.concatenate(all_preds_device)
    all_true_type = np.concatenate(all_true_type)
    all_true_brand = np.concatenate(all_true_brand)
    all_true_device = np.concatenate(all_true_device)

    # 计算准确率和 F1 分数
    acc_type = accuracy_score(all_true_type, all_preds_type)
    acc_brand = accuracy_score(all_true_brand, all_preds_brand)
    acc_device = accuracy_score(all_true_device, all_preds_device)

    f1_type = f1_score(all_true_type, all_preds_type, average='weighted')
    f1_brand = f1_score(all_true_brand, all_preds_brand, average='weighted')
    f1_device = f1_score(all_true_device, all_preds_device, average='weighted')

    print(f"设备类型准确率: {acc_type:.4f}, F1 分数: {f1_type:.4f}")
    print(f"厂商准确率: {acc_brand:.4f}, F1 分数: {f1_brand:.4f}")
    print(f"型号准确率: {acc_device:.4f}, F1 分数: {f1_device:.4f}")

if __name__ == "__main__":
    evaluate()
