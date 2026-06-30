import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class MultiModalIoTDataset(Dataset):
    def __init__(self, csv_path, root_stat, root_seq, root_raw, label_dict_dir):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root_stat = Path(root_stat)
        self.root_seq = Path(root_seq)
        self.root_raw = Path(root_raw)

        # 加载标签映射字典
        with open(Path(label_dict_dir) / "type2idx.json") as f:
            self.type2idx = json.load(f)
        with open(Path(label_dict_dir) / "brand2idx.json") as f:
            self.brand2idx = json.load(f)
        with open(Path(label_dict_dir) / "device2idx.json") as f:
            self.device2idx = json.load(f)

        print(f"✅ 加载样本数: {len(self.df)}")
        print(f"📚 标签类别数: type={len(self.type2idx)}, brand={len(self.brand2idx)}, device={len(self.device2idx)}")

        # 归一化器预训练
        print("📐 正在计算统计特征归一化参数...")
        all_stat_vectors = []
        for path in self.df["file_path"]:
            stat_path = Path(path)
            vec = pd.read_csv(stat_path, skiprows=1, header=None).iloc[0, :31].values.astype(np.float32)
            all_stat_vectors.append(vec)
        all_stat_array = np.stack(all_stat_vectors)
        self.scaler = MinMaxScaler()
        self.scaler.fit(all_stat_array)
        print("✅ 完成统计特征归一化拟合")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stat_path = Path(row["file_path"])  # 原始统计特征路径

        # 加载统计特征
        stat_vec = pd.read_csv(stat_path, skiprows=1, header=None).iloc[0, :31].values.astype(np.float32)
        stat_vec = self.scaler.transform(stat_vec.reshape(1, -1)).squeeze(0)  # 归一化

        # 获取 is_behavior 标签（0/1）
        is_behavior = int(row["is_behavior"])

        # 推导序列嵌入路径
        seq_embed_path = stat_path.as_posix().replace(
            "7_cleaned_features/7_cleaned_statistical_feature",
            "10_contrastive_embeddings/10_contrastive_sequence_embeddings"
        ).replace("_stat.csv", "_seq_embed.npy")

        # 推导原始字节嵌入路径
        raw_embed_path = stat_path.as_posix().replace(
            "7_cleaned_features/7_cleaned_statistical_feature",
            "10_contrastive_embeddings/10_contrastive_rawbyte_embeddings"
        ).replace("_stat.csv", "_raw_embed.npy")

        # 加载嵌入（行为或闲时嵌入只保留一方，其余置零）
        seq_embed = np.load(seq_embed_path)
        raw_embed = np.load(raw_embed_path)

        if is_behavior == 1:
            idle_embed = np.zeros(128, dtype=np.float32)
            behavior_embed = np.concatenate([seq_embed, raw_embed])  # 128维
        else:
            idle_embed = np.concatenate([seq_embed, raw_embed])  # 128维
            behavior_embed = np.zeros(128, dtype=np.float32)

        # 拼接最终输入向量 (31 + 128 + 128 = 287)
        input_vector = np.concatenate([stat_vec, idle_embed, behavior_embed])
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)

        # 获取标签并 one-hot 编码
        type_idx = self.type2idx[row["type_label"]]
        brand_idx = self.brand2idx[row["brand_label"]]
        device_idx = self.device2idx[row["device_label"]]

        type_onehot = torch.nn.functional.one_hot(torch.tensor(type_idx), num_classes=len(self.type2idx)).float()
        brand_onehot = torch.nn.functional.one_hot(torch.tensor(brand_idx), num_classes=len(self.brand2idx)).float()
        device_onehot = torch.nn.functional.one_hot(torch.tensor(device_idx), num_classes=len(self.device2idx)).float()

        return input_tensor, type_onehot, brand_onehot, device_onehot
