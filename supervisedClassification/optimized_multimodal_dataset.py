# optimized_multimodal_dataset.py
# 📆 多模态IoT设备数据集处理脚本
#
# 功能说明：
# 本脚本定义了一个PyTorch Dataset类（MultiModalIoTDataset），用于加载和处理IoT设备识别任务的多模态数据。
# - 加载CSV文件（uk_test.csv或uk_unknown.csv），包含样本的标签和特征文件路径。
# - 处理三种特征：统计特征（31维）、序列嵌入（64维）、原始字节嵌入（64维）。
# - 根据is_behavior标志动态拼接闲时（idle）和行为（behavior）嵌入，生成287维输入向量（31+128+128）。
# - 支持标签的one-hot编码，处理无效标签（未在标签字典中的标签设为-1）。
# - 提供数据验证（如维度检查、NaN/无穷大检测）和错误日志输出。
# - 关键特性：
#   - 动态特征拼接，适配闲时和行为样本。
#   - 无效标签计数，批量输出警告（每100个样本或最后一次）。
#   - 与3_evaluate_multitask_classifier.py兼容，用于模型评估。

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter

class MultiModalIoTDataset(Dataset):
    def __init__(self, csv_path, root_stat, root_seq, root_raw, label_dict_dir):
        super().__init__()
        self.df = pd.read_csv(csv_path)  # 加载CSV文件，包含样本的标签和特征路径
        self.root_stat = Path(root_stat)  # 统计特征目录路径
        self.root_seq = Path(root_seq)  # 序列嵌入目录路径
        self.root_raw = Path(root_raw)  # 原始字节嵌入目录路径

        # 加载标签映射字典（type2idx.json等）
        with open(Path(label_dict_dir) / "type2idx.json") as f:
            self.type2idx = json.load(f)  # 类型标签到索引的映射
        with open(Path(label_dict_dir) / "brand2idx.json") as f:
            self.brand2idx = json.load(f)  # 品牌标签到索引的映射
        with open(Path(label_dict_dir) / "device2idx.json") as f:
            self.device2idx = json.load(f)  # 型号标签到索引的映射

        # 输出数据集信息
        print(f"✅ 加载样本数: {len(self.df)}")
        print(f"📚 标签类别数: type={len(self.type2idx)}, brand={len(self.brand2idx)}, device={len(self.device2idx)}")

        # 预期维度
        self.expected_stat_dim = 31  # 统计特征维度
        self.expected_embed_dim = 64  # 序列和原始字节嵌入各64维，拼接后128维

        # 无效标签计数器，用于记录未在标签字典中的标签
        self.invalid_label_counts = Counter()

    def __len__(self):
        return len(self.df)  # 返回数据集样本总数

    def __getitem__(self, idx):
        row = self.df.iloc[idx]  # 获取指定索引的CSV行
        stat_path = Path(row["stat_feature_path"])  # 统计特征文件路径
        seq_embed_path = Path(row["seq_embed_feature_path"])  # 序列嵌入文件路径
        raw_embed_path = Path(row["raw_embed_feature_path"])  # 原始字节嵌入文件路径

        # 检查特征文件是否存在
        for path, name in [(stat_path, "stat"), (seq_embed_path, "seq_embed"), (raw_embed_path, "raw_embed")]:
            if not path.exists():
                print(f"🔴 Warning: {name} file not found at index {idx}, path: {path}")
                return None, None, None, None  # 文件缺失返回None

        # 加载统计特征
        try:
            stat_vec = pd.read_csv(stat_path, skiprows=1, header=None).iloc[0, :self.expected_stat_dim].values.astype(np.float32)  # 读取CSV第一行，截取31维
        except Exception as e:
            print(f"🔴 Warning: Failed to load stat file at index {idx}, path: {stat_path}, error: {e}")
            return None, None, None, None  # 加载失败返回None

        # 验证统计特征维度
        if stat_vec.shape[0] != self.expected_stat_dim:
            print(f"🔴 Warning: Stat vector dimension mismatch at index {idx}, expected {self.expected_stat_dim}, got {stat_vec.shape[0]}")
            return None, None, None, None  # 维度不匹配返回None

        # 检查 NaN 或无穷大
        if np.any(np.isnan(stat_vec)) or np.any(np.isinf(stat_vec)):
            print(f"🔴 Warning: NaN or inf in stat vector at index {idx}, path: {stat_path}")
            print(f"Stat vector: {stat_vec}")
            return None, None, None, None  # 包含NaN或无穷大返回None

        # 验证is_behavior一致性
        is_behavior = int(row["is_behavior"])  # CSV中的is_behavior标志（0=闲时，1=行为）
        if stat_vec[30] != is_behavior:
            print(f"🔴 Warning: is_behavior mismatch at index {idx}, CSV: {is_behavior}, stat_vec[30]: {stat_vec[30]}")
            # 不返回None，仅记录警告，保持数据完整性

        # 加载序列和原始字节嵌入
        try:
            seq_embed = np.load(seq_embed_path)  # 加载序列嵌入（64维）
            raw_embed = np.load(raw_embed_path)  # 加载原始字节嵌入（64维）
        except Exception as e:
            print(f"🔴 Warning: Failed to load embed files at index {idx}, seq: {seq_embed_path}, raw: {raw_embed_path}, error: {e}")
            return None, None, None, None  # 加载失败返回None

        # 验证嵌入维度
        if seq_embed.shape[0] != self.expected_embed_dim or raw_embed.shape[0] != self.expected_embed_dim:
            print(f"🔴 Warning: Embed dimension mismatch at index {idx}, "
                  f"seq expected {self.expected_embed_dim}, got {seq_embed.shape[0]}, "
                  f"raw expected {self.expected_embed_dim}, got {raw_embed.shape[0]}")
            return None, None, None, None  # 维度不匹配返回None

        # 根据is_behavior动态拼接嵌入
        if is_behavior == 1:
            idle_embed = np.zeros(2 * self.expected_embed_dim, dtype=np.float32)  # 行为样本：闲时嵌入置零（128维）
            behavior_embed = np.concatenate([seq_embed, raw_embed])  # 行为嵌入：拼接序列和原始字节（128维）
        else:
            idle_embed = np.concatenate([seq_embed, raw_embed])  # 闲时样本：闲时嵌入拼接（128维）
            behavior_embed = np.zeros(2 * self.expected_embed_dim, dtype=np.float32)  # 行为嵌入置零（128维）

        # 拼接最终输入向量：统计特征（31维）+闲时嵌入（128维）+行为嵌入（128维）=287维
        input_vector = np.concatenate([stat_vec, idle_embed, behavior_embed])
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)  # 转换为PyTorch张量

        # 获取标签并进行one-hot编码
        try:
            type_idx = self.type2idx[row["type_label"]]  # 获取类型标签索引
        except KeyError:
            self.invalid_label_counts[f"type_label_{row['type_label']}"] += 1  # 记录无效类型标签
            type_idx = -1  # 无效标签设为-1
        try:
            brand_idx = self.brand2idx[row["brand_label"]]  # 获取品牌标签索引
        except KeyError:
            self.invalid_label_counts[f"brand_label_{row['brand_label']}"] += 1  # 记录无效品牌标签
            brand_idx = -1  # 无效标签设为-1
        try:
            device_idx = self.device2idx[row["device_label"]]  # 获取型号标签索引
        except KeyError:
            self.invalid_label_counts[f"device_label_{row['device_label']}"] += 1  # 记录无效型号标签
            device_idx = -1  # 无效标签设为-1

        # 批量输出无效标签警告（每100个样本或最后样本）
        if (idx + 1) % 100 == 0 or idx == len(self.df) - 1:
            for label, count in self.invalid_label_counts.items():
                if count > 0:
                    print(f"🔴 Warning: Invalid {label} appeared {count} times up to index {idx}")
            self.invalid_label_counts.clear()  # 清空计数器

        # 对有效标签进行one-hot编码，无效标签返回全零向量
        if type_idx == -1:
            type_onehot = torch.zeros(len(self.type2idx), dtype=torch.float32)  # 无效类型标签：全零向量
        else:
            type_onehot = torch.nn.functional.one_hot(torch.tensor(type_idx), num_classes=len(self.type2idx)).float()  # one-hot编码
        if brand_idx == -1:
            brand_onehot = torch.zeros(len(self.brand2idx), dtype=torch.float32)  # 无效品牌标签：全零向量
        else:
            brand_onehot = torch.nn.functional.one_hot(torch.tensor(brand_idx), num_classes=len(self.brand2idx)).float()  # one-hot编码
        if device_idx == -1:
            device_onehot = torch.zeros(len(self.device2idx), dtype=torch.float32)  # 无效型号标签：全零向量
        else:
            device_onehot = torch.nn.functional.one_hot(torch.tensor(device_idx), num_classes=len(self.device2idx)).float()  # one-hot编码

        return input_tensor, type_onehot, brand_onehot, device_onehot  # 返回输入张量和三个one-hot标签