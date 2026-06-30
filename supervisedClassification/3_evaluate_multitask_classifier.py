# evaluate_multitask_classifier.py
# 📆 多任务IoT设备识别评估脚本

# 功能说明：
# 本脚本用于评估多任务IoT设备识别模型的性能，支持对已知设备（测试集）和未知设备（未知集）进行分类。
# - 加载预训练的多任务分类模型（optimized_multitask_model.pt），处理统计特征、序列嵌入和原始字节嵌入。
# - 对测试集（uk_test.csv）进行采样并评估类型（type）、品牌（brand）和型号（device）的分类准确率和F1分数。
# - 对未知集（uk_unknown.csv）评估模型对未见设备的泛化能力，计算未知型号识别率（device_unknown_rate）。
# - 支持闲时（idle）和行为（behavior）样本的分别评估，输出详细指标到日志和CSV文件。
# - 日志保存到 eval_results/eval_log.txt，结果保存到 test_results.csv 和 unknown_results.csv。
# - 关键特性：
#   - 使用门控机制融合闲时和行为嵌入。
#   - 处理无效标签（未知集中的未见标签设为-1）。
#   - 支持GPU加速（若CUDA可用）。

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from optimized_multimodal_dataset import MultiModalIoTDataset  # 自定义数据集类
from tqdm import tqdm  # 进度条
from sklearn.metrics import f1_score  # 计算F1分数
import numpy as np
import logging
import sys
import json

# 确保日志目录存在，可以替换成对应模型要保存的文件夹
log_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us/3_eval_results"
os.makedirs(log_dir, exist_ok=True)  # 创建日志目录，若存在则忽略

# 设置日志，输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 控制台输出
        logging.FileHandler(os.path.join(log_dir, 'eval_log.txt'))  # 日志文件
    ]
)
logger = logging.getLogger(__name__)

# 设置CUDA设备（使用GPU 0）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# --------------------------- 参数配置 ---------------------------
# 数据和模型路径
test_csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us/3_us_test.csv"  # 测试集CSV
unknown_csv_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us/3_us_unknown.csv"  # 测试集CSV
root_stat = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings"  # 统计特征目录
root_seq = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings"  # 序列嵌入目录
root_raw = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings"  # 原始字节嵌入目录
label_dict_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us"  # 标签字典目录

# 可以替换模型，当前模型optimized_multitask_model.pt的结果比较优
model_path = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us/3_optimized_multitask_model.pt"  # 预训练模型路径

output_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us/3_eval_results"  # 输出目录
batch_size = 64  # 批次大小
confidence_threshold = 0.5  # 类型和品牌预测的置信度阈值
device_threshold = 0.7  # 型号预测的置信度阈值（低于此值预测为未知）
test_sample_frac = 0.35  # 测试集采样比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备（优先GPU）

# --------------------------- 模型定义 ---------------------------
class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        # 门控网络：根据is_behavior决定使用闲时还是行为嵌入
        self.gate = nn.Sequential(
            nn.Linear(1, 1),  # 输入is_behavior（1维），输出权重
            nn.Sigmoid()  # 输出[0,1]权重
        )
        self.fc = nn.Linear(input_dim, hidden_dim)  # 全连接层：输入159维（31+128），输出256维
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)  # Transformer编码层
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 2层Transformer
        self.classifier_type = nn.Linear(hidden_dim, num_type)  # 类型分类头
        self.classifier_brand = nn.Linear(hidden_dim, num_brand)  # 品牌分类头
        self.classifier_device = nn.Linear(hidden_dim, num_device)  # 型号分类头

    def forward(self, x):
        stats = x[:, :31]  # 提取统计特征（31维）
        idle_beh = x[:, 31:]  # 提取嵌入特征（256维：128闲时+128行为）
        idle_embed = idle_beh[:, :128]  # 闲时嵌入（128维）
        behavior_embed = idle_beh[:, 128:]  # 行为嵌入（128维）
        is_behavior = stats[:, 30:31]  # 提取is_behavior标志（stat_vec[30]）

        gate = self.gate(is_behavior)  # 计算门控权重（0~1）
        weighted_embed = gate * behavior_embed + (1 - gate) * idle_embed  # 融合闲时和行为嵌入

        combined = torch.cat([stats, weighted_embed], dim=1).unsqueeze(1)  # 拼接统计特征和加权嵌入，增加序列维度
        x = self.fc(combined)  # 全连接层转换
        x = self.encoder(x).squeeze(1)  # Transformer编码并移除序列维度

        out_type = self.classifier_type(x)  # 类型预测
        out_brand = self.classifier_brand(x)  # 品牌预测
        out_device = self.classifier_device(x)  # 型号预测
        return out_type, out_brand, out_device

# --------------------------- 评估已知设备 ---------------------------
def evaluate_known_dataset(dataset, dataloader, model, loss_fn, device, dataset_name, idx2type, idx2brand, idx2device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    type_correct = 0
    brand_correct = 0
    device_correct = 0
    total_samples = 0
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    device_preds, device_labels = [], []
    results = []
    invalid_samples = 0

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch  # 获取批次数据：输入特征和one-hot标签
                if x is None or y_type is None or y_brand is None or y_device is None:
                    logger.warning(f"Batch {batch_idx} contains invalid data (None), skipping.")
                    invalid_samples += batch_size
                    continue
                x = x.to(device)  # 将输入移到GPU/CPU
                y_type = y_type.to(device)
                y_brand = y_brand.to(device)
                y_device = y_device.to(device)

                pred_type, pred_brand, pred_device = model(x)  # 模型前向传播
                # 计算损失：类型、品牌、型号的交叉熵损失之和
                loss = loss_fn(pred_type, y_type.argmax(1)) + \
                       loss_fn(pred_brand, y_brand.argmax(1)) + \
                       loss_fn(pred_device, y_device.argmax(1))

                total_loss += loss.item() * x.size(0)  # 累加批次损失
                total_samples += x.size(0)  # 累加样本数

                # 获取预测标签
                type_pred = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()
                device_pred = pred_device.argmax(1).cpu().numpy()
                type_label = y_type.argmax(1).cpu().numpy()
                brand_label = y_brand.argmax(1).cpu().numpy()
                device_label = y_device.argmax(1).cpu().numpy()

                # 计算预测概率
                type_probs = torch.softmax(pred_type, dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs = torch.softmax(pred_device, dim=1).cpu().numpy()

                # 统计正确预测数
                type_correct += (type_pred == type_label).sum()
                brand_correct += (brand_pred == brand_label).sum()
                device_correct += (device_pred == device_label).sum()

                # 保存预测和真实标签用于F1计算
                type_preds.extend(type_pred)
                type_labels.extend(type_label)
                brand_preds.extend(brand_pred)
                brand_labels.extend(brand_label)
                device_preds.extend(device_pred)
                device_labels.extend(device_label)

                # 保存每个样本的预测结果
                for i in range(len(type_pred)):
                    true_type_label = idx2type.get(type_label[i], "unknown")  # 真实类型标签
                    true_brand_label = idx2brand.get(brand_label[i], "unknown")  # 真实品牌标签
                    true_device_label = idx2device.get(device_label[i], "unknown")  # 真实型号标签

                    results.append({
                        'index': total_samples - len(type_pred) + i,  # 真实型号标签
                        'true_type': true_type_label,
                        'pred_type': idx2type[type_pred[i]],  # 预测类型标签
                        'type_prob': type_probs[i].max(),  # 类型最大概率
                        'true_brand': true_brand_label,
                        'pred_brand': idx2brand[brand_pred[i]],  # 预测品牌标签
                        'brand_prob': brand_probs[i].max(),  # 品牌最大概率
                        'true_device': true_device_label,
                        'pred_device': idx2device[device_pred[i]],  # 预测型号标签
                        'device_prob': device_probs[i].max(),  # 型号最大概率
                        'is_behavior': dataset.df.iloc[total_samples - len(type_pred) + i]['is_behavior']  # 是否为行为样本
                    })
            except Exception as e:
                logger.warning(f"Batch {batch_idx} processing failed: {e}")
                invalid_samples += batch_size
                continue

    if total_samples == 0:
        logger.error("No valid samples processed in test set evaluation")
        return None, None

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    type_acc = type_correct / total_samples
    brand_acc = brand_correct / total_samples
    device_acc = device_correct / total_samples
    # 计算F1分数，忽略无效标签
    type_f1 = f1_score(type_labels, type_preds, average='macro', zero_division=0)
    brand_f1 = f1_score(brand_labels, brand_preds, average='macro', zero_division=0)
    device_f1 = f1_score(device_labels, device_preds, average='macro', zero_division=0)

    # 汇总指标
    metrics = {
        'dataset': dataset_name,
        'loss': avg_loss,
        'type_acc': type_acc,
        'type_f1': type_f1,
        'brand_acc': brand_acc,
        'brand_f1': brand_f1,
        'device_acc': device_acc,
        'device_f1': device_f1,
        'total_samples': total_samples,
        'invalid_samples': invalid_samples
    }

    logger.info(f"Processed {total_samples} samples, skipped {invalid_samples} invalid samples")
    return metrics, results

# --------------------------- 评估未知设备 ---------------------------
def evaluate_unknown_dataset(dataset, dataloader, model, device, dataset_name, confidence_threshold, device_threshold, idx2type, idx2brand, idx2device):
    model.eval()  # 设置模型为评估模式
    total_samples = 0
    idle_samples = 0
    behavior_samples = 0
    type_correct = 0
    brand_correct = 0
    device_unknown = 0
    idle_type_correct = 0
    idle_brand_correct = 0
    behavior_type_correct = 0
    behavior_brand_correct = 0
    results = []
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    invalid_samples = 0
    invalid_label_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch  # 获取批次数据
                if x is None or y_type is None or y_brand is None or y_device is None:
                    logger.warning(f"Batch {batch_idx} contains invalid data (None), skipping.")
                    invalid_samples += batch_size
                    continue
                x = x.to(device)  # 将输入移到GPU/CPU
                y_type = y_type.to(device)
                y_brand = y_brand.to(device)

                pred_type, pred_brand, pred_device = model(x)  # 模型前向传播
                # 计算预测概率
                type_probs = torch.softmax(pred_type, dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs = torch.softmax(pred_device, dim=1).cpu().numpy()

                # 获取预测标签
                type_pred = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()
                type_label = y_type.argmax(1).cpu().numpy()
                brand_label = y_brand.argmax(1).cpu().numpy()

                # 从数据集获取 is_behavior 和原始标签
                batch_indices = range(total_samples, total_samples + len(type_pred))  # 当前批次样本索引
                is_behavior = dataset.df.iloc[batch_indices]['is_behavior'].values.astype(int)  # 是否为行为样本
                type_labels_raw = dataset.df.iloc[batch_indices]['type_label'].values  # 原始类型标签
                brand_labels_raw = dataset.df.iloc[batch_indices]['brand_label'].values  # 原始品牌标签
                device_labels_raw = dataset.df.iloc[batch_indices]['device_label'].values  # 原始型号标签

                for i in range(len(type_pred)):
                    max_type_prob = type_probs[i].max()  # 类型预测最大概率
                    max_brand_prob = brand_probs[i].max()  # 品牌预测最大概率
                    max_device_prob = device_probs[i].max()  # 型号预测最大概率

                    # 根据置信度阈值确定预测标签
                    pred_type_label = idx2type[type_pred[i]] if max_type_prob >= confidence_threshold else "unknown"
                    pred_brand_label = idx2brand[brand_pred[i]] if max_brand_prob >= confidence_threshold else "unknown"
                    pred_device_label = idx2device[pred_device.argmax(1)[i].item()] if max_device_prob >= device_threshold else "unknown"

                    # 检查原始标签是否有效（存在于标签字典中）
                    type_idx = -1 if type_labels_raw[i] not in dataset.type2idx else dataset.type2idx[type_labels_raw[i]]
                    brand_idx = -1 if brand_labels_raw[i] not in dataset.brand2idx else dataset.brand2idx[brand_labels_raw[i]]
                    device_idx = -1 if device_labels_raw[i] not in dataset.device2idx else dataset.device2idx[device_labels_raw[i]]

                    # 获取真实标签（无效标签设为"unknown"）
                    true_type_label = idx2type.get(type_idx, "unknown")
                    true_brand_label = idx2brand.get(brand_idx, "unknown")
                    true_device_label = idx2device.get(device_idx, "unknown")

                    # 统计无效标签样本
                    if type_idx == -1 or brand_idx == -1 or device_idx == -1:
                        invalid_label_samples += 1

                    # 保存预测标签（仅当置信度和真实标签有效时）
                    type_preds.append(type_pred[i] if max_type_prob >= confidence_threshold and type_idx != -1 else -1)
                    brand_preds.append(brand_pred[i] if max_brand_prob >= confidence_threshold and brand_idx != -1 else -1)
                    type_labels.append(type_idx)
                    brand_labels.append(brand_idx)

                    # 统计正确预测（仅对有效标签且置信度达标）
                    if max_type_prob >= confidence_threshold and type_pred[i] == type_idx and type_idx != -1:
                        type_correct += 1
                        if is_behavior[i] == 0:
                            idle_type_correct += 1
                        else:
                            behavior_type_correct += 1
                    if max_brand_prob >= confidence_threshold and brand_pred[i] == brand_idx and brand_idx != -1:
                        brand_correct += 1
                        if is_behavior[i] == 0:
                            idle_brand_correct += 1
                        else:
                            behavior_brand_correct += 1
                    # 统计未知型号（置信度低于阈值）
                    if max_device_prob < device_threshold:
                        device_unknown += 1

                    # 保存样本结果
                    results.append({
                        'index': total_samples + i,  # 样本全局索引
                        'true_type': true_type_label,
                        'pred_type': pred_type_label,
                        'type_prob': max_type_prob,
                        'true_brand': true_brand_label,
                        'pred_brand': pred_brand_label,
                        'brand_prob': max_brand_prob,
                        'pred_device': pred_device_label,
                        'device_prob': max_device_prob,
                        'is_behavior': is_behavior[i]
                    })
                    
                    # 统计闲时和行为样本
                    if is_behavior[i] == 0:
                        idle_samples += 1
                    else:
                        behavior_samples += 1

                total_samples += len(type_pred)  # 更新总样本数
                logger.info(f"Batch {batch_idx}: Processed {len(type_pred)} samples, Idle samples: {idle_samples}, Behavior samples: {behavior_samples}, Invalid labels: {invalid_label_samples}")
            except Exception as e:
                logger.warning(f"Batch {batch_idx} processing failed: {e}")
                invalid_samples += batch_size
                continue

    if total_samples == 0:
        logger.error("No valid samples processed in unknown set evaluation")
        return None, None

    # 计算指标
    type_acc = type_correct / total_samples if total_samples > 0 else 0.0  # 类型准确率
    brand_acc = brand_correct / total_samples if total_samples > 0 else 0.0  # 品牌准确率
    device_unknown_rate = device_unknown / total_samples if total_samples > 0 else 0.0  # 未知型号比例
    idle_type_acc = idle_type_correct / idle_samples if idle_samples > 0 else 0.0  # 闲时类型准确率
    idle_brand_acc = idle_brand_correct / idle_samples if idle_samples > 0 else 0.0  # 闲时品牌准确率
    behavior_type_acc = behavior_type_correct / behavior_samples if behavior_samples > 0 else 0.0  # 行为类型准确率
    behavior_brand_acc = behavior_brand_correct / behavior_samples if behavior_samples > 0 else 0.0  # 行为品牌准确率
    # 计算F1分数，排除无效标签
    type_f1 = f1_score([t for t in type_labels if t != -1], [p for p, t in zip(type_preds, type_labels) if t != -1],
                       average='macro', labels=[i for i in range(len(idx2type))], zero_division=0)
    brand_f1 = f1_score([b for b in brand_labels if b != -1], [p for p, b in zip(brand_preds, brand_labels) if b != -1],
                        average='macro', labels=[i for i in range(len(idx2brand))], zero_division=0)

    # 汇总指标
    metrics = {
        'dataset': dataset_name,
        'type_acc': type_acc,
        'type_f1': type_f1,
        'brand_acc': brand_acc,
        'brand_f1': brand_f1,
        'device_unknown_rate': device_unknown_rate,
        'idle_type_acc': idle_type_acc,
        'idle_brand_acc': idle_brand_acc,
        'behavior_type_acc': behavior_type_acc,
        'behavior_brand_acc': behavior_brand_acc,
        'total_samples': total_samples,
        'idle_samples': idle_samples,
        'behavior_samples': behavior_samples,
        'invalid_samples': invalid_samples,
        'invalid_label_samples': invalid_label_samples
    }

    logger.info(f"Processed {total_samples} samples, skipped {invalid_samples} invalid samples, invalid label samples: {invalid_label_samples}")
    return metrics, results

# --------------------------- 主逻辑 ---------------------------
def main():
    logger.info("Starting evaluation...")
    logger.info(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")  # 记录设备信息

    # 检查文件存在性
    for path in [test_csv_path, unknown_csv_path, label_dict_dir, model_path]:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            sys.exit(1)

    # 加载标签字典（type2idx.json等）
    try:
        with open(Path(label_dict_dir) / "type2idx.json") as f:
            type2idx = json.load(f)  # 类型标签到索引的映射
        with open(Path(label_dict_dir) / "brand2idx.json") as f:
            brand2idx = json.load(f)  # 品牌标签到索引的映射
        with open(Path(label_dict_dir) / "device2idx.json") as f:
            device2idx = json.load(f)  # 型号标签到索引的映射
        idx2type = {v: k for k, v in type2idx.items()}  # 反向映射：索引到类型
        idx2brand = {v: k for k, v in brand2idx.items()}  # 反向映射：索引到品牌
        idx2device = {v: k for k, v in device2idx.items()}  # 反向映射：索引到型号
        logger.info(f"Label dictionaries loaded: {len(type2idx)} types, {len(brand2idx)} brands, {len(device2idx)} devices")
    except Exception as e:
        logger.error(f"Failed to load label dictionaries: {e}")
        sys.exit(1)

    # 加载并检查测试集
    required_columns = ['type_label', 'brand_label', 'device_label', 'is_behavior', 'stat_feature_path', 'seq_embed_feature_path', 'raw_embed_feature_path']
    try:
        test_df = pd.read_csv(test_csv_path)  # 加载测试集CSV
        logger.info(f"Test dataset columns: {list(test_df.columns)}")
        missing_cols = [col for col in required_columns if col not in test_df.columns]
        if missing_cols:
            logger.error(f"Missing columns in 3_us_test.csv: {missing_cols}")
            sys.exit(1)
        if test_df[required_columns].isnull().any().any():
            logger.warning("Null values found in test dataset, dropping rows with null labels")
            test_df = test_df.dropna(subset=required_columns)  # 删除缺失值的行
    except Exception as e:
        logger.error(f"Failed to load 3_us_test.csv: {e}")
        sys.exit(1)

    # 加载并检查未知集
    try:
        unknown_df = pd.read_csv(unknown_csv_path)  # 加载未知集CSV
        logger.info(f"Unknown dataset columns: {list(unknown_df.columns)}")
        missing_cols = [col for col in required_columns if col not in unknown_df.columns]
        if missing_cols:
            logger.error(f"Missing columns in 3_us_unknown.csv: {missing_cols}")
            sys.exit(1)
        if unknown_df[required_columns].isnull().any().any():
            logger.warning("Null values found in unknown dataset, dropping rows with null labels")
            unknown_df = unknown_df.dropna(subset=required_columns)  # 删除缺失值的行
    except Exception as e:
        logger.error(f"Failed to load 3_us_unknown.csv: {e}")
        sys.exit(1)

    # 检查 is_behavior 分布
    logger.info(f"Test dataset is_behavior distribution: {test_df['is_behavior'].value_counts().to_dict()}")
    logger.info(f"Unknown dataset is_behavior distribution: {unknown_df['is_behavior'].value_counts().to_dict()}")

    # 采样测试集
    try:
        sampled_test_df = test_df.groupby('type_label', group_keys=True).apply(
            lambda x: x.sample(frac=test_sample_frac, random_state=42)  # 按类型分组采样35%
        ).reset_index(drop=True)
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        sampled_test_df.to_csv(os.path.join(output_dir, 'sampled_test.csv'), index=False)  # 保存采样后的测试集
        logger.info(f"Sampled test dataset columns: {list(sampled_test_df.columns)}")
        test_dataset = MultiModalIoTDataset(os.path.join(output_dir, 'sampled_test.csv'), root_stat, root_seq, root_raw, label_dict_dir)
        logger.info(f"Test dataset size (before sampling): {len(test_df)}")
        logger.info(f"Test dataset size (after sampling): {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load or sample test dataset: {e}")
        sys.exit(1)

    # 加载未知设备集
    try:
        unknown_df.to_csv(os.path.join(output_dir, 'processed_unknown.csv'), index=False)  # 保存未知集副本
        unknown_dataset = MultiModalIoTDataset(os.path.join(output_dir, 'processed_unknown.csv'), root_stat, root_seq, root_raw, label_dict_dir)
        logger.info(f"Unknown dataset size: {len(unknown_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load unknown dataset: {e}")
        sys.exit(1)

    # 创建DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集加载器，不打乱顺序
    unknown_dataloader = DataLoader(unknown_dataset, batch_size=batch_size, shuffle=False)  # 未知集加载器，不打乱顺序

    # 初始化模型
    try:
        model = MultiTaskClassifier(input_dim=159, hidden_dim=256,
                                   num_type=len(test_dataset.type2idx),
                                   num_brand=len(test_dataset.brand2idx),
                                   num_device=len(test_dataset.device2idx)).to(device)  # 初始化多任务分类器
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # 加载预训练权重
        logger.info(f"Model loaded successfully: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    loss_fn = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    # 评估测试集
    try:
        test_metrics, test_results = evaluate_known_dataset(test_dataset, test_dataloader, model, loss_fn, device, "Test Set", idx2type, idx2brand, idx2device)
        if test_metrics is None:
            logger.error("Test set evaluation returned no valid results")
            sys.exit(1)
        logger.info(f"📊 Test Set Results: Loss={test_metrics['loss']:.4f}, "
                    f"Type Acc={test_metrics['type_acc']:.4f}, Type F1={test_metrics['type_f1']:.4f}, "
                    f"Brand Acc={test_metrics['brand_acc']:.4f}, Brand F1={test_metrics['brand_f1']:.4f}, "
                    f"Device Acc={test_metrics['device_acc']:.4f}, Device F1={test_metrics['device_f1']:.4f}, "
                    f"Total Samples={test_metrics['total_samples']}, Invalid Samples={test_metrics['invalid_samples']}")
        pd.DataFrame(test_results).to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)  # 保存测试集结果
    except Exception as e:
        logger.error(f"Test set evaluation failed: {e}")
        sys.exit(1)

    # 评估未知设备
    try:
        unknown_metrics, unknown_results = evaluate_unknown_dataset(
            unknown_dataset, unknown_dataloader, model, device, "Unknown Set", confidence_threshold, device_threshold, idx2type, idx2brand, idx2device
        )
        if unknown_metrics is None:
            logger.error("Unknown set evaluation returned no valid results")
            sys.exit(1)
        logger.info(f"📊 Unknown Set Results: Type Acc={unknown_metrics['type_acc']:.4f}, Type F1={unknown_metrics['type_f1']:.4f}, "
                    f"Brand Acc={unknown_metrics['brand_acc']:.4f}, Brand F1={unknown_metrics['brand_f1']:.4f}, "
                    f"Device Unknown Rate={unknown_metrics['device_unknown_rate']:.4f}, "
                    f"Idle Type Acc={unknown_metrics['idle_type_acc']:.4f}, Idle Brand Acc={unknown_metrics['idle_brand_acc']:.4f}, "
                    f"Behavior Type Acc={unknown_metrics['behavior_type_acc']:.4f}, Behavior Brand Acc={unknown_metrics['behavior_brand_acc']:.4f}, "
                    f"Total Samples={unknown_metrics['total_samples']}, Idle Samples={unknown_metrics['idle_samples']}, "
                    f"Behavior Samples={unknown_metrics['behavior_samples']}, Invalid Samples={unknown_metrics['invalid_samples']}, "
                    f"Invalid Label Samples={unknown_metrics['invalid_label_samples']}")
        pd.DataFrame(unknown_results).to_csv(os.path.join(output_dir, 'unknown_results.csv'), index=False)  # 保存未知集结果
    except Exception as e:
        logger.error(f"Unknown set evaluation failed: {e}")
        raise

    # 保存指标汇总
    metrics_df = pd.DataFrame([test_metrics, unknown_metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)  # 保存指标汇总
    logger.info(f"Metrics summary saved to: {os.path.join(output_dir, 'metrics_summary.csv')}")

if __name__ == "__main__":
    try:
        logger.info("Script started.")
        main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)