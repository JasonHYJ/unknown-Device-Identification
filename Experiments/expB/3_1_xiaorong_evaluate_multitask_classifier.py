# xiaorong_evaluate_multitask_classifier.py
# 📆 多任务 IoT 设备识别 — 批量评估脚本（F0–F12）
#
# 功能说明：
# - 自动枚举训练产出的所有模型（…/12_expB_outputs/<dataset>/models/*.pt），
#   读取同名 .json 配置确定 feat_combo（没有就从 Fxx 推断），
#   构造对应的 MultiModalIoTDataset（与训练一致），
#   依次评估“测试集/未知集”，并将结果保存到独立目录。
#
# 输入：
# - CSV：3_<dataset>_test.csv、3_<dataset>_unknown.csv（列中含绝对路径）
# - 标签字典：type2idx.json / brand2idx.json / device2idx.json
# - 模型与配置：…/12_expB_outputs/<dataset>/models/optimized_multitask_model__Fxx__seedY.pt(+.json)
#
# 输出：
# - 每个模型一个目录：…/12_expB_outputs/<dataset>/eval/Fxx_seedY/
#   - test_results.csv, unknown_results.csv, metrics_summary.csv, eval_log.txt
# - 还有一个总汇总：…/12_expB_outputs/<dataset>/eval/_all_models_summary.csv
#
# 评估指标：
# - 已知集：Loss、Acc、macro-F1（type/brand/device）
# - 未知集：Acc、macro-F1（type/brand），UnknownRate（device），
#           以及 idle/behavior 分拆 Acc。

import os
import re
import sys
import json
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# ★ 使用与训练相同的 Dataset 类（自动处理自监督/对比路径与维度对齐）
from xiaorong_optimized_multimodal_dataset import MultiModalIoTDataset


# --------------------------- 可配置区 ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"
DATASET   = "uk"  # ← 需要换数据集时改这里：uk / us / cicIoT2022

TEST_CSV    = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_test.csv"
UNKNOWN_CSV = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_unknown.csv"
LABEL_DIR   = f"{DATA_ROOT}/11_multitask_training/{DATASET}"

# 训练产出（与训练脚本一致）
OUT_DIR    = f"{DATA_ROOT}/12_expB_outputs/{DATASET}"
MODEL_DIR  = f"{OUT_DIR}/models"    # 模型保存的目录，会遍历其中的每一个模型得到结果
EVAL_ROOT  = f"{OUT_DIR}/eval"  # 全部评估结果的根目录
os.makedirs(EVAL_ROOT, exist_ok=True)

BATCH_SIZE        = 64  # 批次大小
NUM_WORKERS       = 4
TEST_SAMPLE_FRAC  = 0.35   # 测试集采样比例
CONF_TYPE_BRAND   = 0.5   # 未知集：type/brand 置信度阈值
CONF_DEVICE       = 0.3   # 未知集：device 置信度阈值
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # 选择设备（优先GPU）

# Ablation 组名到组合映射（与训练保持一致，缺少 .json 时用它来推断）
ABLATION_MAP = {
    "F0":  "Stat",
    "F1":  "Seq-Embed",
    "F2":  "Raw-Embed",
    "F3":  "Seq-Embed+Raw-Embed",
    "F4":  "Stat+Seq-Embed",
    "F5":  "Stat+Raw-Embed",
    "F6":  "Stat+Seq-Embed+Raw-Embed",
    "F7":  "Seq-CL",
    "F8":  "Raw-CL",
    "F9":  "Seq-CL+Raw-CL",
    "F10": "Stat+Seq-CL",
    "F11": "Stat+Raw-CL",
    "F12": "Stat+Seq-CL+Raw-CL",
}


# --------------------------- 日志 ---------------------------
def build_logger(log_file: Path):
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh  = logging.FileHandler(str(log_file))
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# --------------------------- 模型结构（与训练一致） ---------------------------
class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        # 门控网络：根据is_behavior决定使用闲时还是行为嵌入
        self.gate = nn.Sequential(
            nn.Linear(1, 1),  # 输入is_behavior（1维），输出权重
            nn.Sigmoid()  # 输出[0,1]权重
        )
        self.fc   = nn.Linear(input_dim, hidden_dim)    # 全连接层：输入159维（31+128），输出256维
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)   # 2层Transformer
        self.cls_type   = nn.Linear(hidden_dim, num_type)   # 类型分类头
        self.cls_brand  = nn.Linear(hidden_dim, num_brand)  # 品牌分类头
        self.cls_device = nn.Linear(hidden_dim, num_device) # 型号分类头

    def forward(self, x):
        stats = x[:, :31]   # 提取统计特征（31维）
        both  = x[:, 31:]   # 提取嵌入特征（256维：128闲时+128行为）
        idle_embed = both[:, :128]  # 闲时嵌入（128维）
        beh_embed  = both[:, 128:]  # 行为嵌入（128维）
        is_behavior = stats[:, 30:31]   # 提取is_behavior标志（stat_vec[30]）

        gate = self.gate(is_behavior)   # 计算门控权重（0~1）
        weighted = gate*beh_embed + (1-gate)*idle_embed # 融合闲时和行为嵌入
        fused = torch.cat([stats, weighted], dim=1).unsqueeze(1)  # (B,1,159)，拼接统计特征和加权嵌入，增加序列维度
        h = self.fc(fused)  # 全连接层转换
        h = self.encoder(h).squeeze(1)  # Transformer编码并移除序列维度
        return self.cls_type(h), self.cls_brand(h), self.cls_device(h)


# --------------------------- 评估工具函数 ---------------------------
def evaluate_known_dataset(dataset, dataloader, model, loss_fn, device, dataset_name, idx2type, idx2brand, idx2device, logger):
    model.eval()    # 设置模型为评估模式
    total_loss = 0
    type_correct = brand_correct = device_correct = 0
    total_samples = 0
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    device_preds, device_labels = [], []
    results = []
    invalid_samples = 0

    with torch.no_grad():   # 禁用梯度计算以节省内存
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch    # 获取批次数据：输入特征和one-hot标签
                if x is None:
                    invalid_samples += BATCH_SIZE
                    continue
                x = x.to(device)    # 将输入移到GPU/CPU
                y_type = y_type.to(device)
                y_brand = y_brand.to(device)
                y_device = y_device.to(device)

                pred_type, pred_brand, pred_device = model(x)   # 模型前向传播
                # 计算损失：类型、品牌、型号的交叉熵损失之和
                loss = loss_fn(pred_type, y_type.argmax(1)) + \
                       loss_fn(pred_brand, y_brand.argmax(1)) + \
                       loss_fn(pred_device, y_device.argmax(1))
                total_loss += loss.item() * x.size(0)   # 累加批次损失
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
                type_correct  += (type_pred  == type_label ).sum()
                brand_correct += (brand_pred == brand_label).sum()
                device_correct+= (device_pred== device_label).sum()

                # 保存预测和真实标签用于F1计算
                type_preds.extend(type_pred);    type_labels.extend(type_label)
                brand_preds.extend(brand_pred);  brand_labels.extend(brand_label)
                device_preds.extend(device_pred);device_labels.extend(device_label)

                # 保存每样本
                base_idx = total_samples - len(type_pred)
                for i in range(len(type_pred)):
                    results.append({
                        'index': base_idx + i,  # 真实型号标签
                        'true_type': idx2type.get(type_label[i], "unknown"),    # 真实类型标签
                        'pred_type': idx2type[type_pred[i]],    # 预测类型标签
                        'type_prob': type_probs[i].max(),   # 类型最大概率
                        'true_brand': idx2brand.get(brand_label[i], "unknown"), # 真实品牌标签
                        'pred_brand': idx2brand[brand_pred[i]], # 预测品牌标签
                        'brand_prob': brand_probs[i].max(), # 品牌最大概率
                        'true_device': idx2device.get(device_label[i], "unknown"),  # 真实型号标签
                        'pred_device': idx2device[device_pred[i]],  # 预测型号标签
                        'device_prob': device_probs[i].max(),   # 型号最大概率
                        'is_behavior': dataset.df.iloc[base_idx + i]['is_behavior'] # 是否为行为样本
                    })
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_samples += BATCH_SIZE
                continue

    if total_samples == 0:
        logger.error("No valid samples in known set.")
        return None, None

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    type_acc   = type_correct   / total_samples
    brand_acc  = brand_correct  / total_samples
    device_acc = device_correct / total_samples
    # 计算F1分数，忽略无效标签
    type_f1  = f1_score(type_labels,   type_preds,   average='macro', zero_division=0)
    brand_f1 = f1_score(brand_labels,  brand_preds,  average='macro', zero_division=0)
    device_f1= f1_score(device_labels, device_preds, average='macro', zero_division=0)

    # 汇总指标
    metrics = {
        'dataset': dataset_name,
        'loss': avg_loss,
        'type_acc': type_acc,   'type_f1': type_f1,
        'brand_acc': brand_acc, 'brand_f1': brand_f1,
        'device_acc': device_acc, 'device_f1': device_f1,
        'total_samples': total_samples,
        'invalid_samples': invalid_samples
    }
    logger.info(f"[Known] N={total_samples}  Loss={avg_loss:.4f}  "
                f"Type A/F1={type_acc:.4f}/{type_f1:.4f}  "
                f"Brand A/F1={brand_acc:.4f}/{brand_f1:.4f}  "
                f"Device A/F1={device_acc:.4f}/{device_f1:.4f}")
    return metrics, results


# --------------------------- 评估未知设备 ---------------------------
def evaluate_unknown_dataset(dataset, dataloader, model, device, dataset_name,
                             conf_type_brand, conf_device, idx2type, idx2brand, idx2device, logger):
    model.eval()    # 设置模型为评估模式
    total_samples = idle_samples = behavior_samples = 0
    type_correct = brand_correct = device_unknown = 0
    idle_type_correct = idle_brand_correct = 0
    behavior_type_correct = behavior_brand_correct = 0
    results = []
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    invalid_samples = invalid_label_samples = 0

    with torch.no_grad():   # 禁用梯度计算
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch    # 获取批次数据
                if x is None:
                    invalid_samples += BATCH_SIZE
                    continue
                x = x.to(device);  y_type = y_type.to(device);  y_brand = y_brand.to(device)    # 将输入移到GPU/CPU

                pred_type, pred_brand, pred_device = model(x)   # 模型前向传播
                # 计算预测概率
                type_probs = torch.softmax(pred_type, dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs= torch.softmax(pred_device, dim=1).cpu().numpy()
                
                # 获取预测标签
                type_pred  = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()
                type_label = y_type.argmax(1).cpu().numpy()
                brand_label= y_brand.argmax(1).cpu().numpy()

                # 从数据集获取 is_behavior 和原始标签
                base_idx = total_samples
                total_samples += len(type_pred)
                is_behavior = dataset.df.iloc[base_idx: total_samples]['is_behavior'].values.astype(int)    # 是否为行为样本
                type_labels_raw  = dataset.df.iloc[base_idx: total_samples]['type_label'].values    # 原始类型标签
                brand_labels_raw = dataset.df.iloc[base_idx: total_samples]['brand_label'].values   # 原始品牌标签
                device_labels_raw= dataset.df.iloc[base_idx: total_samples]['device_label'].values  # 原始型号标签

                for i in range(len(type_pred)):
                    mt = type_probs[i].max()    # 类型预测最大概率
                    mb = brand_probs[i].max()   # 品牌预测最大概率
                    md = device_probs[i].max()  # 型号预测最大概率

                    # 根据置信度阈值确定预测标签
                    pred_type_label  = idx2type[type_pred[i]]   if mt >= conf_type_brand else "unknown"
                    pred_brand_label = idx2brand[brand_pred[i]] if mb >= conf_type_brand else "unknown"
                    pred_device_label= idx2device[pred_device.argmax(1)[i].item()] if md >= conf_device else "unknown"

                    # map raw labels to indices; if unseen => -1
                    ti = -1 if type_labels_raw[i]  not in dataset.type2idx  else dataset.type2idx[type_labels_raw[i]]
                    bi = -1 if brand_labels_raw[i] not in dataset.brand2idx else dataset.brand2idx[brand_labels_raw[i]]
                    di = -1 if device_labels_raw[i]not in dataset.device2idx else dataset.device2idx[device_labels_raw[i]]

                    # 统计无效标签样本
                    if ti == -1 or bi == -1 or di == -1:
                        invalid_label_samples += 1

                    # 保存预测标签（仅当置信度和真实标签有效时）
                    type_preds.append(type_pred[i]  if (mt >= conf_type_brand and ti != -1) else -1)
                    brand_preds.append(brand_pred[i] if (mb >= conf_type_brand and bi != -1) else -1)
                    type_labels.append(ti);  brand_labels.append(bi)

                    # 统计正确预测（仅对有效标签且置信度达标）
                    if (mt >= conf_type_brand and ti != -1 and type_pred[i] == ti):
                        type_correct += 1
                        if is_behavior[i] == 0: idle_type_correct += 1
                        else:                    behavior_type_correct += 1
                    if (mb >= conf_type_brand and bi != -1 and brand_pred[i] == bi):
                        brand_correct += 1
                        if is_behavior[i] == 0: idle_brand_correct += 1
                        else:                    behavior_brand_correct += 1

                    # 统计未知型号（置信度低于阈值）
                    if md < conf_device:
                        device_unknown += 1

                    # 保存样本结果
                    results.append({
                        'index': base_idx + i,
                        'true_type':  idx2type.get(ti, "unknown"),
                        'pred_type':  pred_type_label,
                        'type_prob':  mt,
                        'true_brand': idx2brand.get(bi, "unknown"),
                        'pred_brand': pred_brand_label,
                        'brand_prob': mb,
                        'pred_device': pred_device_label,
                        'device_prob': md,
                        'is_behavior': is_behavior[i]
                    })

                    # 统计闲时和行为样本
                    if is_behavior[i] == 0: idle_samples += 1
                    else:                    behavior_samples += 1

                logger.info(f"Batch {batch_idx}: cumN={total_samples}, idle={idle_samples}, beh={behavior_samples}, invalidLabel={invalid_label_samples}")

            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_samples += BATCH_SIZE
                continue

    if total_samples == 0:
        logger.error("No valid samples in unknown set.")
        return None, None
    
    # 计算指标
    type_acc  = type_correct  / total_samples   # 类型准确率
    brand_acc = brand_correct / total_samples   # 品牌准确率
    unk_rate  = device_unknown/ total_samples   # 未知型号比例
    idle_type_acc  = idle_type_correct  / idle_samples if idle_samples>0 else 0.0   # 闲时类型准确率
    idle_brand_acc = idle_brand_correct / idle_samples if idle_samples>0 else 0.0   # 闲时品牌准确率
    beh_type_acc   = behavior_type_correct  / behavior_samples if behavior_samples>0 else 0.0   # 行为类型准确率
    beh_brand_acc  = behavior_brand_correct / behavior_samples if behavior_samples>0 else 0.0   # 行为品牌准确率

    # macro-F1（过滤 -1）
    t_lab = [t for t in type_labels  if t != -1]
    t_pre = [p for p,t in zip(type_preds, type_labels)  if t != -1]
    b_lab = [b for b in brand_labels if b != -1]
    b_pre = [p for p,b in zip(brand_preds, brand_labels) if b != -1]
    type_f1  = f1_score(t_lab, t_pre, average='macro', labels=list(range(len(idx2type))),  zero_division=0) if t_lab else 0.0
    brand_f1 = f1_score(b_lab, b_pre, average='macro', labels=list(range(len(idx2brand))), zero_division=0) if b_lab else 0.0

    # 汇总指标
    metrics = {
        'dataset': dataset_name,
        'type_acc': type_acc, 'type_f1': type_f1,
        'brand_acc': brand_acc, 'brand_f1': brand_f1,
        'device_unknown_rate': unk_rate,
        'idle_type_acc': idle_type_acc, 'idle_brand_acc': idle_brand_acc,
        'behavior_type_acc': beh_type_acc, 'behavior_brand_acc': beh_brand_acc,
        'total_samples': total_samples,
        'idle_samples': idle_samples,
        'behavior_samples': behavior_samples,
        'invalid_samples': invalid_samples,
        'invalid_label_samples': invalid_label_samples
    }
    logger.info(f"[Unknown] N={total_samples}  "
                f"Type A/F1={type_acc:.4f}/{type_f1:.4f}  "
                f"Brand A/F1={brand_acc:.4f}/{brand_f1:.4f}  "
                f"UnkRate(device)={unk_rate:.4f} | idle A: T/B={idle_type_acc:.4f}/{idle_brand_acc:.4f} | "
                f"beh A: T/B={beh_type_acc:.4f}/{beh_brand_acc:.4f}")
    return metrics, results


# --------------------------- 扫描模型 & 读取配置 ---------------------------
def list_models(model_dir: str):
    """返回 [(feat_key, seed, pt_path, cfg_json_or_None)]"""
    items = []
    p = Path(model_dir)
    for pt in sorted(p.glob("optimized_multitask_model__F*__seed*.pt")):
        m = re.search(r"optimized_multitask_model__(F\d{1,2})__seed(\d+)\.pt$", pt.name)
        if not m: 
            continue
        feat_key, seed = m.group(1), int(m.group(2))
        cfg = pt.with_suffix(".json")  # 训练时同名 json
        items.append((feat_key, seed, str(pt), str(cfg if cfg.exists() else "")))
    return items


def load_feat_combo(feat_key: str, cfg_path: str):
    """优先读 cfg 中的 feat_combo；没有就按 feat_key 用 ABLATION_MAP 推断。"""
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = json.loads(Path(cfg_path).read_text())
            if "feat_combo" in cfg:
                return cfg["feat_combo"]
        except Exception:
            pass
    return ABLATION_MAP.get(feat_key, "Stat")


# --------------------------- 主流程：对每个模型评估 ---------------------------
def main():
    # 全局标签字典
    with open(Path(LABEL_DIR) / "type2idx.json") as f:
        type2idx = json.load(f)
    with open(Path(LABEL_DIR) / "brand2idx.json") as f:
        brand2idx = json.load(f)
    with open(Path(LABEL_DIR) / "device2idx.json") as f:
        device2idx = json.load(f)
    idx2type  = {v:k for k,v in type2idx.items()}
    idx2brand = {v:k for k,v in brand2idx.items()}
    idx2device= {v:k for k,v in device2idx.items()}

    # 载入 CSV 并可选采样
    test_df   = pd.read_csv(TEST_CSV)
    unknown_df= pd.read_csv(UNKNOWN_CSV)
    print(f"[INFO] Test is_behavior dist: {test_df['is_behavior'].value_counts().to_dict()}")
    print(f"[INFO] Unknown is_behavior dist: {unknown_df['is_behavior'].value_counts().to_dict()}")

    if TEST_SAMPLE_FRAC < 1.0:
        test_df = (test_df.groupby('type_label', group_keys=True)
                   .apply(lambda x: x.sample(frac=TEST_SAMPLE_FRAC, random_state=42))
                   .reset_index(drop=True))
        print(f"[INFO] Test sampled: {len(test_df)} rows")

    # 扫描模型
    items = list_models(MODEL_DIR)
    if not items:
        print(f"[ERROR] No model .pt found in: {MODEL_DIR}")
        sys.exit(1)
    print(f"[MAIN] Found {len(items)} models to evaluate.")

    # 汇总表
    summary_rows = []

    # 逐模型评估
    for feat_key, seed, pt_path, cfg_path in items:
        feat_combo = load_feat_combo(feat_key, cfg_path)
        save_dir   = Path(EVAL_ROOT) / f"{feat_key}_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = build_logger(save_dir / "eval_log.txt")

        logger.info("="*80)
        logger.info(f"Model: {pt_path}")
        logger.info(f"FeatKey={feat_key}, Seed={seed}, FeatCombo={feat_combo}")
        logger.info("="*80)

        # 落地一份当前使用的数据（方便复现）
        test_csv_out    = save_dir / "test_eval_input.csv"
        unknown_csv_out = save_dir / "unknown_eval_input.csv"
        test_df.to_csv(test_csv_out, index=False)
        unknown_df.to_csv(unknown_csv_out, index=False)

        # 构造 Dataset / Loader（与训练完全一致）
        test_ds    = MultiModalIoTDataset(csv_path=str(test_csv_out),    label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        unknown_ds = MultiModalIoTDataset(csv_path=str(unknown_csv_out), label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        test_loader    = DataLoader(test_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        unknown_loader = DataLoader(unknown_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # 模型定义 & 加载权重
        model = MultiTaskClassifier(
            input_dim=159, hidden_dim=256,
            num_type=len(test_ds.type2idx), 
            num_brand=len(test_ds.brand2idx), 
            num_device=len(test_ds.device2idx)
        ).to(DEVICE)
        try:
            state = torch.load(pt_path, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
            logger.info("Model weights loaded.")
        except Exception as e:
            logger.error(f"Load state_dict failed: {e}")
            continue

        loss_fn = nn.CrossEntropyLoss() # 定义交叉熵损失函数

        # 评估已知集
        known_metrics, known_rows = evaluate_known_dataset(
            test_ds, test_loader, model, loss_fn, DEVICE, "Test Set",
            idx2type, idx2brand, idx2device, logger
        )
        if known_metrics is None:
            logger.error("Known set evaluation returned no result, skip this model.")
            continue
        pd.DataFrame(known_rows).to_csv(save_dir / "test_results.csv", index=False)

        # 评估未知集
        unknown_metrics, unknown_rows = evaluate_unknown_dataset(
            unknown_ds, unknown_loader, model, DEVICE, "Unknown Set",
            CONF_TYPE_BRAND, CONF_DEVICE, idx2type, idx2brand, idx2device, logger
        )
        if unknown_metrics is None:
            logger.error("Unknown set evaluation returned no result, skip this model.")
            continue
        pd.DataFrame(unknown_rows).to_csv(save_dir / "unknown_results.csv", index=False)

        # 保存本模型的指标汇总
        metrics_df = pd.DataFrame([known_metrics, unknown_metrics])
        metrics_df.to_csv(save_dir / "metrics_summary.csv", index=False)
        logger.info(f"Saved per-model metrics to: {save_dir/'metrics_summary.csv'}")

        # 加一行到总汇总
        summary_rows.append({
            "feat_key": feat_key, "seed": seed, "feat_combo": feat_combo,
            "model_path": pt_path,
            **{f"known_{k}": v for k,v in known_metrics.items() if k!="dataset"},
            **{f"unknown_{k}": v for k,v in unknown_metrics.items() if k!="dataset"},
        })

    # 写入总汇总
    all_summary = pd.DataFrame(summary_rows)
    all_summary.to_csv(Path(EVAL_ROOT) / "_all_models_summary.csv", index=False)
    print(f"[DONE] Wrote overall summary: {Path(EVAL_ROOT) / '_all_models_summary.csv'}")


if __name__ == "__main__":
    main()
