# xiaorong_evaluate_multitask_classifier.py
# 📆 多任务 IoT 设备识别 — 批量评估脚本（F0–F12），兼容多命名模式
#
# 功能说明：
# - 自动枚举训练产出的所有模型（…/12_expB_outputs/<dataset>/models/*.pt）；
#   兼容三种命名：
#     1) optimized_multitask_model__F12__seed0.pt                         # 旧版
#     2) optimized_multitask_model__F12__concat__seed0.pt                 # 3.2 融合
#     3) optimized_multitask_model__F12__seed0__tasks[0B0].pt             # 3.3 任务子集
# - 读取同名 .json（若存在）覆盖/补充 feat_combo、fusion、tasks 等元信息；
# - 构造对应 MultiModalIoTDataset（与训练一致），分别评估“测试集/未知集”，落盘结果。
#
# 输出：
# - 每个模型一个目录：…/12_expB_outputs/<dataset>/eval/Fxx[_fusion][_tasksX]_seedY/
#   - test_results.csv, unknown_results.csv, metrics_summary.csv, eval_log.txt
# - 总汇总：…/12_expB_outputs/<dataset>/eval/_all_models_summary.csv

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

# ★ 与训练相同的数据集类（自动处理自监督/对比路径与维度对齐）
from xiaorong_optimized_multimodal_dataset import MultiModalIoTDataset

# --------------------------- 可配置区 ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"
DATASET   = "uk"  # ← 切换数据集时改这里：uk / us / cicIoT2022

TEST_CSV    = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_test.csv"
UNKNOWN_CSV = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_unknown.csv"
LABEL_DIR   = f"{DATA_ROOT}/11_multitask_training/{DATASET}"

# 训练产出（与训练脚本一致）
OUT_DIR    = f"{DATA_ROOT}/12_expB_outputs/{DATASET}"
MODEL_DIR  = f"{OUT_DIR}/3_2_models"   # 会遍历其中的每一个模型，或者改成3_3_models
EVAL_ROOT  = f"{OUT_DIR}/3_2_eval"     # 全部评估结果的根目录，或者改成3_2_eval
os.makedirs(EVAL_ROOT, exist_ok=True)

BATCH_SIZE        = 64
NUM_WORKERS       = 4
TEST_SAMPLE_FRAC  = 0.35   # 测试集采样比例
CONF_TYPE_BRAND   = 0.5    # 未知集：type/brand 置信度阈值
CONF_DEVICE       = 0.3    # 未知集：device 置信度阈值
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh  = logging.FileHandler(str(log_file)); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

# --------------------------- 模型结构（与训练一致） ---------------------------
class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.fc   = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_type   = nn.Linear(hidden_dim, num_type)
        self.cls_brand  = nn.Linear(hidden_dim, num_brand)
        self.cls_device = nn.Linear(hidden_dim, num_device)

    def forward(self, x):
        stats = x[:, :31]
        both  = x[:, 31:]
        idle_embed = both[:, :128]
        beh_embed  = both[:, 128:]
        is_behavior = stats[:, 30:31]
        gate = self.gate(is_behavior)
        weighted = gate*beh_embed + (1-gate)*idle_embed
        fused = torch.cat([stats, weighted], dim=1).unsqueeze(1)
        h = self.fc(fused)
        h = self.encoder(h).squeeze(1)
        return self.cls_type(h), self.cls_brand(h), self.cls_device(h)

# --------------------------- 评估工具函数（原样） ---------------------------
def evaluate_known_dataset(dataset, dataloader, model, loss_fn, device, dataset_name, idx2type, idx2brand, idx2device, logger):
    model.eval()
    total_loss = 0
    type_correct = brand_correct = device_correct = 0
    total_samples = 0
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    device_preds, device_labels = [], []
    results = []
    invalid_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch
                if x is None:
                    invalid_samples += BATCH_SIZE
                    continue
                x = x.to(device); y_type = y_type.to(device); y_brand = y_brand.to(device); y_device = y_device.to(device)

                pred_type, pred_brand, pred_device = model(x)
                loss = (loss_fn(pred_type, y_type.argmax(1)) +
                        loss_fn(pred_brand, y_brand.argmax(1)) +
                        loss_fn(pred_device, y_device.argmax(1)))
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

                type_pred  = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()
                device_pred= pred_device.argmax(1).cpu().numpy()
                type_label  = y_type.argmax(1).cpu().numpy()
                brand_label = y_brand.argmax(1).cpu().numpy()
                device_label= y_device.argmax(1).cpu().numpy()

                type_probs  = torch.softmax(pred_type,  dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs= torch.softmax(pred_device,dim=1).cpu().numpy()

                type_correct  += (type_pred  == type_label).sum()
                brand_correct += (brand_pred == brand_label).sum()
                device_correct+= (device_pred== device_label).sum()

                type_preds.extend(type_pred);     type_labels.extend(type_label)
                brand_preds.extend(brand_pred);   brand_labels.extend(brand_label)
                device_preds.extend(device_pred); device_labels.extend(device_label)

                base_idx = total_samples - len(type_pred)
                for i in range(len(type_pred)):
                    results.append({
                        'index': base_idx + i,
                        'true_type':  idx2type.get(type_label[i], "unknown"),
                        'pred_type':  idx2type[type_pred[i]],
                        'type_prob':  type_probs[i].max(),
                        'true_brand': idx2brand.get(brand_label[i], "unknown"),
                        'pred_brand': idx2brand[brand_pred[i]],
                        'brand_prob': brand_probs[i].max(),
                        'true_device': idx2device.get(device_label[i], "unknown"),
                        'pred_device': idx2device[device_pred[i]],
                        'device_prob': device_probs[i].max(),
                        'is_behavior': dataset.df.iloc[base_idx + i]['is_behavior']
                    })
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_samples += BATCH_SIZE
                continue

    if total_samples == 0:
        logger.error("No valid samples in known set.")
        return None, None

    avg_loss = total_loss / total_samples
    type_acc   = type_correct   / total_samples
    brand_acc  = brand_correct  / total_samples
    device_acc = device_correct / total_samples
    type_f1  = f1_score(type_labels,   type_preds,   average='macro', zero_division=0)
    brand_f1 = f1_score(brand_labels,  brand_preds,  average='macro', zero_division=0)
    device_f1= f1_score(device_labels, device_preds, average='macro', zero_division=0)

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


def evaluate_unknown_dataset(dataset, dataloader, model, device, dataset_name,
                             conf_type_brand, conf_device, idx2type, idx2brand, idx2device, logger):
    model.eval()
    total_samples = idle_samples = behavior_samples = 0
    type_correct = brand_correct = device_unknown = 0
    idle_type_correct = idle_brand_correct = 0
    behavior_type_correct = behavior_brand_correct = 0
    results = []
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    invalid_samples = invalid_label_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch
                if x is None:
                    invalid_samples += BATCH_SIZE
                    continue
                x = x.to(device); y_type = y_type.to(device); y_brand = y_brand.to(device)

                pred_type, pred_brand, pred_device = model(x)
                type_probs  = torch.softmax(pred_type,  dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs= torch.softmax(pred_device,dim=1).cpu().numpy()

                type_pred  = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()
                type_label  = y_type.argmax(1).cpu().numpy()
                brand_label = y_brand.argmax(1).cpu().numpy()

                base_idx = total_samples
                total_samples += len(type_pred)
                is_behavior = dataset.df.iloc[base_idx: total_samples]['is_behavior'].values.astype(int)
                type_labels_raw   = dataset.df.iloc[base_idx: total_samples]['type_label'].values
                brand_labels_raw  = dataset.df.iloc[base_idx: total_samples]['brand_label'].values
                device_labels_raw = dataset.df.iloc[base_idx: total_samples]['device_label'].values

                for i in range(len(type_pred)):
                    mt = type_probs[i].max()
                    mb = brand_probs[i].max()
                    md = device_probs[i].max()

                    pred_type_label  = idx2type[type_pred[i]]   if mt >= conf_type_brand else "unknown"
                    pred_brand_label = idx2brand[brand_pred[i]] if mb >= conf_type_brand else "unknown"
                    pred_device_label= idx2device[pred_device.argmax(1)[i].item()] if md >= conf_device else "unknown"

                    ti = -1 if type_labels_raw[i]   not in dataset.type2idx   else dataset.type2idx[type_labels_raw[i]]
                    bi = -1 if brand_labels_raw[i]  not in dataset.brand2idx  else dataset.brand2idx[brand_labels_raw[i]]
                    di = -1 if device_labels_raw[i] not in dataset.device2idx else dataset.device2idx[device_labels_raw[i]]

                    if ti == -1 or bi == -1 or di == -1:
                        invalid_label_samples += 1

                    type_preds.append(type_pred[i]  if (mt >= conf_type_brand and ti != -1) else -1)
                    brand_preds.append(brand_pred[i] if (mb >= conf_type_brand and bi != -1) else -1)
                    type_labels.append(ti); brand_labels.append(bi)

                    if (mt >= conf_type_brand and ti != -1 and type_pred[i] == ti):
                        type_correct += 1
                        if is_behavior[i] == 0: idle_type_correct += 1
                        else:                    behavior_type_correct += 1
                    if (mb >= conf_type_brand and bi != -1 and brand_pred[i] == bi):
                        brand_correct += 1
                        if is_behavior[i] == 0: idle_brand_correct += 1
                        else:                    behavior_brand_correct += 1
                    if md < conf_device:
                        device_unknown += 1

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

    type_acc  = type_correct  / total_samples
    brand_acc = brand_correct / total_samples
    unk_rate  = device_unknown/ total_samples
    idle_type_acc  = idle_type_correct  / idle_samples if idle_samples>0 else 0.0
    idle_brand_acc = idle_brand_correct / idle_samples if idle_samples>0 else 0.0
    beh_type_acc   = behavior_type_correct  / behavior_samples if behavior_samples>0 else 0.0
    beh_brand_acc  = behavior_brand_correct / behavior_samples if behavior_samples>0 else 0.0

    t_lab = [t for t in type_labels  if t != -1]
    t_pre = [p for p,t in zip(type_preds, type_labels)  if t != -1]
    b_lab = [b for b in brand_labels if b != -1]
    b_pre = [p for p,b in zip(brand_preds, brand_labels) if b != -1]
    type_f1  = f1_score(t_lab, t_pre, average='macro', labels=list(range(len(idx2type))),  zero_division=0) if t_lab else 0.0
    brand_f1 = f1_score(b_lab, b_pre, average='macro', labels=list(range(len(idx2brand))), zero_division=0) if b_lab else 0.0

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
    """
    返回列表：[{feat_key, seed, fusion, tasks, pt_path, cfg_path}]
    兼容三类命名（正则均可匹配）：
      1) optimized_multitask_model__F12__seed0.pt
      2) optimized_multitask_model__F12__concat__seed0.pt
      3) optimized_multitask_model__F12__seed0__tasks[0B0].pt
    """
    items = []
    p = Path(model_dir)
    # 统一正则：feat_key（必选）、fusion（可选）、seed（必选）、tasks（可选，放在 seed 后面）
    pat = re.compile(
        r"^optimized_multitask_model__"
        r"(F\d{1,2})"                      # 1: feat_key
        r"(?:__([a-z]+))?"                 # 2: fusion (optional)
        r"__seed(\d+)"                     # 3: seed
        r"(?:__tasks\[(.+?)\])?"           # 4: tasks (optional)
        r"\.pt$"
    )

    for pt in sorted(p.glob("optimized_multitask_model__*.pt")):
        m = pat.match(pt.name)
        if not m:
            continue
        feat_key = m.group(1)
        fusion   = m.group(2) or ""   # 可能没有
        seed     = int(m.group(3))
        tasks    = m.group(4) or ""   # 可能没有

        # cfg 优先匹配“同基名 .json”
        cfg_path = pt.with_suffix(".json")
        items.append({
            "feat_key": feat_key,
            "fusion": fusion,
            "seed": seed,
            "tasks": tasks,
            "pt_path": str(pt),
            "cfg_path": str(cfg_path if cfg_path.exists() else "")
        })
    return items


def load_meta_from_cfg_or_name(rec):
    """
    优先从 JSON 里读 feat_combo / fusion / tasks；没有就从文件名补；仍没有就给默认。
    返回：feat_combo, fusion, tasks
    """
    feat_key = rec["feat_key"]
    fusion   = rec["fusion"] or "gate"   # 默认 gate（和你之前的一致）
    tasks    = rec["tasks"] or None
    feat_combo = ABLATION_MAP.get(feat_key, "Stat")

    cfg_path = rec["cfg_path"]
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = json.loads(Path(cfg_path).read_text())
            if "feat_combo" in cfg: feat_combo = cfg["feat_combo"]
            if "fusion" in cfg and cfg["fusion"]: fusion = cfg["fusion"]
            if "tasks" in cfg and cfg["tasks"]:   tasks  = cfg["tasks"]
        except Exception:
            pass
    return feat_combo, fusion, tasks

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
    for rec in items:
        feat_combo, fusion, tasks = load_meta_from_cfg_or_name(rec)
        feat_key, seed, pt_path, cfg_path = rec["feat_key"], rec["seed"], rec["pt_path"], rec["cfg_path"]

        # 结果目录名：Fxx[_fusion][_tasksX]_seedY
        name_bits = [feat_key]
        if fusion: name_bits.append(fusion)
        if tasks:  name_bits.append(f"tasks[{tasks}]")
        name_bits.append(f"seed{seed}")
        save_dir = Path(EVAL_ROOT) / ("_".join(name_bits))
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = build_logger(save_dir / "eval_log.txt")

        logger.info("="*80)
        logger.info(f"Model: {pt_path}")
        logger.info(f"FeatKey={feat_key}, Seed={seed}, FeatCombo={feat_combo}, Fusion={fusion or 'gate'}, Tasks={tasks or 'ALL'}")
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

        # 模型定义 & 加载权重（融合策略不影响评估结构，因为评估这版使用 gate-Transformer 模型）
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

        loss_fn = nn.CrossEntropyLoss()

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
            "feat_key": feat_key,
            "seed": seed,
            "feat_combo": feat_combo,
            "fusion": fusion or "gate",
            "tasks": tasks or "ALL",
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
