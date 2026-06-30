# xiaorong_evaluate_multitask_classifier_open_set.py
# 📆 多任务 IoT 设备识别 — 批量评估脚本（仅 F12 + 开放集阈值标定）
#
# 功能概述：
# - 自动枚举 …/12_expB_outputs/<dataset>/models/ 下的 F12 模型（可改 EVAL_ONLY 控制组合）
# - 与训练一致地构造 MultiModalIoTDataset（根据模型配置或文件名推断 feat_combo）
# - 已知集（test）评估：Loss / Acc / macro-F1（type/brand/device）
# - 未知集（unknown）评估：Acc / macro-F1（type/brand）/ UnknownRate（device）/ idle vs behavior 分项 Acc
# - 【3.4】对 device 头进行**开放集阈值标定**：
#     * 先在“测试集/未知集”上前向一遍，收集 device 头最大 softmax 概率分布
#     * 扫描阈值 τ∈[0.05,0.95]，最大化 Youden J = TPR_unknown − FPR_known，得到最优 τ*
#     * 用 τ* 作为 unknown 评估中的“未知判别阈值”，并保存扫描曲线 device_threshold_sweep.csv
#
# 输入：
# - CSV：3_<dataset>_test.csv、3_<dataset>_unknown.csv（列中含绝对路径）
# - 标签字典：type2idx.json / brand2idx.json / device2idx.json
# - 模型与配置：…/12_expB_outputs/<dataset>/models/optimized_multitask_model__Fxx__seedY.pt(+.json)
#
# 输出（每个模型一个目录 …/12_expB_outputs/<dataset>/eval/Fxx_seedY/）：
# - test_results.csv, unknown_results.csv, metrics_summary.csv, device_threshold_sweep.csv, eval_log.txt
# - 总汇总：…/12_expB_outputs/<dataset>/eval/_all_models_summary.csv
#
# 只评估 F12（Stat+Seq-CL+Raw-CL）。如需扩展，只改 EVAL_ONLY 即可。

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

# ★ 与训练一致的数据集（自动处理自监督/对比路径与维度对齐）
from xiaorong_optimized_multimodal_dataset import MultiModalIoTDataset

# ============================ 可配置区 ============================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"
DATASET   = "uk"  # ← 换数据集时改：uk / us / cicIoT2022

TEST_CSV    = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_test.csv"
UNKNOWN_CSV = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_unknown.csv"
LABEL_DIR   = f"{DATA_ROOT}/11_multitask_training/{DATASET}"

# 训练产出（与训练脚本一致）
OUT_DIR    = f"{DATA_ROOT}/12_expB_outputs/{DATASET}"
MODEL_DIR  = f"{OUT_DIR}/models"    # 遍历其中的 pt/json
EVAL_ROOT  = f"{OUT_DIR}/3_4_eval"      # 评估结果根目录
os.makedirs(EVAL_ROOT, exist_ok=True)

BATCH_SIZE        = 64
NUM_WORKERS       = 4
TEST_SAMPLE_FRAC  = 1.0   # 保持 1.0：不开采样，确保阈值标定稳定；如需采样可改小
CONF_TYPE_BRAND   = 0.5   # 未知集：type/brand 置信度阈值（不做自动标定，保持简单）
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只评估这些消融组（默认仅 F12）
EVAL_ONLY = ["F12"]

# Ablation 名到组合（缺少 .json 时用它推断）
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

# ============================ 日志 ============================
def build_logger(log_file: Path):
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh  = logging.FileHandler(str(log_file)); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

# ============================ 模型结构（与训练一致） ============================
class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_type, num_brand, num_device):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())   # is_behavior → [0,1]
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
        h = torch.cat([stats, weighted], dim=1).unsqueeze(1)  # (B,1,159)
        h = self.fc(h)
        h = self.encoder(h).squeeze(1)
        return self.cls_type(h), self.cls_brand(h), self.cls_device(h)

# ============================ 评估工具函数 ============================
def evaluate_known_dataset(dataset, dataloader, model, loss_fn, device,
                           dataset_name, idx2type, idx2brand, idx2device, logger):
    model.eval()
    total_loss = 0
    type_correct = brand_correct = device_correct = 0
    total_samples = 0
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    device_preds, device_labels = [], []
    results = []
    invalid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                if batch is None:
                    invalid_batches += 1
                    continue
                x, y_type, y_brand, y_device = batch
                x = x.to(device); y_type = y_type.to(device); y_brand = y_brand.to(device); y_device = y_device.to(device)
                out_type, out_brand, out_device = model(x)

                loss = (loss_fn(out_type,  y_type.argmax(1))
                      + loss_fn(out_brand, y_brand.argmax(1))
                      + loss_fn(out_device, y_device.argmax(1)))
                total_loss    += loss.item() * x.size(0)
                total_samples += x.size(0)

                t_pred = out_type.argmax(1).cpu().numpy()
                b_pred = out_brand.argmax(1).cpu().numpy()
                d_pred = out_device.argmax(1).cpu().numpy()
                t_lab  = y_type.argmax(1).cpu().numpy()
                b_lab  = y_brand.argmax(1).cpu().numpy()
                d_lab  = y_device.argmax(1).cpu().numpy()

                t_prob = torch.softmax(out_type,  dim=1).cpu().numpy()
                b_prob = torch.softmax(out_brand, dim=1).cpu().numpy()
                d_prob = torch.softmax(out_device,dim=1).cpu().numpy()

                type_correct   += (t_pred == t_lab).sum()
                brand_correct  += (b_pred == b_lab).sum()
                device_correct += (d_pred == d_lab).sum()

                type_preds.extend(t_pred);   type_labels.extend(t_lab)
                brand_preds.extend(b_pred);  brand_labels.extend(b_lab)
                device_preds.extend(d_pred); device_labels.extend(d_lab)

                base_idx = total_samples - len(t_pred)
                for i in range(len(t_pred)):
                    results.append({
                        'index': base_idx + i,
                        'true_type':   idx2type.get(t_lab[i], "unknown"),
                        'pred_type':   idx2type[t_pred[i]],
                        'type_prob':   float(t_prob[i].max()),
                        'true_brand':  idx2brand.get(b_lab[i], "unknown"),
                        'pred_brand':  idx2brand[b_pred[i]],
                        'brand_prob':  float(b_prob[i].max()),
                        'true_device': idx2device.get(d_lab[i], "unknown"),
                        'pred_device': idx2device[d_pred[i]],
                        'device_prob': float(d_prob[i].max()),
                        'is_behavior': dataset.df.iloc[base_idx + i]['is_behavior']
                    })
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_batches += 1
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
        'invalid_batches': invalid_batches
    }
    logger.info(f"[Known] N={total_samples}  Loss={avg_loss:.4f}  "
                f"Type A/F1={type_acc:.4f}/{type_f1:.4f}  "
                f"Brand A/F1={brand_acc:.4f}/{brand_f1:.4f}  "
                f"Device A/F1={device_acc:.4f}/{device_f1:.4f}")
    return metrics, results


def evaluate_unknown_dataset(dataset, dataloader, model, device, dataset_name,
                             conf_type_brand, conf_device, idx2type, idx2brand, idx2device, logger):
    """
    conf_type_brand: 类型/品牌的固定置信度阈值（默认 0.5）
    conf_device:     设备型号的未知判别阈值（这里传入的是“标定后的最优阈值”）
    """
    model.eval()
    total_samples = idle_samples = behavior_samples = 0
    type_correct = brand_correct = device_unknown = 0
    idle_type_correct = idle_brand_correct = 0
    behavior_type_correct = behavior_brand_correct = 0
    results = []
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    invalid_batches = invalid_label_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                if batch is None:
                    invalid_batches += 1
                    continue
                x, y_type, y_brand, y_device = batch
                x = x.to(device); y_type = y_type.to(device); y_brand = y_brand.to(device)
                out_type, out_brand, out_device = model(x)

                t_prob = torch.softmax(out_type,  dim=1).cpu().numpy()
                b_prob = torch.softmax(out_brand, dim=1).cpu().numpy()
                d_prob = torch.softmax(out_device,dim=1).cpu().numpy()

                t_pred = out_type.argmax(1).cpu().numpy()
                b_pred = out_brand.argmax(1).cpu().numpy()
                t_lab  = y_type.argmax(1).cpu().numpy()
                b_lab  = y_brand.argmax(1).cpu().numpy()

                base_idx = total_samples
                total_samples += len(t_pred)
                isb = dataset.df.iloc[base_idx: total_samples]['is_behavior'].values.astype(int)
                t_raw = dataset.df.iloc[base_idx: total_samples]['type_label'].values
                b_raw = dataset.df.iloc[base_idx: total_samples]['brand_label'].values
                d_raw = dataset.df.iloc[base_idx: total_samples]['device_label'].values

                for i in range(len(t_pred)):
                    mt = float(t_prob[i].max())
                    mb = float(b_prob[i].max())
                    md = float(d_prob[i].max())

                    # 基于阈值的标签输出
                    pred_type_label  = idx2type[t_pred[i]]   if mt >= conf_type_brand else "unknown"
                    pred_brand_label = idx2brand[b_pred[i]]  if mb >= conf_type_brand else "unknown"
                    pred_device_label= idx2device[int(np.argmax(d_prob[i]))] if md >= conf_device else "unknown"

                    # 把原始字符串标签映射为 idx（未知设为 -1）
                    ti = -1 if t_raw[i] not in dataset.type2idx   else dataset.type2idx[t_raw[i]]
                    bi = -1 if b_raw[i] not in dataset.brand2idx  else dataset.brand2idx[b_raw[i]]
                    di = -1 if d_raw[i] not in dataset.device2idx else dataset.device2idx[d_raw[i]]
                    if ti == -1 or bi == -1 or di == -1:
                        invalid_label_samples += 1

                    # 存储用于 f1 的索引（只在有效标签且置信度达标时记录预测）
                    type_labels.append(ti); brand_labels.append(bi)
                    type_preds.append(t_pred[i]   if (mt >= conf_type_brand and ti != -1) else -1)
                    brand_preds.append(b_pred[i]  if (mb >= conf_type_brand and bi != -1) else -1)

                    # 正确率（仅对有效标签 + 置信度达标）
                    if (mt >= conf_type_brand and ti != -1 and t_pred[i] == ti):
                        type_correct += 1
                        if isb[i] == 0: idle_type_correct += 1
                        else:           behavior_type_correct += 1
                    if (mb >= conf_type_brand and bi != -1 and b_pred[i] == bi):
                        brand_correct += 1
                        if isb[i] == 0: idle_brand_correct += 1
                        else:           behavior_brand_correct += 1

                    # 型号未知判别
                    if md < conf_device:
                        device_unknown += 1

                    results.append({
                        'index': base_idx + i,
                        'true_type':   idx2type.get(ti, "unknown"),
                        'pred_type':   pred_type_label,
                        'type_prob':   mt,
                        'true_brand':  idx2brand.get(bi, "unknown"),
                        'pred_brand':  pred_brand_label,
                        'brand_prob':  mb,
                        'pred_device': pred_device_label,
                        'device_prob': md,
                        'is_behavior': int(isb[i]),
                    })
                    if isb[i] == 0: idle_samples += 1
                    else:           behavior_samples += 1

                logger.info(f"Batch {batch_idx}: cumN={total_samples}, idle={idle_samples}, beh={behavior_samples}, invalidLabel={invalid_label_samples}")

            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_batches += 1
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
    type_f1  = f1_score(t_lab, t_pre, average='macro',
                        labels=list(range(len(idx2type))),  zero_division=0) if t_lab else 0.0
    brand_f1 = f1_score(b_lab, b_pre, average='macro',
                        labels=list(range(len(idx2brand))), zero_division=0) if b_lab else 0.0

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
        'invalid_batches': invalid_batches,
        'invalid_label_samples': invalid_label_samples
    }
    logger.info(f"[Unknown] N={total_samples}  "
                f"Type A/F1={type_acc:.4f}/{type_f1:.4f}  "
                f"Brand A/F1={brand_acc:.4f}/{brand_f1:.4f}  "
                f"UnkRate(device)={unk_rate:.4f} | idle A: T/B={idle_type_acc:.4f}/{idle_brand_acc:.4f} | "
                f"beh A: T/B={beh_type_acc:.4f}/{beh_brand_acc:.4f}")
    return metrics, results

# ============================ 开放集阈值标定（device） ============================
def collect_device_confidences(dataloader, model, device):
    """
    前向一遍，不做阈值判断，只收集每个样本的 device 头 max softmax 概率。
    返回：list[float]（与 dataloader 顺序一致）
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            x, y_type, y_brand, y_device = batch
            x = x.to(device)
            _, _, out_device = model(x)
            dev_p = torch.softmax(out_device, dim=1).max(dim=1).values  # (B,)
            probs.extend(dev_p.detach().cpu().tolist())
    return probs


def tune_device_threshold(known_probs, unknown_probs, grid=None):
    """
    在“测试集(known)与未知集(unknown)”的 device 头最大软概率分布上做阈值扫描：
      TPR_unknown = P_unknown(prob < τ)
      FPR_known   = P_known (prob < τ)
      目标：最大化 Youden J = TPR_unknown − FPR_known
    返回：best_tau(float), curve_df(pd.DataFrame)
    """
    import numpy as np
    import pandas as pd
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    known = np.asarray(known_probs, dtype=np.float32)
    unk   = np.asarray(unknown_probs, dtype=np.float32)
    rows = []
    best_j = -1.0
    best_tau = 0.3  # fallback

    for tau in grid:
        tpr_unknown = float((unk   < tau).mean()) if unk.size>0 else 0.0
        fpr_known   = float((known < tau).mean()) if known.size>0 else 0.0
        j = tpr_unknown - fpr_known
        rows.append({"tau": float(tau), "TPR_unknown": tpr_unknown, "FPR_known": fpr_known, "YoudenJ": j})
        if j > best_j:
            best_j = j
            best_tau = float(tau)
    return best_tau, pd.DataFrame(rows)

# ============================ 扫描模型 & 读取配置 ============================
def list_models(model_dir: str, eval_only=None):
    """
    返回 [(feat_key, seed, pt_path, cfg_json_or_None)]，仅保留 eval_only 列表中的组。
    模型文件名形如：optimized_multitask_model__F12__seed0.pt
    """
    items = []
    p = Path(model_dir)
    for pt in sorted(p.glob("optimized_multitask_model__F*__seed*.pt")):
        m = re.search(r"optimized_multitask_model__(F\d{1,2})__seed(\d+)\.pt$", pt.name)
        if not m:
            continue
        feat_key, seed = m.group(1), int(m.group(2))
        if eval_only and feat_key not in eval_only:
            continue
        cfg = pt.with_suffix(".json")
        items.append((feat_key, seed, str(pt), str(cfg if cfg.exists() else "")))
    return items


def load_feat_combo(feat_key: str, cfg_path: str):
    """优先读 cfg feat_combo；没有就按 feat_key 用 ABLATION_MAP 推断。"""
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = json.loads(Path(cfg_path).read_text())
            if "feat_combo" in cfg:
                return cfg["feat_combo"]
        except Exception:
            pass
    return ABLATION_MAP.get(feat_key, "Stat")

# ============================ 主流程 ============================
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

    # 载入 CSV & 可选采样（建议 TEST_SAMPLE_FRAC=1.0，确保阈值标定稳定）
    test_df   = pd.read_csv(TEST_CSV)
    unknown_df= pd.read_csv(UNKNOWN_CSV)
    print(f"[INFO] Test is_behavior dist: {test_df['is_behavior'].value_counts().to_dict()}")
    print(f"[INFO] Unknown is_behavior dist: {unknown_df['is_behavior'].value_counts().to_dict()}")

    if TEST_SAMPLE_FRAC < 1.0:
        test_df = (test_df.groupby('type_label', group_keys=True)
                   .apply(lambda x: x.sample(frac=TEST_SAMPLE_FRAC, random_state=42))
                   .reset_index(drop=True))
        print(f"[INFO] Test sampled: {len(test_df)} rows")

    # 仅扫描 F12（或 EVAL_ONLY 中指定的组合）
    items = list_models(MODEL_DIR, eval_only=EVAL_ONLY)
    if not items:
        print(f"[ERROR] No model .pt found for {EVAL_ONLY} in: {MODEL_DIR}")
        sys.exit(1)
    print(f"[MAIN] Found {len(items)} model(s) to evaluate: {EVAL_ONLY}")

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

        # 落地评估用的 CSV（便于复现）
        test_csv_out    = save_dir / "test_eval_input.csv"
        unknown_csv_out = save_dir / "unknown_eval_input.csv"
        test_df.to_csv(test_csv_out, index=False)
        unknown_df.to_csv(unknown_csv_out, index=False)

        # Dataset / DataLoader（与训练一致）
        test_ds    = MultiModalIoTDataset(csv_path=str(test_csv_out),    label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        unknown_ds = MultiModalIoTDataset(csv_path=str(unknown_csv_out), label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        test_loader    = DataLoader(test_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        unknown_loader = DataLoader(unknown_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # 模型定义 & 加载
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

        # ======== 3.4 开放集：先做“device 阈值标定” ========
        logger.info("Calibrating device threshold (open-set) ...")
        known_dev_probs   = collect_device_confidences(test_loader,    model, DEVICE)
        unknown_dev_probs = collect_device_confidences(unknown_loader, model, DEVICE)

        best_tau, curve_df = tune_device_threshold(known_dev_probs, unknown_dev_probs)
        logger.info(f"[Calib] Best device threshold τ* = {best_tau:.3f} "
                    f"(YoudenJ maximized on known-vs-unknown confidence distributions)")
        curve_csv = save_dir / "device_threshold_sweep.csv"
        curve_df.to_csv(curve_csv, index=False)
        logger.info(f"[Calib] Saved sweep curve: {curve_csv}")

        # ======== 已知集评估 ========
        known_metrics, known_rows = evaluate_known_dataset(
            test_ds, test_loader, model, loss_fn, DEVICE, "Test Set",
            idx2type, idx2brand, idx2device, logger
        )
        if known_metrics is None:
            logger.error("Known set evaluation returned no result, skip this model.")
            continue
        pd.DataFrame(known_rows).to_csv(save_dir / "test_results.csv", index=False)

        # ======== 未知集评估（使用标定后的 best_tau） ========
        unknown_metrics, unknown_rows = evaluate_unknown_dataset(
            unknown_ds, unknown_loader, model, DEVICE, "Unknown Set",
            CONF_TYPE_BRAND, best_tau, idx2type, idx2brand, idx2device, logger
        )
        if unknown_metrics is None:
            logger.error("Unknown set evaluation returned no result, skip this model.")
            continue

        # 把阈值写进 unknown 指标
        unknown_metrics["best_device_threshold"] = float(best_tau)
        pd.DataFrame(unknown_rows).to_csv(save_dir / "unknown_results.csv", index=False)

        # 保存本模型的指标汇总
        metrics_df = pd.DataFrame([known_metrics, unknown_metrics])
        metrics_df.to_csv(save_dir / "metrics_summary.csv", index=False)
        logger.info(f"Saved per-model metrics to: {save_dir/'metrics_summary.csv'}")

        # 汇总到总表
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
