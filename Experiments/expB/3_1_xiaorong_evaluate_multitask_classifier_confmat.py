# xiaorong_evaluate_multitask_classifier_confmat.py
# 📆 多任务 IoT 设备识别评估脚本（批量评估 F0–F12） （多了个混淆矩阵的输出）
#
# 功能概述：
# 1) 自动扫描模型目录 12_expB_outputs/uk/models 下的 *.pt（支持 F0–F12、多 seed）
# 2) 对每个模型，依据同名 .json 的 feat_combo（或从文件名推断）构建 Dataset：
#    - 使用 xiaorong_optimized_multimodal_dataset.MultiModalIoTDataset
#    - 确保“对比/自监督路径选择 + 128→64 对齐 + 287 维输入”与训练一致
# 3) 分别评估 测试集(Test) 与 未知集(Unknown)：
#    - Test：保存 metrics_summary.csv 与 test_results.csv，并绘制：
#      * type/brand/device 的 混淆矩阵、PR 曲线、ROC 曲线
#    - Unknown：保存 metrics_summary.csv（追加）与 unknown_results.csv
# 4) 输出目录结构：
#    12_expB_outputs/uk/eval/Fxx_seedY/
#       ├─ test_results.csv
#       ├─ unknown_results.csv
#       ├─ metrics_summary.csv
#       └─ plots/
#           ├─ type_confusion_matrix.png / type_PR.png / type_ROC.png
#           ├─ brand_confusion_matrix.png / brand_PR.png / brand_ROC.png
#           └─ device_confusion_matrix.png / device_PR.png / device_ROC.png
#
# 备注：
# - 混淆矩阵/PR/ROC 仅针对 Test（未知集标签含未知/无效类别时，混淆矩阵意义不大）
# - PR/ROC 采用多分类 macro 方式（逐类绘制并可阅读 AUC）
# - 若需要评估其他数据集(us/cicIoT2022)，只要改参数区路径即可

import os
import re
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 避免无显示环境出图报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 依赖你的 Dataset（与训练一致） =====
from xiaorong_optimized_multimodal_dataset import MultiModalIoTDataset

# --------------------------- 参数区 ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # 可改
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT   = "/home/hyj/unknownDeviceIdentification/dataset"
# 使用 uk 数据集（如需切换 us/cicIoT2022，替换下列三行路径）
TEST_CSV    = f"{DATA_ROOT}/11_multitask_training/uk/3_uk_test.csv"
UNKNOWN_CSV = f"{DATA_ROOT}/11_multitask_training/uk/3_uk_unknown.csv"
LABEL_DIR   = f"{DATA_ROOT}/11_multitask_training/uk"

# 模型所在目录（训练脚本输出）
MODEL_DIR   = f"{DATA_ROOT}/12_expB_outputs/uk/models"
# 评估输出根目录
EVAL_ROOT   = f"{DATA_ROOT}/12_expB_outputs/uk/eval_confmat"

BATCH_SIZE = 64
TEST_SAMPLE_FRAC = 0.35   # 如需仅抽样测试集，改成 0.35 等
CONF_THRESHOLD_TYPE_BRAND = 0.5
CONF_THRESHOLD_DEVICE     = 0.3

# 日志
LOG_DIR = f"{EVAL_ROOT}/_logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "evaluate_log.txt")),
    ],
)
logger = logging.getLogger("eval")

# --------------------------- 模型定义（与训练保持一致） ---------------------------
class MultiTaskClassifier(nn.Module):
    """
    输入：x ∈ R^{B,287} = [31(stat), 128(idle_embed), 128(behavior_embed)]
    门控：g = σ(Linear(is_behavior))，按 is_behavior 融合 idle/behavior 嵌入
    编码：FC(159→256) → TransformerEncoder(2层, 4头)
    输出：三头线性层，分别预测 type/brand/device
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_type: int, num_brand: int, num_device: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(1,1), nn.Sigmoid())
        self.fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_type   = nn.Linear(hidden_dim, num_type)
        self.cls_brand  = nn.Linear(hidden_dim, num_brand)
        self.cls_device = nn.Linear(hidden_dim, num_device)

    def forward(self, x: torch.Tensor):
        stats = x[:, :31]           # (B,31)
        tb = x[:, 31:]              # (B,256)
        idle_embed = tb[:, :128]    # (B,128)
        beh_embed  = tb[:, 128:]    # (B,128)
        is_behavior = stats[:, 30:31]  # (B,1)
        gate = self.gate(is_behavior)  # (B,1)
        weighted = gate*beh_embed + (1-gate)*idle_embed
        fused = torch.cat([stats, weighted], dim=1).unsqueeze(1)  # (B,1,159)
        h = self.fc(fused)           # (B,1,256)
        h = self.encoder(h).squeeze(1)
        return self.cls_type(h), self.cls_brand(h), self.cls_device(h)

# --------------------------- 工具函数 ---------------------------
def list_models(model_dir: str):
    """
    扫描模型目录，返回 [(pt_path, json_path or None, feat_key, seed), ...]
    文件命名约定（训练脚本已使用）：
      optimized_multitask_model__F10__seed0.pt
      optimized_multitask_model__F10__seed0.json  (可选，用于保存 feat_combo 等)
    """
    model_dir = Path(model_dir)
    items = []
    for pt in model_dir.glob("optimized_multitask_model__F*__seed*.pt"):
        m = re.match(r".*__(F\d{1,2})__seed(\d+)\.pt$", pt.name)
        if not m:
            continue
        feat_key, seed = m.group(1), int(m.group(2))
        js = pt.with_suffix(".json")
        items.append((str(pt), str(js) if js.exists() else None, feat_key, seed))
    return sorted(items)

def load_feat_combo(json_path: str, feat_key: str):
    """
    从同名 .json 读取 feat_combo；
    若缺失，则按 feat_key 推断（与训练脚本的 ABLATION_MAP 一致）
    """
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
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "feat_combo" in cfg:
                return cfg["feat_combo"]
        except Exception as e:
            logger.warning(f"read json failed: {json_path} -> {e}")
    return ABLATION_MAP.get(feat_key, "Stat")

def safe_collate(batch):
    """过滤掉 None 批次元素（极少情况文件缺失/读取失败）"""
    from torch.utils.data._utils.collate import default_collate
    batch = [b for b in batch if b is not None and all(x is not None for x in b)]
    if not batch:
        return None
    return default_collate(batch)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# --------------------------- 评估（Test） ---------------------------
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

    # 收集概率用于 PR/ROC
    type_probs_all = []
    brand_probs_all = []
    device_probs_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch
                if x is None:
                    invalid_samples += BATCH_SIZE
                    continue
                x = x.to(device)
                y_type = y_type.to(device)
                y_brand = y_brand.to(device)
                y_device = y_device.to(device)

                pred_type, pred_brand, pred_device = model(x)
                loss = (loss_fn(pred_type, y_type.argmax(1)) +
                        loss_fn(pred_brand, y_brand.argmax(1)) +
                        loss_fn(pred_device, y_device.argmax(1)))

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

                type_pred = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()
                device_pred = pred_device.argmax(1).cpu().numpy()
                type_label = y_type.argmax(1).cpu().numpy()
                brand_label = y_brand.argmax(1).cpu().numpy()
                device_label = y_device.argmax(1).cpu().numpy()

                # 概率
                type_probs = torch.softmax(pred_type, dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs = torch.softmax(pred_device, dim=1).cpu().numpy()
                type_probs_all.append(type_probs)
                brand_probs_all.append(brand_probs)
                device_probs_all.append(device_probs)

                type_correct += (type_pred == type_label).sum()
                brand_correct += (brand_pred == brand_label).sum()
                device_correct += (device_pred == device_label).sum()

                type_preds.extend(type_pred)
                type_labels.extend(type_label)
                brand_preds.extend(brand_pred)
                brand_labels.extend(brand_label)
                device_preds.extend(device_pred)
                device_labels.extend(device_label)

                # 保存每样本（便于回溯/误差分析）
                base_idx = total_samples - len(type_pred)
                for i in range(len(type_pred)):
                    results.append({
                        "index": base_idx + i,
                        "true_type": idx2type[type_label[i]],
                        "pred_type": idx2type[type_pred[i]],
                        "type_prob": float(type_probs[i].max()),
                        "true_brand": idx2brand[brand_label[i]],
                        "pred_brand": idx2brand[brand_pred[i]],
                        "brand_prob": float(brand_probs[i].max()),
                        "true_device": idx2device[device_label[i]],
                        "pred_device": idx2device[device_pred[i]],
                        "device_prob": float(device_probs[i].max()),
                        "is_behavior": int(dataset.df.iloc[base_idx + i]["is_behavior"]),
                    })
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_samples += BATCH_SIZE
                continue

    if total_samples == 0:
        logger.error("No valid samples processed in test set evaluation")
        return None, None, None

    avg_loss = total_loss / total_samples
    type_acc = type_correct / total_samples
    brand_acc = brand_correct / total_samples
    device_acc = device_correct / total_samples
    type_f1 = f1_score(type_labels, type_preds, average="macro", zero_division=0)
    brand_f1 = f1_score(brand_labels, brand_preds, average="macro", zero_division=0)
    device_f1 = f1_score(device_labels, device_preds, average="macro", zero_division=0)

    metrics = {
        "dataset": dataset_name,
        "loss": avg_loss,
        "type_acc": type_acc, "type_f1": type_f1,
        "brand_acc": brand_acc, "brand_f1": brand_f1,
        "device_acc": device_acc, "device_f1": device_f1,
        "total_samples": total_samples,
        "invalid_samples": invalid_samples
    }
    logger.info(f"[Test] samples={total_samples}, invalid={invalid_samples}")

    raw = {
        "type":   {"y_true": np.array(type_labels),   "y_pred": np.array(type_preds),   "y_prob": np.vstack(type_probs_all)},
        "brand":  {"y_true": np.array(brand_labels),  "y_pred": np.array(brand_preds),  "y_prob": np.vstack(brand_probs_all)},
        "device": {"y_true": np.array(device_labels), "y_pred": np.array(device_preds), "y_prob": np.vstack(device_probs_all)},
    }
    return metrics, results, raw

# --------------------------- 评估（Unknown） ---------------------------
def evaluate_unknown_dataset(dataset, dataloader, model, device, dataset_name,
                             conf_type_brand, conf_device, idx2type, idx2brand, idx2device, logger):
    model.eval()
    total_samples = 0
    idle_samples = behavior_samples = 0
    type_correct = brand_correct = 0
    device_unknown = 0
    idle_type_correct = idle_brand_correct = 0
    behavior_type_correct = behavior_brand_correct = 0
    results = []
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    invalid_samples = 0
    invalid_label_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            try:
                x, y_type, y_brand, y_device = batch
                if x is None:
                    invalid_samples += BATCH_SIZE
                    continue
                x = x.to(device)
                y_type = y_type.to(device)
                y_brand = y_brand.to(device)

                pred_type, pred_brand, pred_device = model(x)

                type_probs = torch.softmax(pred_type, dim=1).cpu().numpy()
                brand_probs = torch.softmax(pred_brand, dim=1).cpu().numpy()
                device_probs = torch.softmax(pred_device, dim=1).cpu().numpy()

                type_pred = pred_type.argmax(1).cpu().numpy()
                brand_pred = pred_brand.argmax(1).cpu().numpy()

                batch_indices = range(total_samples, total_samples + len(type_pred))
                is_behavior = dataset.df.iloc[batch_indices]["is_behavior"].values.astype(int)
                type_labels_raw = dataset.df.iloc[batch_indices]["type_label"].values
                brand_labels_raw = dataset.df.iloc[batch_indices]["brand_label"].values
                device_labels_raw = dataset.df.iloc[batch_indices]["device_label"].values

                for i in range(len(type_pred)):
                    mtp = type_probs[i].max()
                    mbp = brand_probs[i].max()
                    mdp = device_probs[i].max()

                    pred_type_label = idx2type[type_pred[i]] if mtp >= conf_type_brand else "unknown"
                    pred_brand_label = idx2brand[brand_pred[i]] if mbp >= conf_type_brand else "unknown"
                    pred_device_label = idx2device[pred_device.argmax(1)[i].item()] if mdp >= conf_device else "unknown"

                    ti = -1 if type_labels_raw[i] not in dataset.type2idx else dataset.type2idx[type_labels_raw[i]]
                    bi = -1 if brand_labels_raw[i] not in dataset.brand2idx else dataset.brand2idx[brand_labels_raw[i]]
                    di = -1 if device_labels_raw[i] not in dataset.device2idx else dataset.device2idx[device_labels_raw[i]]

                    if ti == -1 or bi == -1 or di == -1:
                        invalid_label_samples += 1

                    type_preds.append(type_pred[i] if mtp >= conf_type_brand and ti != -1 else -1)
                    brand_preds.append(brand_pred[i] if mbp >= conf_type_brand and bi != -1 else -1)
                    type_labels.append(ti)
                    brand_labels.append(bi)

                    if mtp >= conf_type_brand and type_pred[i] == ti and ti != -1:
                        type_correct += 1
                        if is_behavior[i] == 0: idle_type_correct += 1
                        else: behavior_type_correct += 1
                    if mbp >= conf_type_brand and brand_pred[i] == bi and bi != -1:
                        brand_correct += 1
                        if is_behavior[i] == 0: idle_brand_correct += 1
                        else: behavior_brand_correct += 1
                    if mdp < conf_device:
                        device_unknown += 1

                    results.append({
                        "index": total_samples + i,
                        "true_type": idx2type.get(ti, "unknown"),
                        "pred_type": pred_type_label,
                        "type_prob": float(mtp),
                        "true_brand": idx2brand.get(bi, "unknown"),
                        "pred_brand": pred_brand_label,
                        "brand_prob": float(mbp),
                        "pred_device": pred_device_label,
                        "device_prob": float(mdp),
                        "is_behavior": int(is_behavior[i]),
                    })

                    if is_behavior[i] == 0: idle_samples += 1
                    else: behavior_samples += 1

                total_samples += len(type_pred)
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                invalid_samples += BATCH_SIZE
                continue

    if total_samples == 0:
        logger.error("No valid samples processed in unknown set evaluation")
        return None, None

    type_acc = type_correct / total_samples
    brand_acc = brand_correct / total_samples
    device_unknown_rate = device_unknown / total_samples
    idle_type_acc = idle_type_correct / idle_samples if idle_samples > 0 else 0.0
    idle_brand_acc = idle_brand_correct / idle_samples if idle_samples > 0 else 0.0
    behavior_type_acc = behavior_type_correct / behavior_samples if behavior_samples > 0 else 0.0
    behavior_brand_acc = behavior_brand_correct / behavior_samples if behavior_samples > 0 else 0.0

    type_f1 = f1_score([t for t in type_labels if t != -1],
                       [p for p, t in zip(type_preds, type_labels) if t != -1],
                       average="macro",
                       labels=list(range(len(idx2type))),
                       zero_division=0)
    brand_f1 = f1_score([b for b in brand_labels if b != -1],
                        [p for p, b in zip(brand_preds, brand_labels) if b != -1],
                        average="macro",
                        labels=list(range(len(idx2brand))),
                        zero_division=0)

    metrics = {
        "dataset": dataset_name,
        "type_acc": type_acc, "type_f1": type_f1,
        "brand_acc": brand_acc, "brand_f1": brand_f1,
        "device_unknown_rate": device_unknown_rate,
        "idle_type_acc": idle_type_acc, "idle_brand_acc": idle_brand_acc,
        "behavior_type_acc": behavior_type_acc, "behavior_brand_acc": behavior_brand_acc,
        "total_samples": total_samples,
        "idle_samples": idle_samples, "behavior_samples": behavior_samples,
        "invalid_samples": invalid_samples,
        "invalid_label_samples": invalid_label_samples
    }
    logger.info(f"[Unknown] samples={total_samples}, invalid={invalid_samples}, invalid_labels={invalid_label_samples}")
    return metrics, results

# --------------------------- 绘图（Test 专用） ---------------------------
def plot_confusion_and_curves(save_dir, task_name, y_true, y_pred, y_prob, classes_dict, logger):
    """
    绘制并保存 混淆矩阵 / PR / ROC（macro，逐类折线）
    - classes_dict: {idx: name}
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    class_indices = list(classes_dict.keys())
    class_names = [classes_dict[i] for i in class_indices]
    n_classes = len(class_indices)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=90)
    ax.set_title(f"{task_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_dir / f"{task_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)
    logger.info(f"[PLOT] {task_name} confusion matrix saved.")

    # PR 曲线（逐类）
    y_true_bin = label_binarize(y_true, classes=class_indices)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, lw=1, alpha=0.7, label=f"{class_names[i]}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{task_name} Precision-Recall Curves")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(save_dir / f"{task_name}_PR.png", dpi=200)
    plt.close(fig)
    logger.info(f"[PLOT] {task_name} PR curves saved.")

    # ROC 曲线（逐类）
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, alpha=0.7, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{task_name} ROC Curves")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(save_dir / f"{task_name}_ROC.png", dpi=200)
    plt.close(fig)
    logger.info(f"[PLOT] {task_name} ROC curves saved.")

# --------------------------- 主流程 ---------------------------
def main():
    logger.info(f"Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    for p in [TEST_CSV, UNKNOWN_CSV, LABEL_DIR, MODEL_DIR]:
        if not os.path.exists(p):
            logger.error(f"Path not found: {p}")
            sys.exit(1)

    # 标签 dict
    with open(Path(LABEL_DIR)/"type2idx.json","r") as f:
        type2idx = json.load(f)
    with open(Path(LABEL_DIR)/"brand2idx.json","r") as f:
        brand2idx = json.load(f)
    with open(Path(LABEL_DIR)/"device2idx.json","r") as f:
        device2idx = json.load(f)
    idx2type = {v:k for k,v in type2idx.items()}
    idx2brand= {v:k for k,v in brand2idx.items()}
    idx2device={v:k for k,v in device2idx.items()}
    logger.info(f"Labels: type={len(type2idx)}, brand={len(brand2idx)}, device={len(device2idx)}")

    # 载入 Test/Unknown CSV
    test_df = pd.read_csv(TEST_CSV)
    unknown_df = pd.read_csv(UNKNOWN_CSV)
    logger.info(f"Test is_behavior: {test_df['is_behavior'].value_counts().to_dict()}")
    logger.info(f"Unknown is_behavior: {unknown_df['is_behavior'].value_counts().to_dict()}")

    # 采样 Test（如不抽样，保持 1.0）
    if TEST_SAMPLE_FRAC < 1.0:
        test_df = (test_df.groupby('type_label', group_keys=True)
                   .apply(lambda x: x.sample(frac=TEST_SAMPLE_FRAC, random_state=42))
                   .reset_index(drop=True))

    # 为评估输出准备目录
    ensure_dir(EVAL_ROOT)

    # 批量评估每个模型
    models = list_models(MODEL_DIR)
    if not models:
        logger.error(f"No models found in {MODEL_DIR}")
        sys.exit(1)

    for pt_path, json_path, feat_key, seed in models:
        feat_combo = load_feat_combo(json_path, feat_key)
        save_dir   = Path(EVAL_ROOT) / f"{feat_key}_seed{seed}"
        plots_dir  = save_dir / "plots"
        ensure_dir(save_dir)
        ensure_dir(plots_dir)

        logger.info("="*80)
        logger.info(f"[EVAL] model={pt_path} | feat_key={feat_key} | seed={seed} | feat_combo={feat_combo}")
        logger.info("="*80)

        # 构建 Dataset / DataLoader（传入 feat_combo，确保路径/维度处理与训练一致）
        test_tmp = save_dir / "sampled_test.csv"
        test_df.to_csv(test_tmp, index=False)
        unknown_tmp = save_dir / "processed_unknown.csv"
        unknown_df.to_csv(unknown_tmp, index=False)

        test_ds = MultiModalIoTDataset(str(test_tmp), label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        unk_ds  = MultiModalIoTDataset(str(unknown_tmp), label_dict_dir=LABEL_DIR, feat_combo=feat_combo)

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate)
        unk_loader  = DataLoader(unk_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate)

        # 初始化模型并加载权重
        model = MultiTaskClassifier(
            input_dim=159, hidden_dim=256,
            num_type=len(test_ds.type2idx),
            num_brand=len(test_ds.brand2idx),
            num_device=len(test_ds.device2idx)
        ).to(DEVICE)
        try:
            state = torch.load(pt_path, map_location=DEVICE)
            model.load_state_dict(state)
        except Exception as e:
            logger.error(f"load model failed: {pt_path} -> {e}")
            continue

        loss_fn = nn.CrossEntropyLoss()

        # ---------- Test ----------
        test_metrics, test_rows, test_raw = evaluate_known_dataset(
            test_ds, test_loader, model, loss_fn, DEVICE, "Test Set",
            idx2type, idx2brand, idx2device, logger
        )
        if test_metrics is None:
            logger.error("[Test] no valid result, skip this model.")
            continue
        pd.DataFrame(test_rows).to_csv(save_dir/"test_results.csv", index=False)
        logger.info(f"[SAVE] test_results.csv @ {save_dir}")

        # 绘图（Test）
        plot_confusion_and_curves(
            plots_dir, "type",
            test_raw["type"]["y_true"], test_raw["type"]["y_pred"], test_raw["type"]["y_prob"],
            idx2type, logger
        )
        plot_confusion_and_curves(
            plots_dir, "brand",
            test_raw["brand"]["y_true"], test_raw["brand"]["y_pred"], test_raw["brand"]["y_prob"],
            idx2brand, logger
        )
        plot_confusion_and_curves(
            plots_dir, "device",
            test_raw["device"]["y_true"], test_raw["device"]["y_pred"], test_raw["device"]["y_prob"],
            idx2device, logger
        )

        # ---------- Unknown ----------
        unk_metrics, unk_rows = evaluate_unknown_dataset(
            unk_ds, unk_loader, model, DEVICE, "Unknown Set",
            CONF_THRESHOLD_TYPE_BRAND, CONF_THRESHOLD_DEVICE,
            idx2type, idx2brand, idx2device, logger
        )
        if unk_metrics is not None:
            pd.DataFrame(unk_rows).to_csv(save_dir/"unknown_results.csv", index=False)
            logger.info(f"[SAVE] unknown_results.csv @ {save_dir}")

        # 汇总指标（Test + Unknown）
        ms = []
        ms.append({**{"feat_key": feat_key, "seed": seed, "split": "test"}, **test_metrics})
        if unk_metrics is not None:
            ms.append({**{"feat_key": feat_key, "seed": seed, "split": "unknown"}, **unk_metrics})
        pd.DataFrame(ms).to_csv(save_dir/"metrics_summary.csv", index=False)
        logger.info(f"[SAVE] metrics_summary.csv @ {save_dir}")

    logger.info("All models evaluated. Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
