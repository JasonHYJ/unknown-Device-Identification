# 3_2_xiaorong_evaluate_multitask_classifier.py
# 📆 多任务 IoT 设备识别 — 3.2 融合策略批量评估脚本
#
# 功能概述：
# - 扫描 .../12_expB_outputs/<dataset>/3_2_models/ 下的所有 .pt 权重
# - 从文件名解析 feat_key、seed、fusion（gate/idleonly/behonly/concat/attn）、tasks（如有）
# - 依据 fusion 构建“融合感知”的模型结构，加载权重（严格加载失败会回退到非严格并给出告警）
# - 与训练一致地构建 MultiModalIoTDataset（feat_combo 从 .json 或 Fxx 推断）
# - 分别在已知集（Test）与未知集（Unknown）上评估并保存结果
# - 输出每模型一个目录 + 总汇总 CSV
#
# 输入前提：
# - CSV：3_<dataset>_test.csv、3_<dataset>_unknown.csv（列中含绝对路径）
# - 标签字典：type2idx.json / brand2idx.json / device2idx.json
# - 模型与配置：…/12_expB_outputs/<dataset>/3_2_models/optimized_multitask_model__Fxx__[fusion]__seedY[__tasks[...]].pt(.json)
#
# 输出：
# - 每个模型目录：…/12_expB_outputs/<dataset>/3_2_eval/Fxx_[fusion]_seedY[_tasksXXX]/
#   - test_results.csv, unknown_results.csv, metrics_summary.csv, eval_log.txt, 以及 test/unknown 输入快照
# - 总汇总：…/12_expB_outputs/<dataset>/3_2_eval/_all_models_summary.csv

import os
import re
import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# 与训练一致的数据集（会按 feat_combo 自动切换对比/自监督路径，并把 128→64 对齐）
from xiaorong_optimized_multimodal_dataset import MultiModalIoTDataset

# ===================== 可配置区 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"
DATASET   = "uk"    # ← 需要换数据集时改这里：uk / us / cicIoT2022

TEST_CSV    = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_test.csv"
UNKNOWN_CSV = f"{DATA_ROOT}/11_multitask_training/{DATASET}/3_{DATASET}_unknown.csv"
LABEL_DIR   = f"{DATA_ROOT}/11_multitask_training/{DATASET}"

# 3.2 模型统一建议放在单独目录下，避免和 3.1/3.3 混在一起
MODEL_DIR = f"{DATA_ROOT}/12_expB_outputs/{DATASET}/3_2_models"
EVAL_ROOT = f"{DATA_ROOT}/12_expB_outputs/{DATASET}/3_2_eval"
os.makedirs(EVAL_ROOT, exist_ok=True)

BATCH_SIZE        = 64
NUM_WORKERS       = 4
TEST_SAMPLE_FRAC  = 1.0   # 需要下采样可改 <1.0
CONF_TYPE_BRAND   = 0.5   # 未知集：type/brand 置信度阈值
CONF_DEVICE       = 0.3   # 未知集：device 置信度阈值
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feat_key → feat_combo（缺少 .json 时按此推断）
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

# ===================== 日志 =====================
def build_logger(log_file: Path):
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh  = logging.FileHandler(str(log_file)); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

# ===================== 解析文件名（feat_key / fusion / seed / tasks） =====================
def parse_from_filename(pt_path: str):
    """
    兼容以下命名：
      optimized_multitask_model__F12__seed0.pt
      optimized_multitask_model__F12__idleonly__seed0.pt
      optimized_multitask_model__F12__concat__seed0.pt
      optimized_multitask_model__F12__attn__seed0__tasks[0B0].pt
      optimized_multitask_model__F12__seed0__tasks[001].pt
    """
    name = os.path.basename(pt_path)
    m = re.match(
        r"optimized_multitask_model__(F\d{1,2})(?:__([a-zA-Z]+))?__seed(\d+)(?:__tasks\[(.+?)\])?\.pt$",
        name
    )
    if not m:
        return None, None, None, None
    feat_key = m.group(1)
    fusion   = (m.group(2) or "gate").lower()
    seed     = int(m.group(3))
    tasks    = m.group(4) or "ALL"
    return feat_key, fusion, seed, tasks

def list_models(model_dir: str):
    """返回 [(pt_path, cfg_path or "")]，自动过滤不匹配命名的文件"""
    items = []
    p = Path(model_dir)
    for pt in sorted(p.glob("optimized_multitask_model__F*__seed*.pt")):
        if parse_from_filename(str(pt))[0] is None:
            continue
        cfg = pt.with_suffix(".json")
        items.append((str(pt), str(cfg if cfg.exists() else "")))
    return items

def load_feat_combo(feat_key: str, cfg_path: str):
    """优先读 cfg 中的 feat_combo；没有就按 feat_key 推断。"""
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = json.loads(Path(cfg_path).read_text())
            if "feat_combo" in cfg:
                return cfg["feat_combo"]
        except Exception:
            pass
    return ABLATION_MAP.get(feat_key, "Stat")

# ===================== 融合感知模型（与 3.2 训练一致） =====================
class FusionAwareMultiTaskClassifier(nn.Module):
    """
    支持 5 种融合：
      - fusion='gate'     : 门控（使用 is_behavior）
      - fusion='idleonly' : 仅 idle_embed
      - fusion='behonly'  : 仅 behavior_embed
      - fusion='concat'   : 先把  经 concat_reduce(256->128)，再与 stats 拼 159
      - fusion='alpha'    : learnable α 做线性融合：α*beh + (1-α)*idle

    统一让 FC 的输入维度都是 159（31 + 128），和你训练时的权重形状 [256,159] 对齐。
    """
    def __init__(self, num_type, num_brand, num_device, fusion: str = "gate", hidden_dim: int = 256):
        super().__init__()
        self.fusion = fusion.lower()
        self.input_dim = 31 + 128  # 统一 159，和 ckpt 的 fc.weight [256,159] 匹配

        # gate 融合：需要门控层
        if self.fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        else:
            self.gate = None

        # concat 融合：需要把 256 降到 128，名称与 ckpt 对齐（concat_reduce.*）
        if self.fusion == "concat":
            self.concat_reduce = nn.Linear(256, 128)
        else:
            self.concat_reduce = None

        # alpha 融合：可学习标量（名称与 ckpt 对齐：alpha）
        if self.fusion == "alpha":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            # 以防加载严格检查时需要这个属性存在（但不注册为参数）
            self.register_buffer("_alpha_dummy", torch.tensor(0.0), persistent=False)

        # 主干编码：FC + Transformer
        self.fc = nn.Linear(self.input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)

        # heads
        self.cls_type   = nn.Linear(hidden_dim, num_type)
        self.cls_brand  = nn.Linear(hidden_dim, num_brand)
        self.cls_device = nn.Linear(hidden_dim, num_device)

    def forward(self, x):
        stats = x[:, :31]
        both  = x[:, 31:]          # 256
        idle_embed = both[:, :128]
        beh_embed  = both[:, 128:]
        is_behavior = stats[:, 30:31]  # (B,1)

        if self.fusion == "gate":
            g = self.gate(is_behavior)
            fused_embed = g * beh_embed + (1 - g) * idle_embed          # 128
            fused_input = torch.cat([stats, fused_embed], dim=1)        # 159
        elif self.fusion == "idleonly":
            fused_input = torch.cat([stats, idle_embed], dim=1)         # 159
        elif self.fusion == "behonly":
            fused_input = torch.cat([stats, beh_embed], dim=1)          # 159
        elif self.fusion == "concat":
            both256 = torch.cat([idle_embed, beh_embed], dim=1)         # 256
            reduced = self.concat_reduce(both256)                        # 128
            fused_input = torch.cat([stats, reduced], dim=1)            # 159
        elif self.fusion == "alpha":
            # α ∈ R；与 ckpt 的参数名一致（alpha）
            fused_embed = self.alpha * beh_embed + (1.0 - self.alpha) * idle_embed  # 128
            fused_input = torch.cat([stats, fused_embed], dim=1)         # 159
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")

        h = self.fc(fused_input).unsqueeze(1)
        h = self.encoder(h).squeeze(1)
        return self.cls_type(h), self.cls_brand(h), self.cls_device(h)


# ===================== 评估函数（与之前一致，精简版：无混淆矩阵/PR/ROC） =====================
def evaluate_known_dataset(dataset, dataloader, model, loss_fn, device, dataset_name, idx2type, idx2brand, idx2device, logger):
    model.eval()
    total_loss = 0
    type_correct = brand_correct = device_correct = 0
    total_samples = 0
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    device_preds, device_labels = [], []
    rows = []
    invalid_batches = 0

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            if batch is None:
                invalid_batches += 1
                continue
            x, y_type_oh, y_brand_oh, y_device_oh = batch
            x = x.to(device); y_type_oh = y_type_oh.to(device); y_brand_oh = y_brand_oh.to(device); y_device_oh = y_device_oh.to(device)

            out_type, out_brand, out_device = model(x)
            loss = loss_fn(out_type, y_type_oh.argmax(1)) + loss_fn(out_brand, y_brand_oh.argmax(1)) + loss_fn(out_device, y_device_oh.argmax(1))
            total_loss += loss.item() * x.size(0); total_samples += x.size(0)

            tp = out_type.argmax(1).cpu().numpy(); bp = out_brand.argmax(1).cpu().numpy(); dp = out_device.argmax(1).cpu().numpy()
            tl = y_type_oh.argmax(1).cpu().numpy(); bl = y_brand_oh.argmax(1).cpu().numpy(); dl = y_device_oh.argmax(1).cpu().numpy()

            type_correct  += (tp == tl).sum(); brand_correct += (bp == bl).sum(); device_correct += (dp == dl).sum()
            type_preds.extend(tp); brand_preds.extend(bp); device_preds.extend(dp)
            type_labels.extend(tl); brand_labels.extend(bl); device_labels.extend(dl)

            # 保存样本级记录（可选精简）
            base = total_samples - len(tp)
            tprob = torch.softmax(out_type, dim=1).cpu().numpy()
            bprob = torch.softmax(out_brand, dim=1).cpu().numpy()
            dprob = torch.softmax(out_device, dim=1).cpu().numpy()
            for i in range(len(tp)):
                rows.append({
                    "index": base+i,
                    "true_type": idx2type.get(tl[i],"unknown"), "pred_type": idx2type[tp[i]], "type_prob": float(tprob[i].max()),
                    "true_brand": idx2brand.get(bl[i],"unknown"), "pred_brand": idx2brand[bp[i]], "brand_prob": float(bprob[i].max()),
                    "true_device": idx2device.get(dl[i],"unknown"), "pred_device": idx2device[dp[i]], "device_prob": float(dprob[i].max()),
                    "is_behavior": int(dataset.df.iloc[base+i]["is_behavior"])
                })

    if total_samples == 0:
        logger.error("No valid samples in known set.")
        return None, None

    avg_loss = total_loss / total_samples
    type_acc = type_correct / total_samples; brand_acc = brand_correct / total_samples; device_acc = device_correct / total_samples
    type_f1  = f1_score(type_labels, type_preds, average='macro', zero_division=0)
    brand_f1 = f1_score(brand_labels, brand_preds, average='macro', zero_division=0)
    device_f1= f1_score(device_labels, device_preds, average='macro', zero_division=0)
    metrics = {
        "dataset": dataset_name,
        "loss": avg_loss,
        "type_acc": type_acc, "type_f1": type_f1,
        "brand_acc": brand_acc, "brand_f1": brand_f1,
        "device_acc": device_acc, "device_f1": device_f1,
        "total_samples": total_samples,
        "invalid_batches": invalid_batches
    }
    logger.info(f"[Known] N={total_samples}  Loss={avg_loss:.4f}  "
                f"T A/F1={type_acc:.4f}/{type_f1:.4f}  "
                f"B A/F1={brand_acc:.4f}/{brand_f1:.4f}  "
                f"D A/F1={device_acc:.4f}/{device_f1:.4f}")
    return metrics, rows

def evaluate_unknown_dataset(dataset, dataloader, model, device, dataset_name,
                             conf_type_brand, conf_device, idx2type, idx2brand, idx2device, logger):
    model.eval()
    total_samples = idle_samples = beh_samples = 0
    type_correct = brand_correct = device_unknown = 0
    idle_type_correct = idle_brand_correct = 0
    beh_type_correct  = beh_brand_correct  = 0
    type_preds, type_labels = [], []
    brand_preds, brand_labels = [], []
    rows = []
    invalid_batches = invalid_label_samples = 0

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", file=sys.stdout)):
            if batch is None:
                invalid_batches += 1
                continue
            x, y_type_oh, y_brand_oh, y_device_oh = batch
            x = x.to(device); y_type_oh = y_type_oh.to(device); y_brand_oh = y_brand_oh.to(device)

            out_type, out_brand, out_device = model(x)
            tp = out_type.argmax(1).cpu().numpy(); bp = out_brand.argmax(1).cpu().numpy()
            tprob = torch.softmax(out_type, dim=1).cpu().numpy()
            bprob = torch.softmax(out_brand, dim=1).cpu().numpy()
            dprob = torch.softmax(out_device, dim=1).cpu().numpy()

            base = total_samples
            total_samples += len(tp)
            isb = dataset.df.iloc[base:total_samples]["is_behavior"].values.astype(int)
            tl_raw = dataset.df.iloc[base:total_samples]["type_label"].values
            bl_raw = dataset.df.iloc[base:total_samples]["brand_label"].values
            dl_raw = dataset.df.iloc[base:total_samples]["device_label"].values

            for i in range(len(tp)):
                mt, mb, md = tprob[i].max(), bprob[i].max(), dprob[i].max()
                # map raw→idx，未知设 -1
                ti = -1 if tl_raw[i] not in dataset.type2idx else dataset.type2idx[tl_raw[i]]
                bi = -1 if bl_raw[i] not in dataset.brand2idx else dataset.brand2idx[bl_raw[i]]
                di = -1 if dl_raw[i] not in dataset.device2idx else dataset.device2idx[dl_raw[i]]
                if ti==-1 or bi==-1 or di==-1:
                    invalid_label_samples += 1

                # 仅在置信度达标且标签有效时计入 acc / F1
                if mt >= conf_type_brand and ti != -1:
                    type_preds.append(tp[i]); type_labels.append(ti)
                    if tp[i] == ti:
                        type_correct += 1
                        if isb[i]==0: idle_type_correct += 1
                        else:         beh_type_correct  += 1
                else:
                    type_labels.append(ti); type_preds.append(-1)

                if mb >= conf_type_brand and bi != -1:
                    brand_preds.append(bp[i]); brand_labels.append(bi)
                    if bp[i] == bi:
                        brand_correct += 1
                        if isb[i]==0: idle_brand_correct += 1
                        else:         beh_brand_correct  += 1
                else:
                    brand_labels.append(bi); brand_preds.append(-1)

                if md < conf_device:
                    device_unknown += 1

                rows.append({
                    "index": base+i,
                    "true_type": idx2type.get(ti,"unknown"),
                    "pred_type": idx2type[tp[i]] if mt>=conf_type_brand else "unknown",
                    "type_prob": float(mt),
                    "true_brand": idx2brand.get(bi,"unknown"),
                    "pred_brand": idx2brand[bp[i]] if mb>=conf_type_brand else "unknown",
                    "brand_prob": float(mb),
                    "pred_device": idx2device[out_device.argmax(1)[i].item()] if md>=conf_device else "unknown",
                    "device_prob": float(md),
                    "is_behavior": int(isb[i])
                })

                if isb[i]==0: idle_samples += 1
                else:         beh_samples  += 1

            logger.info(f"Batch {bidx}: cumN={total_samples}, idle={idle_samples}, beh={beh_samples}, invalidLabel={invalid_label_samples}")

    if total_samples == 0:
        logger.error("No valid samples in unknown set.")
        return None, None

    type_acc = type_correct/total_samples; brand_acc = brand_correct/total_samples
    unk_rate = device_unknown/total_samples
    idle_t  = idle_type_correct/idle_samples if idle_samples>0 else 0.0
    idle_b  = idle_brand_correct/idle_samples if idle_samples>0 else 0.0
    beh_t   = beh_type_correct/beh_samples   if beh_samples>0 else 0.0
    beh_b   = beh_brand_correct/beh_samples  if beh_samples>0 else 0.0

    t_lab = [t for t in type_labels if t!=-1]; t_pre = [p for p,t in zip(type_preds,type_labels) if t!=-1]
    b_lab = [b for b in brand_labels if b!=-1]; b_pre = [p for p,b in zip(brand_preds,brand_labels) if b!=-1]
    type_f1  = f1_score(t_lab, t_pre, average='macro', zero_division=0) if t_lab else 0.0
    brand_f1 = f1_score(b_lab, b_pre, average='macro', zero_division=0) if b_lab else 0.0

    metrics = {
        "dataset": dataset_name,
        "type_acc": type_acc, "type_f1": type_f1,
        "brand_acc": brand_acc, "brand_f1": brand_f1,
        "device_unknown_rate": unk_rate,
        "idle_type_acc": idle_t, "idle_brand_acc": idle_b,
        "behavior_type_acc": beh_t, "behavior_brand_acc": beh_b,
        "total_samples": total_samples,
        "idle_samples": idle_samples,
        "behavior_samples": beh_samples,
        "invalid_batches": invalid_batches,
        "invalid_label_samples": invalid_label_samples
    }
    logger.info(f"[Unknown] N={total_samples}  "
                f"T A/F1={type_acc:.4f}/{type_f1:.4f}  "
                f"B A/F1={brand_acc:.4f}/{brand_f1:.4f}  "
                f"UnkRate={unk_rate:.4f} | idle T/B={idle_t:.4f}/{idle_b:.4f} | beh T/B={beh_t:.4f}/{beh_b:.4f}")
    return metrics, rows

# ===================== 主流程 =====================
def main():
    # 载入标签字典
    with open(Path(LABEL_DIR)/"type2idx.json") as f:   type2idx = json.load(f)
    with open(Path(LABEL_DIR)/"brand2idx.json") as f:  brand2idx = json.load(f)
    with open(Path(LABEL_DIR)/"device2idx.json") as f: device2idx = json.load(f)
    idx2type  = {v:k for k,v in type2idx.items()}
    idx2brand = {v:k for k,v in brand2idx.items()}
    idx2device= {v:k for k,v in device2idx.items()}

    # 载入 CSV
    test_df    = pd.read_csv(TEST_CSV)
    unknown_df = pd.read_csv(UNKNOWN_CSV)
    print(f"[INFO] Test is_behavior dist: {test_df['is_behavior'].value_counts().to_dict()}")
    print(f"[INFO] Unknown is_behavior dist: {unknown_df['is_behavior'].value_counts().to_dict()}")

    if TEST_SAMPLE_FRAC < 1.0:
        test_df = (test_df.groupby('type_label', group_keys=True)
                   .apply(lambda x: x.sample(frac=TEST_SAMPLE_FRAC, random_state=42))
                   .reset_index(drop=True))
        print(f"[INFO] Test sampled rows: {len(test_df)}")

    # 扫描 3.2 模型
    items = list_models(MODEL_DIR)
    if not items:
        print(f"[ERROR] No model .pt found in: {MODEL_DIR}")
        sys.exit(1)
    print(f"[MAIN] Found {len(items)} models to evaluate in 3.2.")

    summary_rows = []

    for pt_path, cfg_path in items:
        feat_key, fusion, seed, tasks = parse_from_filename(pt_path)
        feat_combo = load_feat_combo(feat_key, cfg_path)

        # 每模型一个输出目录（含 fusion / tasks）
        suffix = f"{feat_key}_{fusion}_seed{seed}" + (f"_tasks[{tasks}]" if tasks!="ALL" else "")
        save_dir = Path(EVAL_ROOT) / suffix
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = build_logger(save_dir/"eval_log.txt")

        logger.info("="*80)
        logger.info(f"Model: {pt_path}")
        logger.info(f"FeatKey={feat_key}, Seed={seed}, FeatCombo={feat_combo}, Fusion={fusion}, Tasks={tasks}")
        logger.info("="*80)

        # 落地输入快照
        (save_dir/"test_eval_input.csv").write_text(test_df.to_csv(index=False))
        (save_dir/"unknown_eval_input.csv").write_text(unknown_df.to_csv(index=False))

        # 构造 Dataset/DataLoader
        test_ds    = MultiModalIoTDataset(csv_path=str(save_dir/"test_eval_input.csv"),    label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        unknown_ds = MultiModalIoTDataset(csv_path=str(save_dir/"unknown_eval_input.csv"), label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
        test_loader    = DataLoader(test_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        unknown_loader = DataLoader(unknown_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # 构造融合感知模型并加载权重
        model = FusionAwareMultiTaskClassifier(
            num_type=len(test_ds.type2idx),
            num_brand=len(test_ds.brand2idx),
            num_device=len(test_ds.device2idx),
            fusion=fusion,
            hidden_dim=256
        ).to(DEVICE)

        try:
            state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
            try:
                model.load_state_dict(state, strict=True)
            except Exception as e_strict:
                logger.warning(f"Strict load failed ({e_strict}); fallback to strict=False.")
                missing = model.load_state_dict(state, strict=False)
                if getattr(missing, "missing_keys", None):
                    logger.warning(f"Missing keys: {missing.missing_keys}")
                if getattr(missing, "unexpected_keys", None):
                    logger.warning(f"Unexpected keys: {missing.unexpected_keys}")
            logger.info("Model weights loaded.")
        except Exception as e:
            logger.error(f"Load state_dict failed: {e}")
            continue

        loss_fn = nn.CrossEntropyLoss()

        # 评估 Test
        known_metrics, known_rows = evaluate_known_dataset(
            test_ds, test_loader, model, loss_fn, DEVICE, "Test Set",
            idx2type, idx2brand, idx2device, logger
        )
        if known_metrics is None:
            logger.error("Known set evaluation failed, skip.")
            continue
        pd.DataFrame(known_rows).to_csv(save_dir/"test_results.csv", index=False)

        # 评估 Unknown
        unknown_metrics, unknown_rows = evaluate_unknown_dataset(
            unknown_ds, unknown_loader, model, DEVICE, "Unknown Set",
            CONF_TYPE_BRAND, CONF_DEVICE, idx2type, idx2brand, idx2device, logger
        )
        if unknown_metrics is None:
            logger.error("Unknown set evaluation failed, skip.")
            continue
        pd.DataFrame(unknown_rows).to_csv(save_dir/"unknown_results.csv", index=False)

        # 保存本模型指标汇总
        md = pd.DataFrame([known_metrics, unknown_metrics])
        md.to_csv(save_dir/"metrics_summary.csv", index=False)
        logger.info(f"Saved per-model metrics -> {save_dir/'metrics_summary.csv'}")

        # 追加到总汇总
        summary_rows.append({
            "model_path": pt_path,
            "feat_key": feat_key, "fusion": fusion, "seed": seed, "tasks": tasks, "feat_combo": feat_combo,
            **{f"known_{k}": v for k,v in known_metrics.items() if k!="dataset"},
            **{f"unknown_{k}": v for k,v in unknown_metrics.items() if k!="dataset"},
        })

    # 写总汇总
    pd.DataFrame(summary_rows).to_csv(Path(EVAL_ROOT)/"_all_models_summary.csv", index=False)
    print(f"[DONE] Wrote overall summary: {Path(EVAL_ROOT) / '_all_models_summary.csv'}")

if __name__ == "__main__":
    main()
