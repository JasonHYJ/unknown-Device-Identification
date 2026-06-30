#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_runner.py — Experiment C (Baselines & Fusion) unified trainer with efficiency logging

WHAT THIS FILE DOES
-------------------
本脚本用于 **实验C：对比实验** 的统一训练入口，覆盖以下模型族：
1) 传统机器学习（仅统计特征，单任务）：SVM / RandomForest / XGBoost
2) 单模态深度学习（端到端，单任务）：1D-CNN(Seq) / LSTM(Seq) / ByteCNN(Raw)
3) 自监督/对比学习线探针（单任务）：LinearProbe on Seq-SSL / Seq-CL 表征（64d）
4) 多模态融合（多任务三头）：EarlyConcat / AttentionFusion（对照你的主方法，不含门控）

数据读取与处理流程
------------------
- 三个划分文件：
  /home/hyj/unknownDeviceIdentification/dataset/12_expC_outputs/uk/3_uk_train.csv
  /home/hyj/unknownDeviceIdentification/dataset/12_expC_outputs/uk/3_uk_test.csv
  /home/hyj/unknownDeviceIdentification/dataset/12_expC_outputs/uk/3_uk_unknown.csv

- 每条样本的路径列（来自你的 split CSV，**直接使用，不做字符串替换**）：
  * 统计（CL 版 31 维CSV）：stat_feature_path_cl
  * 序列矩阵（3×L .npz）：seq_feature_path
  * 原始字节矩阵（L×128 .npz）：raw_feature_path
  * 序列自监督嵌入（128d .npy）：seq_embed_feature_path_ssl
  * 序列对比学习嵌入（64d .npy）：seq_embed_feature_path_cl
  * 原始字节对比学习嵌入（64d .npy）：raw_embed_feature_path_cl

- 标签列：type_label / brand_label / device_label（通过 type2idx.json / brand2idx.json / device2idx.json 转成索引）

训练输出目录（统一）
-------------------
/home/hyj/unknownDeviceIdentification/dataset/12_expC_outputs/uk/
  baselines/  # 经典/单模态/线探针
    models/*.pt|pkl|json
    logs/*.csv  ← 每个模型训练的按 epoch 指标与效率（时间/吞吐/显存）
  fusion/     # EarlyConcat / AttentionFusion
    models/*.pt
    logs/*.csv

如何选择要运行的模型
--------------------
本脚本 **不再使用命令行参数**，而是在文件顶部的 CONFIG 中进行选择：
  CONFIG = {
    "model": "svm",        # 见可选项
    "tasks": "T",          # 单任务：T/B/D；多模态融合：TBD（固定三任务）
    "seed": 0,
    "cuda_visible_devices": "2",  # 选择 GPU，如 "0" 或 "2"
    "epochs": 40, "batch_size": 64, "num_workers": 4, "lr": 1e-3,
  }
修改保存后直接运行： python train_runner.py

注意
----
- 多模态的 ours 在此仅占位（与 EarlyConcat 同结构），你的主方法请使用你原先的训练脚本产出。
- 经典 ML 使用 sklearn/xgboost，需安装相应依赖；若未安装 xgboost，会报友好错误。
"""

# ---------- GPU 选择（在导入 torch 之前设置） ----------
import os
CONFIG = {
    "model": "attnfusion",            # ["svm","rf","xgb","cnnseq","lstmseq","bytecnn","lp_seqssl","lp_seqcl","earlyconcat","attnfusion","ours"]
    "tasks": "T",              # 单任务基线：T|B|D；融合：TBD（忽略此项）
    "seed": 0,
    "cuda_visible_devices": "3",  # 例如 "2" 表示使用第3张卡；多卡训练不在此脚本范围内
    "epochs": 40,
    "batch_size": 64,
    "num_workers": 4,
    "lr": 1e-3,
    "use_cpu": False,          # 若设为 True，强制使用 CPU
}
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import time

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Sklearn / XGBoost
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ---------------------- Constants ----------------------
DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"
# 使用你刚合并产出的 12_expC_outputs 版本 CSV
TRAIN_CSV = f"{DATA_ROOT}/12_expC_outputs/uk/3_uk_train.csv"
TEST_CSV  = f"{DATA_ROOT}/12_expC_outputs/uk/3_uk_test.csv"
UNKN_CSV  = f"{DATA_ROOT}/12_expC_outputs/uk/3_uk_unknown.csv"
LABEL_DIR = f"{DATA_ROOT}/11_multitask_training/uk"  # 标签映射还是沿用原来的 json

EXP_ROOT  = f"{DATA_ROOT}/12_expC_outputs/uk"

# 统一固定的时间步长度（避免 batch 内尺寸不一致）
SEQ_FIXED_LEN = 256   # (3, L) for sequence inputs to cnnseq/lstmseq
RAW_FIXED_LEN = 256   # (L, 128) for raw byte inputs to bytecnn

# ---------------------- Utilities ----------------------
def load_label_maps(label_dir: str):
    """读取三类标签映射字典。"""
    p = Path(label_dir)
    type2idx   = json.loads((p/"type2idx.json").read_text())
    brand2idx  = json.loads((p/"brand2idx.json").read_text())
    device2idx = json.loads((p/"device2idx.json").read_text())
    return type2idx, brand2idx, device2idx

TYPE2IDX, BRAND2IDX, DEVICE2IDX = load_label_maps(LABEL_DIR)

def set_seed(seed: int):
    """固定随机种子，保证可复现。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def idx_from_map(m, key):
    """将标签字符串映射到索引；若缺失则返回 -1。"""
    return m[key] if key in m else -1

def reduce_128_to_64(vec: np.ndarray)->np.ndarray:
    """自监督 128d 表征降到 64d（简单分块均值降维），与对比学习 64d 对齐。"""
    v = vec.reshape(-1).astype(np.float32)
    if v.shape[0] == 64:
        return v
    if v.shape[0] >= 128:
        try:
            return v[:128].reshape(64,2).mean(axis=1).astype(np.float32)
        except Exception:
            return v[:64].astype(np.float32)
    out = np.zeros(64, np.float32); out[:len(v)] = v
    return out

# ---------------------- Dataset variants ----------------------
class StatOnlyDataset(Dataset):
    """经典 ML 单任务数据集：读取 CL 版 31 维统计特征（stat_feature_path_cl）。"""
    def __init__(self, csv_path: str, task: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        need = ["type_label","brand_label","device_label","stat_feature_path_cl"]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV missing columns: {miss}")
        self.task = task

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stat_fp = str(row["stat_feature_path_cl"])
        # 读取 31 维统计特征（文件第二行是数值行）
        try:
            v = pd.read_csv(stat_fp, skiprows=1, header=None).iloc[0,:31].values.astype(np.float32)
        except Exception as e:
            print(f"[WARN] read stat_cl fail: {stat_fp} -> {e}")
            v = np.zeros(31, np.float32)
        # 选择任务标签
        if self.task == "T":
            y = idx_from_map(TYPE2IDX, row["type_label"])
        elif self.task == "B":
            y = idx_from_map(BRAND2IDX, row["brand_label"])
        else:
            y = idx_from_map(DEVICE2IDX, row["device_label"])
        return v, y

class SeqMatrixDataset(Dataset):
    """单模态序列端到端（CNN/LSTM），读取 3×L 矩阵（seq_feature_path）。"""
    def __init__(self, csv_path: str, task: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        need = ["type_label","brand_label","device_label","is_behavior","seq_feature_path"]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV missing columns: {miss}")
        self.task = task

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = str(row["seq_feature_path"])  # 直接使用 CSV 中的序列矩阵路径
        isb = int(row["is_behavior"])
        L_target = SEQ_FIXED_LEN   # 统一固定长度，忽略 is_behavior

        try:
            z = np.load(p)
            mat = z["feature_matrix"].astype(np.float32)   # 期望 (3, L) 或 (L, 3)
            # ---- 自动纠正维度朝向 ----
            # 目标是得到 (3, L)
            if mat.ndim != 2:
                raise ValueError(f"feature_matrix ndim={mat.ndim}, expect 2")
            if mat.shape[0] == 3:
                # 已是 (3, L)，ok
                pass
            elif mat.shape[1] == 3:
                # 是 (L, 3) -> 转置
                mat = mat.T
            else:
                # 两个维度都不是 3，无法判断，强制尝试转置到 (3, L_target) 以避免训练中断
                print(f"[WARN] seq shape ambiguous {mat.shape} @ {p}; fallback zeros.")
                raise ValueError("ambiguous seq shape")

            # ---- 安全裁剪/填充到固定长度 L_target ----
            L = mat.shape[1]
            if L >= L_target:
                mat = mat[:, :L_target]
            else:
                pad = np.zeros((3, L_target - L), dtype=np.float32)
                mat = np.concatenate([mat, pad], axis=1)

        except Exception as e:
            print(f"[WARN] read seq npz fail: {p} -> {e}")
            # 按 is_behavior 给出正确的时间长度，填零占位
            mat = np.zeros((3, L_target), np.float32)

        # 选择任务标签
        if self.task == "T":
            y = idx_from_map(TYPE2IDX, row["type_label"])
        elif self.task == "B":
            y = idx_from_map(BRAND2IDX, row["brand_label"])
        else:
            y = idx_from_map(DEVICE2IDX, row["device_label"])

        return torch.tensor(mat, dtype=torch.float32), y

class RawMatrixDataset(Dataset):
    """单模态字节端到端（ByteCNN），读取 L×128 矩阵（raw_feature_path）。"""
    def __init__(self, csv_path: str, task: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        need = ["type_label","brand_label","device_label","is_behavior","raw_feature_path"]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV missing columns: {miss}")
        self.task = task

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = str(row["raw_feature_path"])  # 直接使用 CSV 中的原始字节矩阵路径
        isb = int(row["is_behavior"])
        L_target = RAW_FIXED_LEN   # 统一固定长度，忽略 is_behavior

        try:
            z = np.load(p)
            mat = z["raw_matrix"].astype(np.float32)   # 期望 (L, 128) 或 (128, L)
            if mat.ndim != 2:
                raise ValueError(f"raw_matrix ndim={mat.ndim}, expect 2")

            # ---- 自动纠正维度朝向到 (L, 128) ----
            if mat.shape[1] == 128:
                # 已是 (L,128) -> ok
                pass
            elif mat.shape[0] == 128:
                # 是 (128, L) -> 转置
                mat = mat.T
            else:
                # 两个维都不是 128，无法判断
                print(f"[WARN] raw shape ambiguous {mat.shape} @ {p}; fallback zeros.")
                raise ValueError("ambiguous raw shape")

            # ---- 安全裁剪/填充到固定长度 L_target ----
            L = mat.shape[0]
            if L >= L_target:
                mat = mat[:L_target, :]
            else:
                pad = np.zeros((L_target - L, 128), np.float32)
                mat = np.concatenate([mat, pad], axis=0)

        except Exception as e:
            print(f"[WARN] read raw npz fail: {p} -> {e}")
            mat = np.zeros((L_target, 128), np.float32)

        if self.task == "T":
            y = idx_from_map(TYPE2IDX, row["type_label"])
        elif self.task == "B":
            y = idx_from_map(BRAND2IDX, row["brand_label"])
        else:
            y = idx_from_map(DEVICE2IDX, row["device_label"])

        # 增加通道维度以适配 2D 卷积：(1, L, 128)
        mat = np.expand_dims(mat, axis=0)
        return torch.tensor(mat, dtype=torch.float32), y

class EmbeddingDataset(Dataset):
    """线探针（Seq-SSL / Seq-CL），读取 128/64 维序列嵌入（对应列：_ssl/_cl）。"""
    def __init__(self, csv_path: str, task: str, mode: str = "ssl"):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        need = ["type_label","brand_label","device_label"]
        need += ["seq_embed_feature_path_ssl"] if mode=="ssl" else ["seq_embed_feature_path_cl"]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV missing columns: {miss}")
        self.task = task
        self.mode  = mode  # "ssl" or "cl"

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = str(row["seq_embed_feature_path_ssl" if self.mode=="ssl" else "seq_embed_feature_path_cl"])
        try:
            emb = np.load(p).astype(np.float32)
            if self.mode == "ssl":
                emb = reduce_128_to_64(emb)  # 128d -> 64d
            else:
                emb = emb.reshape(-1)
                if emb.shape[0] != 64:
                    emb = reduce_128_to_64(emb)
        except Exception as e:
            print(f"[WARN] read embed fail: {p} -> {e}")
            emb = np.zeros(64, np.float32)

        if self.task == "T":
            y = idx_from_map(TYPE2IDX, row["type_label"])
        elif self.task == "B":
            y = idx_from_map(BRAND2IDX, row["brand_label"])
        else:
            y = idx_from_map(DEVICE2IDX, row["device_label"])

        return torch.tensor(emb, dtype=torch.float32), y

class MultiModalFusionDataset(Dataset):
    """多模态融合（EarlyConcat/AttentionFusion），读取 Stat31 + SeqCL64 + RawCL64。"""
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        need = ["type_label","brand_label","device_label",
                "stat_feature_path_cl","seq_embed_feature_path_cl","raw_embed_feature_path_cl"]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV missing columns: {miss}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 统计特征（前 31 列）
        try:
            s = pd.read_csv(str(row["stat_feature_path_cl"]), skiprows=1, header=None).iloc[0,:31].values.astype(np.float32)
        except Exception as e:
            print(f"[WARN] read stat_cl fail: {row['stat_feature_path_cl']} -> {e}")
            s = np.zeros(31, np.float32)
        # 序列嵌入（CL 64d；若出现 128d 也降到 64d 以保持健壮）
        try:
            q = np.load(str(row["seq_embed_feature_path_cl"])).astype(np.float32).reshape(-1)
            if q.shape[0] != 64: q = reduce_128_to_64(q)
        except Exception as e:
            print(f"[WARN] read seq_cl fail: {row['seq_embed_feature_path_cl']} -> {e}")
            q = np.zeros(64, np.float32)
        # 原始字节嵌入（CL 64d；健壮处理同上）
        try:
            r = np.load(str(row["raw_embed_feature_path_cl"])).astype(np.float32).reshape(-1)
            if r.shape[0] != 64: r = reduce_128_to_64(r)
        except Exception as e:
            print(f"[WARN] read raw_cl fail: {row['raw_embed_feature_path_cl']} -> {e}")
            r = np.zeros(64, np.float32)

        x_stat = torch.tensor(s, dtype=torch.float32)
        x_seq  = torch.tensor(q, dtype=torch.float32)
        x_raw  = torch.tensor(r, dtype=torch.float32)

        yt = idx_from_map(TYPE2IDX, row["type_label"])
        yb = idx_from_map(BRAND2IDX, row["brand_label"])
        yd = idx_from_map(DEVICE2IDX, row["device_label"])

        return (x_stat, x_seq, x_raw), (yt, yb, yd)

# ---------------------- DL model defs ----------------------
class CNNSeq(nn.Module):
    """1D-CNN over sequence (3xL) -> single-task classifier"""
    def __init__(self, num_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_cls)

    def forward(self, x):  # x: (B,3,L)
        h = self.net(x).squeeze(-1)  # (B,128)
        return self.fc(h)

class LSTMSeq(nn.Module):
    """BiLSTM over sequence -> single-task classifier"""
    def __init__(self, num_cls: int, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden*2, num_cls)

    def forward(self, x):  # x: (B,3,L)
        x = x.transpose(1,2)  # (B,L,3)
        out,_ = self.lstm(x)  # (B,L,2H)
        h = out.mean(dim=1)
        return self.fc(h)

class ByteCNN(nn.Module):
    """2D CNN over raw byte matrix (1,L,128) -> single-task classifier"""
    def __init__(self, num_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, num_cls)

    def forward(self, x):  # x: (B,1,L,128)
        h = self.net(x).view(x.size(0), -1)  # (B,64)
        return self.fc(h)

class LinearProbe(nn.Module):
    """Linear classifier on top of a (64d) embedding."""
    def __init__(self, num_cls: int, in_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_cls)
    def forward(self, x):  # (B,in_dim)
        return self.fc(x)

class EarlyConcatFusion(nn.Module):
    """Concat [Stat31, Seq64, Raw64] -> Transformer -> three heads (multi-task)."""
    def __init__(self, num_type: int, num_brand: int, num_device: int, d_model=256):
        super().__init__()
        in_dim = 31 + 64 + 64  # 159
        self.fc = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_t = nn.Linear(d_model, num_type)
        self.cls_b = nn.Linear(d_model, num_brand)
        self.cls_d = nn.Linear(d_model, num_device)

    def forward(self, x_stat, x_seq, x_raw):
        x = torch.cat([x_stat, x_seq, x_raw], dim=1).unsqueeze(1)  # (B,1,159)
        h = self.enc(self.fc(x)).squeeze(1)   # (B,256)
        return self.cls_t(h), self.cls_b(h), self.cls_d(h)

class AttnFusion(nn.Module):
    """3-token self-attention fusion -> encoder -> three heads (multi-task)."""
    def __init__(self, num_type: int, num_brand: int, num_device: int, tok_dim=64, d_model=256):
        super().__init__()
        self.proj_stat = nn.Linear(31, tok_dim)
        self.proj_seq  = nn.Identity()  # already 64
        self.proj_raw  = nn.Identity()  # already 64

        self.self_attn = nn.TransformerEncoderLayer(d_model=tok_dim, nhead=4, batch_first=True)
        self.self_stack= nn.TransformerEncoder(self.self_attn, num_layers=2)

        self.fc = nn.Linear(tok_dim*3, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.cls_t = nn.Linear(d_model, num_type)
        self.cls_b = nn.Linear(d_model, num_brand)
        self.cls_d = nn.Linear(d_model, num_device)

    def forward(self, x_stat, x_seq, x_raw):
        t_stat = self.proj_stat(x_stat).unsqueeze(1)  # (B,1,64)
        t_seq  = self.proj_seq(x_seq).unsqueeze(1)    # (B,1,64)
        t_raw  = self.proj_raw(x_raw).unsqueeze(1)    # (B,1,64)
        tokens = torch.cat([t_stat, t_seq, t_raw], dim=1)  # (B,3,64)
        z = self.self_stack(tokens).reshape(tokens.size(0), -1)  # (B, 3*64)
        h = self.enc(self.fc(z).unsqueeze(1)).squeeze(1)   # (B,256)
        return self.cls_t(h), self.cls_b(h), self.cls_d(h)

# ---------------------- Training loops (with efficiency logs) ----------------------
def train_single_task(model, loader, vloader, device, epochs=40, lr=1e-3, max_grad_norm=2.0, log_csv_path: Path=None):
    """通用单任务训练循环（用于 CNN/LSTM/ByteCNN/LinearProbe）+ 效率记录到CSV。"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best = {"val_acc": 0.0, "state": None}
    print(f"[INFO] Start training single-task model on {device} for {epochs} epochs")
    
    logs = []
    t_total_start = time.perf_counter()
    for ep in range(1, epochs+1):
        # ---- Train ----
        ep_start = time.perf_counter()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for xb, y in loader:
            xb, y = xb.to(device), torch.tensor(y, dtype=torch.long, device=device)
            logits = model(xb)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            total_loss += loss.item() * len(y)
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total_samples += len(y)
        ep_train_time = time.perf_counter() - ep_start
        train_acc = total_correct / max(total_samples, 1)
        train_sps = total_samples / max(ep_train_time, 1e-8)  # samples/sec
        print(f"[TRAIN] Epoch {ep}/{epochs} | loss={total_loss/total_samples:.4f} | acc={train_acc:.4f}")

        # 验证
        v_start = time.perf_counter()
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, y in vloader:
                xb = xb.to(device)
                y = torch.tensor(y, dtype=torch.long, device=device)
                preds.extend(torch.argmax(model(xb), dim=1).cpu().tolist())
                ys.extend(y.cpu().tolist())
        val_time = time.perf_counter() - v_start
        val_acc = accuracy_score(ys, preds)
        val_sps = len(ys) / max(val_time, 1e-8)
        print(f"[VAL] Epoch {ep}/{epochs} | val_acc={val_acc:.4f} | best={best['val_acc']:.4f}")

        # ---- GPU 显存（可选）----
        if torch.cuda.is_available() and device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            mem_mb = 0.0

        print(f"[EPOCH {ep:02d}/{epochs}] "
              f"train: loss={total_loss/total_samples:.4f}, acc={train_acc:.4f}, time={ep_train_time:.2f}s, sps={train_sps:.1f} "
              f"| val: acc={val_acc:.4f}, time={val_time:.2f}s, sps={val_sps:.1f} "
              f"| max_mem={mem_mb:.1f}MiB | best_val={best['val_acc']:.4f}")

        if val_acc > best["val_acc"]:
            best["val_acc"] = val_acc
            best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
        
        logs.append({
            "epoch": ep,
            "train_loss": float(total_loss/total_samples),
            "train_acc": float(train_acc),
            "epoch_train_time_sec": float(ep_train_time),
            "train_samples_per_sec": float(train_sps),
            "val_acc": float(val_acc),
            "val_time_sec": float(val_time),
            "val_samples_per_sec": float(val_sps),
            "max_mem_MiB": float(mem_mb),
            "best_val_acc_so_far": float(best["val_acc"])
        })

    t_total = time.perf_counter() - t_total_start
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    print(f"[DONE] Training finished. total_time={t_total:.2f}s\n")

    # 保存日志CSV
    if log_csv_path is not None:
        df = pd.DataFrame(logs)
        df["total_time_sec"] = float(t_total)
        df.to_csv(log_csv_path, index=False)
        print(f"[SAVE] Train log CSV → {log_csv_path}")
    return model

def train_multi_task(model, loader, vloader, device, epochs=40, lr=1e-3, max_grad_norm=2.0, log_csv_path: Path=None):
    """多任务三头训练循环（EarlyConcat/AttentionFusion 占位实现）。"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best = {"val_macroF1": 0.0, "state": None}
    print(f"[INFO] Start training multi-task model on {device} for {epochs} epochs")

    logs = []
    t_total_start = time.perf_counter()
    for ep in range(1, epochs+1):
        # ---- Train ----
        ep_start = time.perf_counter()
        model.train()
        total_loss, total_samples = 0.0, 0
        for (xs, xtup) in loader:
            (x_stat, x_seq, x_raw) = xs
            yt, yb, yd = xtup
            x_stat = x_stat.to(device); x_seq = x_seq.to(device); x_raw = x_raw.to(device)
            yt = torch.tensor(yt, dtype=torch.long, device=device)
            yb = torch.tensor(yb, dtype=torch.long, device=device)
            yd = torch.tensor(yd, dtype=torch.long, device=device)

            logit_t, logit_b, logit_d = model(x_stat, x_seq, x_raw)
            loss = loss_fn(logit_t, yt) + loss_fn(logit_b, yb) + loss_fn(logit_d, yd)
            total_loss += loss.item() * len(yt)
            total_samples += len(yt)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        ep_train_time = time.perf_counter() - ep_start
        train_sps = total_samples / max(ep_train_time, 1e-8)

        # 验证集 macro-F1（T/B/D 平均）用于选最优
        v_start = time.perf_counter()
        model.eval()
        preds_t, ys_t = [], []
        preds_b, ys_b = [], []
        preds_d, ys_d = [], []
        with torch.no_grad():
            for (xs, xtup) in vloader:
                x_stat, x_seq, x_raw = xs
                yt, yb, yd = xtup
                x_stat = x_stat.to(device); x_seq = x_seq.to(device); x_raw = x_raw.to(device)
                logit_t, logit_b, logit_d = model(x_stat, x_seq, x_raw)
                preds_t.extend(torch.argmax(logit_t, dim=1).cpu().tolist()); ys_t.extend(yt.tolist())
                preds_b.extend(torch.argmax(logit_b, dim=1).cpu().tolist()); ys_b.extend(yb.tolist())
                preds_d.extend(torch.argmax(logit_d, dim=1).cpu().tolist()); ys_d.extend(yd.tolist())
        val_time = time.perf_counter() - v_start

        f1_t = f1_score(ys_t, preds_t, average="macro") if len(set(ys_t))>1 else 0.0
        f1_b = f1_score(ys_b, preds_b, average="macro") if len(set(ys_b))>1 else 0.0
        f1_d = f1_score(ys_d, preds_d, average="macro") if len(set(ys_d))>1 else 0.0
        macroF1 = (f1_t + f1_b + f1_d)/3.0
        val_sps = (len(ys_t)+len(ys_b)+len(ys_d))/3.0 / max(val_time, 1e-8)  # 平均吞吐
        print(f"[TRAIN] Epoch {ep}/{epochs} | loss={total_loss/max(len(loader.dataset),1):.4f} | val_macroF1={macroF1:.4f} | best={best['val_macroF1']:.4f}")

        # GPU 显存
        if torch.cuda.is_available() and device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            mem_mb = 0.0

        print(f"[EPOCH {ep:02d}/{epochs}] "
              f"train: loss={total_loss/max(total_samples,1):.4f}, sps={train_sps:.1f} "
              f"| val: macroF1={macroF1:.4f}, time={val_time:.2f}s, sps={val_sps:.1f} "
              f"| max_mem={mem_mb:.1f}MiB | best={best['val_macroF1']:.4f}")
        
        if macroF1 > best["val_macroF1"]:
            best["val_macroF1"] = macroF1
            best["state"] = {k:v.cpu() for k,v in model.state_dict().items()}
        
        logs.append({
            "epoch": ep,
            "train_loss_per_sample": float(total_loss/max(total_samples,1)),
            "train_samples_per_sec": float(train_sps),
            "val_macroF1": float(macroF1),
            "val_time_sec": float(val_time),
            "val_samples_per_sec": float(val_sps),
            "max_mem_MiB": float(mem_mb),
            "best_val_macroF1_so_far": float(best["val_macroF1"])
        })

    t_total = time.perf_counter() - t_total_start
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    print(f"[DONE] Training finished. total_time={t_total:.2f}s\n")

    if log_csv_path is not None:
        df = pd.DataFrame(logs)
        df["total_time_sec"] = float(t_total)
        df.to_csv(log_csv_path, index=False)
        print(f"[SAVE] Train log CSV → {log_csv_path}")
    return model

# ---------------------- Entry ----------------------
def main():
    print(f"[INFO] Visible GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES')} (mapped as cuda:0)")
    print(f"[INFO] Config: {CONFIG}")
    set_seed(CONFIG["seed"])
    # 设备选择：若 use_cpu=True 强制 CPU；否则优先 GPU
    device = torch.device("cpu" if CONFIG["use_cpu"] or not torch.cuda.is_available() else "cuda:0")
    print(f"[INFO] Using device: {device}")

    out_group = "baselines" if CONFIG["model"] in ["svm","rf","xgb","cnnseq","lstmseq","bytecnn","lp_seqssl","lp_seqcl"] else "fusion"
    model_dir = f"{EXP_ROOT}/{out_group}/models"; log_dir = f"{EXP_ROOT}/{out_group}/logs"
    ensure_dirs(model_dir); ensure_dirs(log_dir)

    # --------- Classic ML ---------
    if CONFIG["model"] in ["svm","rf","xgb"]:
        print(f"[INFO] Loading datasets for classic ML: {CONFIG['model']} ({CONFIG['tasks']})")
        ds_tr = StatOnlyDataset(TRAIN_CSV, CONFIG["tasks"])
        ds_va = StatOnlyDataset(TEST_CSV,  CONFIG["tasks"])
        print(f"[INFO] Loaded train/test dataset: {len(ds_tr)} / {len(ds_va)} samples")

        Xtr = []; ytr = []
        for i in range(len(ds_tr)):
            xi, yi = ds_tr[i]; Xtr.append(xi); ytr.append(yi)
        Xtr = np.vstack(Xtr); ytr = np.array(ytr)

        Xva = []; yva = []
        for i in range(len(ds_va)):
            xi, yi = ds_va[i]; Xva.append(xi); yva.append(yi)
        Xva = np.vstack(Xva); yva = np.array(yva)

        if CONFIG["model"] == "svm":
            model = Pipeline([("scaler", StandardScaler()),
                              ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))])
        elif CONFIG["model"] == "rf":
            model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=CONFIG["seed"])
        else:
            if not _HAS_XGB:
                raise RuntimeError("xgboost is not installed, please 'pip install xgboost'")
            model = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=300,
                                  subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
                                  random_state=CONFIG["seed"])
        t0 = time.perf_counter()
        print(f"[INFO] Training {CONFIG['model']} model...")

        model.fit(Xtr, ytr)
        t1 = time.perf_counter()
        test_acc = accuracy_score(yva, model.predict(Xva))
        print(f"[DONE] Training finished. Test acc={test_acc:.4f} | train_time={t1-t0:.2f}s")
        # 保存
        import joblib
        name = f"{CONFIG['model']}_statCL__{CONFIG['tasks']}_seed{CONFIG['seed']}.pkl"
        joblib.dump(model, Path(model_dir)/name)
        (Path(model_dir)/f"{name}.json").write_text(json.dumps({"config":CONFIG}, indent=2, ensure_ascii=False))
        # 经典ML也保存简易效率CSV
        pd.DataFrame([{
            "model": CONFIG["model"], "task": CONFIG["tasks"],
            "train_time_sec": float(t1-t0),
            "test_acc": float(test_acc)
        }]).to_csv(Path(log_dir)/f"{Path(name).stem}_trainlog.csv", index=False)
        print(f"[SAVE] Classic ML model saved: {Path(model_dir)/name}\n")
        return

    # --------- DL / Fusion ---------
    # 构建 DataLoader（单任务 or 多任务）
    if CONFIG["model"] in ["cnnseq","lstmseq"]:
        print(f"[INFO] Loading SeqMatrixDataset for {CONFIG['model']} ({CONFIG['tasks']})")
        tr = SeqMatrixDataset(TRAIN_CSV, CONFIG["tasks"]); va = SeqMatrixDataset(TEST_CSV, CONFIG["tasks"])
        print(f"[INFO] Loaded train/test dataset: {len(tr)} / {len(va)} samples")
        dl_tr = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"])
        dl_va = DataLoader(va, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        # 推断类别数
        _, y0 = tr[0]; num_cls = len(set([y for _,y in tr]))
        print(f"[INFO] Initialized model: {CONFIG['model']} (num_cls={num_cls})")
        net = CNNSeq(num_cls) if CONFIG["model"]=="cnnseq" else LSTMSeq(num_cls)

        log_path = Path(log_dir)/f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}_trainlog.csv"
        net = train_single_task(net, dl_tr, dl_va, device, epochs=CONFIG["epochs"], lr=CONFIG["lr"], log_csv_path=log_path)
        name = f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}.pt"
        torch.save(net.state_dict(), Path(model_dir)/name)
        (Path(model_dir)/f"{name}.json").write_text(json.dumps({"config":CONFIG}, indent=2, ensure_ascii=False))
        print(f"[SAVE] DL single-task model saved: {Path(model_dir)/name}")
        return

    if CONFIG["model"] == "bytecnn":
        print(f"[INFO] Loading RawMatrixDataset for bytecnn ({CONFIG['tasks']})")
        tr = RawMatrixDataset(TRAIN_CSV, CONFIG["tasks"]); va = RawMatrixDataset(TEST_CSV, CONFIG["tasks"])
        print(f"[INFO] Loaded train/test dataset: {len(tr)} / {len(va)} samples")
        dl_tr = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"])
        dl_va = DataLoader(va, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        _, y0 = tr[0]; num_cls = len(set([y for _,y in tr]))
        print(f"[INFO] Initialized model: bytecnn (num_cls={num_cls})")
        net = ByteCNN(num_cls)

        log_path = Path(log_dir)/f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}_trainlog.csv"
        net = train_single_task(net, dl_tr, dl_va, device, epochs=CONFIG["epochs"], lr=CONFIG["lr"], log_csv_path=log_path)
        name = f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}.pt"
        torch.save(net.state_dict(), Path(model_dir)/name)
        (Path(model_dir)/f"{name}.json").write_text(json.dumps({"config":CONFIG}, indent=2, ensure_ascii=False))
        print(f"[SAVE] DL single-task model saved: {Path(model_dir)/name}")
        return

    if CONFIG["model"] in ["lp_seqssl","lp_seqcl"]:
        mode = "ssl" if CONFIG["model"]=="lp_seqssl" else "cl"
        print(f"[INFO] Loading EmbeddingDataset for {CONFIG['model']} ({CONFIG['tasks']})")
        tr = EmbeddingDataset(TRAIN_CSV, CONFIG["tasks"], mode); va = EmbeddingDataset(TEST_CSV, CONFIG["tasks"], mode)
        print(f"[INFO] Loaded train/test dataset: {len(tr)} / {len(va)} samples")
        dl_tr = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"])
        dl_va = DataLoader(va, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        # 推断类别数
        _, y0 = tr[0]; num_cls = len(set([y for _,y in tr]))
        in_dim = 64  # 统一用 64d；SSL 128d 已在 Dataset 内降到 64d
        print(f"[INFO] Initialized model: LinearProbe (num_cls={num_cls}, in_dim={in_dim})")
        net = LinearProbe(num_cls, in_dim=in_dim)

        log_path = Path(log_dir)/f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}_trainlog.csv"
        net = train_single_task(net, dl_tr, dl_va, device, epochs=CONFIG["epochs"], lr=CONFIG["lr"], log_csv_path=log_path)
        name = f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}.pt"
        torch.save(net.state_dict(), Path(model_dir)/name)
        (Path(model_dir)/f"{name}.json").write_text(json.dumps({"config":CONFIG}, indent=2, ensure_ascii=False))
        print(f"[SAVE] Linear probe model saved: {Path(model_dir)/name}")
        return

    if CONFIG["model"] in ["earlyconcat","attnfusion","ours"]:
        print(f"[INFO] Loading MultiModalFusionDataset for {CONFIG['model']} (multi-task)")
        tr = MultiModalFusionDataset(TRAIN_CSV); va = MultiModalFusionDataset(TEST_CSV)
        print(f"[INFO] Loaded train/test dataset: {len(tr)} / {len(va)} samples")
        dl_tr = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"])
        dl_va = DataLoader(va, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        num_type = len(TYPE2IDX); num_brand = len(BRAND2IDX); num_device = len(DEVICE2IDX)
        if CONFIG["model"] == "earlyconcat":
            net = EarlyConcatFusion(num_type, num_brand, num_device)
        elif CONFIG["model"] == "attnfusion":
            net = AttnFusion(num_type, num_brand, num_device)
        else:
            # 占位：此处仍使用 EarlyConcat 结构；你的主方法请用独立脚本训练。
            net = EarlyConcatFusion(num_type, num_brand, num_device)
        print(f"[INFO] Initialized fusion model: {CONFIG['model']} (T/B/D = {num_type}/{num_brand}/{num_device})")

        log_path = Path(log_dir)/f"{CONFIG['model']}__TBD_seed{CONFIG['seed']}_trainlog.csv"
        net = train_multi_task(net, dl_tr, dl_va, device, epochs=CONFIG["epochs"], lr=CONFIG["lr"], log_csv_path=log_path)
        name = f"{CONFIG['model']}__TBD_seed{CONFIG['seed']}.pt"
        torch.save(net.state_dict(), Path(model_dir)/name)
        (Path(model_dir)/f"{name}.json").write_text(json.dumps({"config":CONFIG}, indent=2, ensure_ascii=False))
        print(f"[SAVE] Fusion multi-task model saved: {Path(model_dir)/name}")
        return

if __name__ == "__main__":
    main()
