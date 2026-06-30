# -*- coding: utf-8 -*-
"""
featureFeasibilityStudy.py
Experiment A: Feature Feasibility Study (A1 + A2 + A3 + Report)
========================================================
目标：
  在不训练模型的前提下，验证所选特征（统计 + 序列衍生 + 原始字节衍生）
  对 "类型(type)" 与 "厂商(brand)" 两个标签的区分能力；型号(device)分析可选开启。

实现内容（本文件）：
  - ✅ A1 单特征统计检验与排名（已实现）
      * 读取统计特征 CSV
      * 从序列/原始字节 NPZ 计算 16 个衍生指标，各8个（处理 0 填充、mask、original_len）
      * 合并所有特征，30统计 + 8序列 + 8原始字节 + is_behavior=1，共 47 数值列；
      * 合并后总列数维度：30统计 + 8序列 + 8原始字节 + 4标签样本 （type_label, brand_label, device_label, sample_file） + \
      * 元数据9 （device, is_behavior, set_type, sample_base, behavior_type, idle_group, file_path, seq_feature_path, raw_feature_path） = 59维
      * 计算每列特征对标签的：互信息(MI)、ANOVA F 或 Kruskal–Wallis H、p 值、效应量(Eta²/ε²)
      * Levene 方差齐性检验 p 值记录
      * 生成排序表 CSV、Top-K 箱线图、小提琴图、特征×标签热力图
  - ⏳ A2 整体特征空间可分性（PCA→UMAP），计算 Silhouette/CH/DB，
        在嵌入空间做 KMeans 与真标签比较 NMI/ARI，并输出 UMAP 按标签着色散点图
  - ⏳ A3 例证展示（序列/原始字节衍生指标），优先展示偏好特征，回退 A1 排名/高方差列
  - ⏳ A1-device 型号标签分析（基于 A1 的可选开关，占位）
  - Report：汇总 A1/A2/A3 的关键结果，生成 Markdown 报告（附图路径）

数据假设：
  1) 样本划分表：SPLIT_CSV（与 uk_full_split.csv 同结构）
     - 必备列：file_path, seq_feature_path, raw_feature_path, is_behavior, set_type,
              type_label, brand_label, device_label, sample_file, sample_base,
              behavior_type, idle_group
  2) 统计特征 *_stat.csv：包含 30 + 1 is_behavior 特征 + 标签列（如你示例）
  3) 序列特征 *_seq.npz：包含 feature_matrix(L,3) 或 (3,L)，mask(L,), original_len, is_behavior
  4) 原始字节 *_raw.npz：包含 raw_matrix(P,128), mask(P,), original_len, is_behavior
     - 字段屏蔽已在生成时完成，0 代表 padding/屏蔽字节

使用方式：
  - 修改 CONFIG 的路径、开关，直接运行：python featureFeasibilityStudy.py
  - 输出：
    OUT_DIR/
      A1/*.csv, *.png
      A2/*.csv, *.png
      A3/*.png
      report/ExpA_report_YYYYmmdd_HHMMSS.md
      logs/, cache/

注意：
  - 序列 NPZ：feature_matrix 形状为 (3, L)，包含长度/方向/间隔三路；L 为 128(闲时)或 256(行为)。
    填充为 0 的位置不可参与统计；请用 mask 或 original_len 筛选有效部分。
  - 原始字节 NPZ：raw_matrix 形状为 (P, 128)，P 为 128(闲时)或 256(行为)，每包取 128 字节，不足 0 填充。
    行级 mask 或 original_len 用来筛选有效包行；列级没有 mask 时，0 既可能代表 padding 也可能是有效字节，
    因此本脚本采用“弱语义统计”（非零比率、熵、run-length）以提高鲁棒性。
"""

from pathlib import Path
import json
import math
import warnings
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import f_oneway, kruskal, levene
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")  # 后台绘图，保存为文件
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    normalized_mutual_info_score, adjusted_rand_score
)
from sklearn.cluster import KMeans

# UMAP 可选导入（没装时会自动回退）
try:
    import umap.umap_ as umap
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False


# =========================
# CONFIG: 运行选项与路径
# =========================

# 视角选择：可包含 "idle", "activity", "mixed"
VIEWS = ["idle", "activity"]    # 例如只跑闲时：["idle"]

# 目标标签：默认 type/brand；如需型号，把 RUN_DEVICE_LABEL 打开或在 TARGETS 加 "device"
TARGETS = ["type", "brand"] # 可以根据需要选择要几个标签
RUN_DEVICE_LABEL = False    # True 时会额外对 device_label 做 A1 分析

# 是否排除 unknown 样本（unknownType/Brand 或 set_type=="unknown"）
EXCLUDE_UNKNOWN = True      

# A1 Top-K 可视化的特征数量（箱线/小提琴）（例如 4）
TOPK = 4

# 可视化时每类最多样本数（None 表示不采样）
SAMPLE_CAP_PER_CLASS = None

# ==== A2 相关开关 ====
A2_USE_WINSORIZE = True      # 是否对每个特征做左右1%裁尾（抗异常值）
A2_WINSORIZE_PCT = 0.01      # 裁尾比例(0~0.5)，1%通常足够
A2_STD_SCALER = "standard"   # "standard" 或 "robust"，对特征做标准化
A2_MAX_ROWS = None           # 若想对计算也限流(如>60000太大)，可填整数；None 表示使用全部
A2_PLOT_SAMPLE_CAP = 30000   # UMAP散点图最多绘图的点数，避免图片过大

# UMAP 参数
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

# ==== A3 相关开关 ====
# 优先展示A3_FEATURE_PREFS中这些“有代表性”的衍生指标（若不存在会自动回退到 A1 排名Top特征）
# A3_FEATURE_PREFS中的个标签中的特征可以根据特征重要性排名进行替换，替换成对应的特征进行展示
# 按标签分别给出候选清单；可按需增减或调整顺序
A3_FEATURE_PREFS = {
    "type":  ["seq_iat_entropy", "raw_nonzero_ratio", "seq_len_entropy", "avg_pkt_len"],
    "brand": ["direction_switch_rate", "raw_byte_entropy_head16", "payload_bytes_ratio", "raw_byte_entropy_256"],
    "device": []  # 型号一般不主打展示，留空或自行添加
}
A3_USE_A1_RANK = True   # 若候选缺失，是否从 A1 排名表里取 top 特征做展示
A3_VIOLIN = False       # True 则画小提琴图，False 画箱线图
A3_MAX_CLASSES = 10     # 单图最多展示的类别数（频次Top-N）

# ==== 报告生成相关 ====
REPORT_TOPK_PER_TARGET = 8   # 报告中每个目标展示A1排名Top-K（表格）
REPORT_TITLE = "Experiment A: Feature Feasibility Study (A1/A2/A3 Summary)"
REPORT_AUTHOR = "Jason Hu"
REPORT_ADD_SYSTEM_INFO = True

# 路径：请改为你的真实路径
SPLIT_CSV = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk/3_uk_full_split.csv"  # 样本划分情况保存路径

STAT_ROOT = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature/uk" # 初始统计特征路径
SEQ_ROOT  = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_sequence_feature_matrix/uk" # 初始序列特征路径
RAW_ROOT  = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_rawByte_feature_matrix/uk"  # 初始原始字节特征路径

OUT_DIR   = "/home/hyj/unknownDeviceIdentification/dataset/12_expA_outputs/uk" # 输出文件保存路径

# 标签映射（可选：如需将字符串映射为索引做 MI/ANOVA，填上 json 路径；否则自动编码）
TYPE2IDX_JSON   = None  # "/path/to/type2idx.json"
BRAND2IDX_JSON  = None  # "/path/to/brand2idx.json"
DEVICE2IDX_JSON = None  # "/path/to/device2idx.json"

# 序列通道顺序，允许值集合：{"len", "dir", "iat"}，长度必须为 3，且互不重复
SEQ_CHANNEL_ORDER = ["len", "iat", "dir"]


# =========================
# 工具：目录/映射/读取/过滤
# =========================

def ensure_out_dirs():
    root = Path(OUT_DIR)
    for sub in ["A1", "A2", "A3", "logs", "cache", "report"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def load_label_maps() -> Dict[str, Dict[str, int]]:
    def _maybe(path):
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    return {
        "type2idx": _maybe(TYPE2IDX_JSON),
        "brand2idx": _maybe(BRAND2IDX_JSON),
        "device2idx": _maybe(DEVICE2IDX_JSON),
    }


def load_split_csv(path_csv: str) -> pd.DataFrame:
    print(f"[INFO] 读取划分文件: {path_csv}")
    df = pd.read_csv(path_csv)
    print(f"[INFO] 样本总数: {len(df)}")
    expected_cols = {
        "file_path","device","is_behavior","set_type","type_label","brand_label",
        "device_label","sample_file","sample_base","seq_feature_path","raw_feature_path",
        "behavior_type","idle_group"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        warnings.warn(f"[WARN] 划分文件缺少列: {missing}")
    return df


def filter_view(df_all: pd.DataFrame, view: str, exclude_unknown: bool) -> pd.DataFrame:
    df = df_all.copy()
    if view == "idle":
        df = df[df["is_behavior"] == 0]
    elif view == "activity":
        df = df[df["is_behavior"] == 1]
    elif view == "mixed":
        pass
    else:
        raise ValueError(f"未知视角: {view}")

    if exclude_unknown:
        before = len(df)
        df = df[df["set_type"] != "unknown"]
        if "type_label" in df.columns:
            df = df[df["type_label"] != "unknownType"]
        if "brand_label" in df.columns:
            df = df[df["brand_label"] != "unknownBrand"]
        after = len(df)
        print(f"[INFO] 过滤 unknown：{before} -> {after}")
    df = df.reset_index(drop=True)
    print(f"[INFO] 视角={view} 样本数={len(df)} (行为占比={df['is_behavior'].mean():.3f})")
    return df


# =========================
# 载入统计特征（CSV）
# =========================

STAT_FEATURE_COLUMNS = [
    # 30 数值特征 + 1 标记（is_behavior），与你提供的列一一对应
    "packet_count","avg_pkt_len","std_pkt_len","max_pkt_len","min_pkt_len",
    "total_bytes","payload_bytes_total","payload_bytes_ratio",
    "up_pkt_count","down_pkt_count","up_bytes","down_bytes",
    "up_down_pkt_ratio","up_down_byte_ratio","udp_ratio",
    "avg_iat","std_iat","min_iat","max_iat","pkt_rate","pkt_interval_entropy",
    "burst_count","heartbeat_period_fft","active_ratio","burstiness",
    "tcp_count","udp_count","session_count","unique_dst_ports","entropy_pkt_size",
    "is_behavior"
]

LABEL_COLUMNS = ["type_label","brand_label","device_label","sample_file"]

def safe_read_stat_csv(stat_csv_path: str) -> dict:
    """
    读取单个 *_stat.csv，返回一个 dict（键为特征/标签列）
    若文件空/损坏，返回 None
    """
    try:
        df = pd.read_csv(stat_csv_path)
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        # 仅保留关心的列（存在则取）
        out = {}
        for c in STAT_FEATURE_COLUMNS:
            if c in row:
                out[c] = row[c]
        for c in LABEL_COLUMNS:
            if c in row:
                out[c] = row[c]
        return out
    except Exception as e:
        warnings.warn(f"[WARN] 读取统计特征失败: {stat_csv_path} -> {e}")
        return None


def load_and_merge_stat_features(df_view: pd.DataFrame, stat_root: str) -> pd.DataFrame:
    """
    将 df_view 中的每个 file_path 指向的 *_stat.csv 读取并合并。
    """
    print(f"[INFO] 载入统计特征 CSV ...")
    records = []
    missing = 0
    for idx, row in df_view.iterrows():
        p = row["file_path"]
        if not Path(p).exists():
            warnings.warn(f"[WARN] 统计特征文件不存在: {p}")
            missing += 1
            records.append({})
            continue
        record = safe_read_stat_csv(p)
        if record is None:
            missing += 1
            records.append({})
        else:
            records.append(record)
    df_stats = pd.DataFrame(records)
    # 将标签缺失补到 df_view 的相应列
    for c in LABEL_COLUMNS:
        if c not in df_stats.columns and c in df_view.columns:
            df_stats[c] = df_view[c].values
    # 必要元数据（非数值）从 df_view 带上
    for c in ["device","is_behavior","set_type","sample_base","behavior_type","idle_group","file_path","seq_feature_path","raw_feature_path"]:
        if c in df_view.columns:
            df_stats[c] = df_view[c].values

    print(f"[INFO] 统计特征载入完成，缺失文件数: {missing}, 合并后形状: {df_stats.shape}")
    return df_stats


# =======================================
# 序列衍生指标（L×3，处理 0 填充/掩码） 
# =======================================

SEQ_DERIVED_COLUMNS = [
    "seq_len_mean","seq_len_std","seq_len_entropy",
    "seq_iat_mean","seq_iat_std","seq_iat_entropy",
    "direction_up_ratio","direction_switch_rate"
]

def _shannon_entropy_from_vector(x: np.ndarray, bins: int = 32) -> float:
    """对连续变量构建直方图估计并计算香农熵（忽略 NaN）"""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist.astype(float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _count_switches_per_second(dir_vec: np.ndarray, iat_vec: np.ndarray) -> float:
    """
    方向切换率：相邻符号变化次数 / 总时长（秒）
    - dir_vec: 方向序列（>0 视为上行，<=0 视为下行；若是 {0,1}，则 1 为上行）
    - iat_vec: 包间隔（秒），总时长为其和；缺失时用长度替代，避免除零
    """
    if dir_vec.size == 0:
        return 0.0
    # 方向二值化
    d = (dir_vec > 0).astype(int)
    # 计算相邻变化次数
    switches = int((d[1:] != d[:-1]).sum()) if d.size >= 2 else 0
    total_time = float(np.nansum(iat_vec)) if iat_vec.size else float(len(d))
    total_time = total_time if total_time > 1e-6 else float(len(d))
    return float(switches / total_time)


def compute_seq_derived(df_view: pd.DataFrame, seq_root: str) -> pd.DataFrame:
    print(f"[INFO] 计算序列衍生指标（来自 *_seq.npz） ...")
    out = pd.DataFrame(index=df_view.index, columns=SEQ_DERIVED_COLUMNS, dtype=float)

    valid_set = {"len","dir","iat"}
    if set(SEQ_CHANNEL_ORDER) != valid_set: # set: 无序不重复集合
        raise ValueError(f"SEQ_CHANNEL_ORDER 必须是 {valid_set} 的一个全排列，目前是: {SEQ_CHANNEL_ORDER}")

    miss = 0
    # 仅首次打印一次方向信息，避免刷屏
    printed_orientation_hint = False

    for i, row in df_view.iterrows():
        p = row["seq_feature_path"]
        if not isinstance(p, str) or not Path(p).exists():
            miss += 1
            continue
        try:
            npz = np.load(p)
            fm = npz["feature_matrix"]  # 可能是 (3, L) 或 (L, 3)
            mask = npz["mask"] if "mask" in npz.files else None
            original_len = int(npz["original_len"]) if "original_len" in npz.files else None

            if fm.ndim != 2 or 3 not in fm.shape:
                raise ValueError(f"feature_matrix 形状异常: {fm.shape}")

            # ---- 关键：自动识别通道轴 ----
            # ch_first: (3, L)；ch_last: (L, 3)
            if fm.shape[0] == 3 and fm.shape[1] != 3:
                ch_first = True
                L = fm.shape[1]
                if not printed_orientation_hint:
                    print(f"[INFO] 序列矩阵方向：channels-first (3, L)，L≈{L}")
                    printed_orientation_hint = True
                # 通道索引映射
                ch_map = {name: idx for idx, name in enumerate(SEQ_CHANNEL_ORDER)}
                len_vec = fm[ch_map["len"], :]   # (L,)
                iat_vec = fm[ch_map["iat"], :]   # (L,)
                dir_vec = fm[ch_map["dir"], :]   # (L,)

            elif fm.shape[1] == 3 and fm.shape[0] != 3:
                ch_first = False
                L = fm.shape[0]
                if not printed_orientation_hint:
                    print(f"[INFO] 序列矩阵方向：channels-last (L, 3)，L≈{L}")
                    printed_orientation_hint = True
                # 通道索引映射（在列）
                ch_map = {name: idx for idx, name in enumerate(SEQ_CHANNEL_ORDER)}
                len_vec = fm[:, ch_map["len"]].reshape(-1)  # (L,)
                iat_vec = fm[:, ch_map["iat"]].reshape(-1)  # (L,)
                dir_vec = fm[:, ch_map["dir"]].reshape(-1)  # (L,)

            else:
                # 非法形状，例如 (3,3) 或 (L,L)
                raise ValueError(f"无法判定通道轴: {fm.shape}")

            # ---- 有效索引：优先用 mask，其次 original_len ----
            valid_idx = np.ones(L, dtype=bool)
            if mask is not None:
                mask = np.asarray(mask).reshape(-1)
                if mask.size == L:
                    valid_idx = mask.astype(bool)
                else:
                    # 少见：mask 跟轴不一致；尽量容错
                    warnings.warn(f"[WARN] mask 长度({mask.size})与 L({L}) 不一致：{p}，将退化为 original_len 规则")
                    if original_len is not None:
                        valid_idx = np.zeros(L, dtype=bool)
                        valid_idx[:min(original_len, L)] = True
            elif original_len is not None:
                valid_idx = np.zeros(L, dtype=bool)
                valid_idx[:min(original_len, L)] = True
            # 否则：保留默认全 True

            # ---- 切取有效部分 ----
            len_valid = len_vec[valid_idx]
            iat_valid = iat_vec[valid_idx]
            dir_valid = dir_vec[valid_idx]

            # ---- 计算衍生统计 ----
            seq_len_mean = float(np.nanmean(len_valid)) if len_valid.size else 0.0
            seq_len_std  = float(np.nanstd(len_valid))  if len_valid.size else 0.0
            seq_len_entropy = _shannon_entropy_from_vector(len_valid, bins=32)

            seq_iat_mean = float(np.nanmean(iat_valid)) if iat_valid.size else 0.0
            seq_iat_std  = float(np.nanstd(iat_valid))  if iat_valid.size else 0.0
            seq_iat_entropy = _shannon_entropy_from_vector(iat_valid, bins=32)

            # 方向：>0 视为上行
            direction_up_ratio = float((dir_valid > 0).mean()) if dir_valid.size else 0.0
            direction_switch_rate = _count_switches_per_second(dir_valid, iat_valid)

            out.loc[i, :] = [
                seq_len_mean, seq_len_std, seq_len_entropy,
                seq_iat_mean, seq_iat_std, seq_iat_entropy,
                direction_up_ratio, direction_switch_rate
            ]

        except Exception as e:
            warnings.warn(f"[WARN] 读取序列特征失败: {p} -> {e}")
            miss += 1

    print(f"[INFO] 序列衍生指标完成，缺失/失败样本数: {miss}")
    return out



# =========================
# 原始字节衍生指标（P×128）
# =========================

RAW_DERIVED_COLUMNS = [
    "raw_nonzero_ratio",
    "raw_row_sparsity_mean","raw_row_sparsity_std",
    "raw_byte_entropy_256","raw_byte_entropy_head16",
    "raw_payload_row_ratio",
    "raw_avg_runlen_zero","raw_avg_runlen_nonzero"
]

def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts.astype(float) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _avg_run_length(binary_vec: np.ndarray, value: int) -> float:
    """
    计算 binary_vec 中等于 value 的平均 run 长度（沿一维向量）
    用于估计“连续零字节”的平均长度或“连续非零字节”的平均长度。
    """
    if binary_vec.size == 0:
        return 0.0
    runs = []
    cnt = 0
    for b in binary_vec:
        if b == value:
            cnt += 1
        else:
            if cnt > 0:
                runs.append(cnt)
            cnt = 0
    if cnt > 0:
        runs.append(cnt)
    return float(np.mean(runs)) if runs else 0.0


def compute_raw_derived(df_view: pd.DataFrame, raw_root: str) -> pd.DataFrame:
    print(f"[INFO] 计算原始字节衍生指标（来自 *_raw.npz） ...")
    out = pd.DataFrame(index=df_view.index, columns=RAW_DERIVED_COLUMNS, dtype=float)
    miss = 0

    for i, row in df_view.iterrows():
        p = row["raw_feature_path"]
        if not isinstance(p, str) or not Path(p).exists():
            miss += 1
            continue
        try:
            npz = np.load(p)
            rm = npz["raw_matrix"]  # 形状 (P, 128)
            mask = npz["mask"] if "mask" in npz.files else None
            original_len = int(npz["original_len"]) if "original_len" in npz.files else None

            P, W = rm.shape  # P 行数据包，W=128 字节
            # 有效行：mask 或 [:original_len]
            if mask is not None:
                valid_rows = mask.astype(bool).reshape(-1)
                valid_rows = valid_rows[:P]
            elif original_len is not None:
                valid_rows = np.zeros(P, dtype=bool)
                valid_rows[:min(original_len, P)] = True
            else:
                valid_rows = np.ones(P, dtype=bool)

            data_valid = rm[valid_rows, :]  # 仅保留有效行
            if data_valid.size == 0:
                out.loc[i, :] = [0]*len(RAW_DERIVED_COLUMNS)
                continue

            # 非零比率（弱语义）：反映“有效字节密度”
            nonzero = (data_valid != 0).astype(np.uint8)
            total_bytes = data_valid.size
            nonzero_count = int(nonzero.sum())
            raw_nonzero_ratio = float(nonzero_count / total_bytes) if total_bytes > 0 else 0.0

            # 行级稀疏度（零比例）均值/方差
            row_zero_ratio = 1.0 - nonzero.mean(axis=1)  # 每行零比例
            raw_row_sparsity_mean = float(np.mean(row_zero_ratio))
            raw_row_sparsity_std  = float(np.std(row_zero_ratio))

            # 全体字节直方图的熵（0..255）
            flat = data_valid.reshape(-1)
            counts = np.bincount(flat, minlength=256)
            raw_byte_entropy_256 = _entropy_from_counts(counts)

            # 前 16 字节熵（握手/头部的弱语义模式）
            head = data_valid[:, :16].reshape(-1)
            head_counts = np.bincount(head, minlength=256)
            raw_byte_entropy_head16 = _entropy_from_counts(head_counts)

            # 有效行占比（payload 行比率）
            raw_payload_row_ratio = float(valid_rows.mean())

            # run-length：对展开后的二值向量计算连续 0/非0 的平均 run 长度
            binvec = (flat != 0).astype(np.uint8)
            raw_avg_runlen_zero    = _avg_run_length(binvec, value=0)
            raw_avg_runlen_nonzero = _avg_run_length(binvec, value=1)

            out.loc[i, :] = [
                raw_nonzero_ratio,
                raw_row_sparsity_mean, raw_row_sparsity_std,
                raw_byte_entropy_256, raw_byte_entropy_head16,
                raw_payload_row_ratio,
                raw_avg_runlen_zero, raw_avg_runlen_nonzero
            ]

        except Exception as e:
            warnings.warn(f"[WARN] 读取原始字节特征失败: {p} -> {e}")
            miss += 1

    print(f"[INFO] 原始字节衍生指标完成，缺失/失败样本数: {miss}")
    return out


# =========================
# 合并特征表
# =========================

def merge_feature_tables(df_stats: pd.DataFrame,
                         df_seq_derived: pd.DataFrame,
                         df_raw_derived: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_stats.join(df_seq_derived, how="left").join(df_raw_derived, how="left")

    # 将数值列统一为 float（标签/元数据除外）
    for c in df_feat.columns:
        if c in LABEL_COLUMNS or c in ["device","set_type","sample_base","behavior_type","idle_group","file_path","seq_feature_path","raw_feature_path"]:
            continue
        try:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
        except Exception:
            pass

    print(f"[INFO] 合并后特征表形状: {df_feat.shape}")
    return df_feat


# =========================
# A1：单特征统计检验与排名
# =========================

def _encode_labels(series: pd.Series, provided_map: Dict[str,int] = None) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    将字符串标签编码为整数。优先使用提供的映射；否则用 LabelEncoder 自动编码。
    返回：整数数组、映射字典（str->int）
    """
    if provided_map:
        mapping = dict(provided_map)
        arr = series.map(lambda x: mapping.get(str(x), -1)).values
        mask = arr >= 0
        return arr[mask], mapping  # 上层需对齐 X 同步过滤
    else:
        le = LabelEncoder()
        arr = le.fit_transform(series.values.astype(str))
        mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
        return arr, mapping


def _group_by_label_values(x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    """将特征 x 按标签 y 分组，返回每个类别对应的一维数组（去除 NaN）"""
    groups = []
    for cls in np.unique(y):
        xi = x[y == cls]
        xi = xi[np.isfinite(xi)]
        if xi.size > 0:
            groups.append(xi)
    return groups


def _effect_size_anova_eta2(groups: List[np.ndarray]) -> float:
    """
    一元 ANOVA 的 Eta^2 近似：SS_between / SS_total
    这里用 F 与自由度估计也可，但我们直接从分组均值与整体均值估计。
    """
    if not groups:
        return 0.0
    ns = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups], dtype=float)
    grand_mean = np.average(means, weights=ns)
    ss_between = np.sum(ns * (means - grand_mean) ** 2)
    ss_total = np.sum([np.sum((g - grand_mean) ** 2) for g in groups])
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _effect_size_kruskal_epsilon2(groups: List[np.ndarray]) -> float:
    """
    Kruskal–Wallis 的 epsilon^2 估计（常用近似）
    ε^2 = (H - k + 1) / (N - k)
    """
    try:
        k = len(groups)
        N = int(np.sum([g.size for g in groups]))
        if k < 2 or N <= k:
            return 0.0
        # 仅计算 H，不取 p
        H, _ = kruskal(*groups)
        eps2 = (H - k + 1) / (N - k)
        return float(max(0.0, min(1.0, eps2)))
    except Exception:
        return 0.0


def _choose_anova_or_kruskal(groups: List[np.ndarray]) -> Tuple[str, float, float, float]:
    """
    选择 ANOVA 或 Kruskal：
      - 如果每组 n>=20 且 Levene 方差齐性 p>0.05，用 ANOVA
      - 否则用 Kruskal
    返回：(method, stat, p_value, effect_size)
    """
    # Levene 齐性检验（仅用于判断；真正的 p 值另行计算并返回）
    lev_p = 1.0
    try:
        if len(groups) >= 2 and all(g.size >= 2 for g in groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _, lev_p = levene(*groups, center="median")
    except Exception:
        lev_p = 1.0

    all_n_ge20 = all(g.size >= 20 for g in groups)
    if all_n_ge20 and lev_p > 0.05:
        # ANOVA
        try:
            F, p = f_oneway(*groups)
            eta2 = _effect_size_anova_eta2(groups)
            return "ANOVA_F", float(F), float(p), float(eta2)
        except Exception:
            # 回退 Kruskal
            pass

    # Kruskal–Wallis
    try:
        H, p = kruskal(*groups)
        eps2 = _effect_size_kruskal_epsilon2(groups)
        return "KW_H", float(H), float(p), float(eps2)
    except Exception:
        return "NA", float("nan"), float("nan"), 0.0


def _levene_p(groups: List[np.ndarray]) -> float:
    try:
        if len(groups) >= 2 and all(g.size >= 2 for g in groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _, p = levene(*groups, center="median")
            return float(p)
        return float("nan")
    except Exception:
        return float("nan")


def _mutual_info_safe(x: np.ndarray, y: np.ndarray) -> float:
    """MI 计算的封装：去 NaN，保持与 y 对齐"""
    mask = np.isfinite(x)
    x2 = x[mask].reshape(-1, 1)
    y2 = y[mask]
    if x2.shape[0] < 5 or len(np.unique(y2)) < 2:
        return 0.0
    try:
        mi = mutual_info_classif(x2, y2, discrete_features=False, random_state=0)
        return float(mi[0])
    except Exception:
        return 0.0


def _numeric_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    """识别数值特征列（去除标签/元数据）"""
    exclude = set(LABEL_COLUMNS + [
        "device","set_type","sample_base","behavior_type","idle_group",
        "file_path","seq_feature_path","raw_feature_path"
    ])
    cols = []
    for c in df_feat.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df_feat[c]):
            cols.append(c)
    return cols


def _make_boxplot(df: pd.DataFrame, feature: str, target_col: str, out_path_png: Path, violin=False, max_classes: int = 10):
    """
    画箱线图/小提琴图；类别过多时只取频次 Top-N。
    """
    labels = df[target_col].astype(str)
    values = df[feature].astype(float)

    # 只取前 max_classes 个频次最高的类，避免图太挤
    counts = labels.value_counts()
    keep_classes = set(counts.head(max_classes).index)
    mask = labels.isin(keep_classes) & np.isfinite(values)
    labels = labels[mask]
    values = values[mask]

    classes = list(sorted(keep_classes, key=lambda k: (-counts[k], k)))
    data = [values[labels == cls].values for cls in classes]

    plt.figure(figsize=(max(6, len(classes)*0.6), 4.5))
    if violin:
        plt.violinplot(data, showmeans=True, showextrema=True)
    else:
        plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(classes)+1), classes, rotation=45, ha="right")
    plt.title(f"{feature} by {target_col}")
    plt.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_png, dpi=180)
    plt.close()


def _heatmap_feat_scores(mi_scores: Dict[str, Dict[str, float]], out_path_png: Path):
    """
    特征×标签的打分热力图（仅用 MI 做展示）
    mi_scores: {feature: {"type": mi1, "brand": mi2, ...}}
    """
    feats = sorted(mi_scores.keys())
    targets = sorted({t for d in mi_scores.values() for t in d.keys()})
    M = np.zeros((len(feats), len(targets)), dtype=float)
    for i, f in enumerate(feats):
        for j, t in enumerate(targets):
            M[i, j] = mi_scores[f].get(t, 0.0)

    # 归一化到 [0,1] 便于视觉比较
    if np.isfinite(M).any():
        mmin, mmax = np.nanmin(M), np.nanmax(M)
        if mmax > mmin:
            M = (M - mmin) / (mmax - mmin)

    plt.figure(figsize=(6 + len(targets)*0.6, 0.25*len(feats) + 2))
    im = plt.imshow(M, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Normalized MI")
    plt.yticks(range(len(feats)), feats)
    plt.xticks(range(len(targets)), targets)
    plt.title("Feature × Target Score (MI normalized)")
    plt.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_png, dpi=180)
    plt.close()


# ========== 下面是A1，A2，A3的具体实现 ==========

# =========================
# A1：单特征统计检验与排名
# =========================

def run_A1_single_feature_analysis(df_feat: pd.DataFrame,
                                   targets=("type", "brand"),
                                   out_dir=OUT_DIR,
                                   view="idle",
                                   topk=4,
                                   run_device_label=False):
    """
    A1 主流程：
      1) 选择数值特征列
      2) 对每个目标标签（type/brand）逐特征计算 MI + (ANOVA 或 Kruskal) + 效应量 + Levene
      3) 输出排序表 CSV
      4) 自动绘制 Top-K 箱线图/小提琴图（按当前目标标签分组）
      5) 生成 Feature×Target 的打分热力图
    """
    print(f"[A1] 开始：视角={view}，目标={targets}")

    # 确定标签列名
    target_map = {
        "type": "type_label",
        "brand": "brand_label",
        "device": "device_label"
    }
    _targets = list(targets)
    if run_device_label and "device" not in _targets:
        _targets.append("device")

    # 数值特征列
    value_cols = _numeric_feature_columns(df_feat)
    print(f"[A1] 数值特征列数量: {len(value_cols)}")

    # 准备 MI 热力图用的字典
    mi_for_heatmap: Dict[str, Dict[str, float]] = {f: {} for f in value_cols}

    for tgt in _targets:
        tgt_col = target_map[tgt]
        print(f"[A1] 处理目标：{tgt} ({tgt_col})")

        # 丢掉缺失标签
        df_clean = df_feat[pd.notna(df_feat[tgt_col])].copy()
        if df_clean.empty:
            warnings.warn(f"[A1] 目标 {tgt_col} 没有可用样本，跳过。")
            continue

        # 编码标签
        y_str = df_clean[tgt_col].astype(str)
        y_enc, mapping = _encode_labels(y_str, provided_map=None)
        # 与 y 对齐的索引掩码（若 _encode_labels 返回 -1 会被过滤，但我们这里使用 LabelEncoder，直接一一对齐）
        mask_y = np.ones(len(y_str), dtype=bool)

        rows = []
        levene_ps = {}
        for feat in value_cols:
            x = pd.to_numeric(df_clean[feat], errors="coerce").values
            # 对齐 y 掩码（此处 mask_y 全 True；保留逻辑以防后续引入映射过滤）
            x = x[mask_y]
            y = y_enc[mask_y]

            # MI
            mi = _mutual_info_safe(x, y)
            mi_for_heatmap[feat][tgt] = mi

            # 分组数据
            groups = _group_by_label_values(x, y)
            # 方差齐性（用于记录展示）
            lev_p = _levene_p(groups)
            levene_ps[feat] = lev_p

            # 选择 ANOVA 或 Kruskal（并返回效应量）
            method, stat, pval, eff = _choose_anova_or_kruskal(groups)

            rows.append({
                "feature": feat,
                "MI": mi,
                "method": method,
                "F_or_H": stat,
                "p_value": pval,
                "effect_size": eff,
                "levene_p": lev_p
            })

        df_rank = pd.DataFrame(rows)
        # 排序：先 MI，再 F/H，再效应量
        df_rank = df_rank.sort_values(
            by=["MI","F_or_H","effect_size"],
            ascending=[False, False, False],
            na_position="last"
        ).reset_index(drop=True)
        df_rank["rank"] = np.arange(1, len(df_rank)+1)

        out_csv = Path(out_dir) / "A1" / f"A1_{tgt}_rank_{view}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_rank.to_csv(out_csv, index=False)
        print(f"[A1] 排名表已保存：{out_csv} (前5项)\n{df_rank.head(5)}")

        # Top-K 图（箱线 + 小提琴）
        top_feats = df_rank["feature"].head(topk).tolist()
        for vis_kind in ["box", "violin"]:
            out_png = Path(out_dir) / "A1" / f"A1_{view}_{tgt}_top{topk}_{vis_kind}.png"
            # 将多个特征绘制到一张图中：这里采用“子图循环”
            n = len(top_feats)
            cols = min(2, n)
            rows = math.ceil(n / cols)
            plt.figure(figsize=(6*cols, 4.5*rows))
            for i, feat in enumerate(top_feats, start=1):
                plt.subplot(rows, cols, i)
                # 采样（可选）以避免超大数据绘图过慢
                df_plot = df_clean[[feat, tgt_col]].copy()
                df_plot = df_plot[np.isfinite(df_plot[feat])]
                if SAMPLE_CAP_PER_CLASS is not None:
                    # 每类最多采样 SAMPLE_CAP_PER_CLASS 条
                    parts = []
                    for cls, grp in df_plot.groupby(tgt_col):
                        if len(grp) > SAMPLE_CAP_PER_CLASS:
                            parts.append(grp.sample(SAMPLE_CAP_PER_CLASS, random_state=0))
                        else:
                            parts.append(grp)
                    df_plot = pd.concat(parts, ignore_index=True)

                # 调用单图绘制函数
                tmp_png = out_png.parent / f"__tmp_{feat}.png"
                _make_boxplot(df_plot, feature=feat, target_col=tgt_col, out_path_png=tmp_png, violin=(vis_kind=="violin"))
                # 将单图贴到子图上
                img = plt.imread(tmp_png)
                plt.imshow(img)
                plt.axis('off')
                tmp_png.unlink(missing_ok=True)

            plt.suptitle(f"Top-{topk} features by {tgt_col} ({vis_kind})")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(out_png, dpi=180)
            plt.close()
            print(f"[A1] 已保存 {vis_kind} 图：{out_png}")

    # 生成 MI 热力图（两列：type/brand；若开启 device 则三列）
    heat_png = Path(out_dir) / "A1" / f"A1_{view}_feat_score_heatmap.png"
    _heatmap_feat_scores(mi_for_heatmap, heat_png)
    print(f"[A1] 特征×标签热力图已保存：{heat_png}")


# =========================
# A2：整体特征空间可分性
# =========================

def _winsorize_df(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """对每列做左右裁尾（分位数剪裁），返回复制后的 DataFrame。"""
    if not 0 <= pct < 0.5:
        return df.copy()
    X = df.copy()
    for c in X.columns:
        col = X[c].values
        lo, hi = np.nanpercentile(col, [pct*100, 100 - pct*100])
        X[c] = np.clip(col, lo, hi)
    return X

def _standardize_df(df: pd.DataFrame, mode: str = "standard") -> Tuple[np.ndarray, object]:
    """对数值特征标准化，返回 numpy 数组与 scaler 对象。"""
    if mode == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X = scaler.fit_transform(df.values.astype(float))
    return X, scaler

def _scatter_umap(Z: np.ndarray, labels: np.ndarray, title: str, out_png: Path, sample_cap: int = None):
    """
    UMAP 二维散点图绘制（按 labels 着色）。
    - Z: (n, 2)
    - labels: (n,)
    """
    n = Z.shape[0]
    if sample_cap is not None and n > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=sample_cap, replace=False)
        Z_plot = Z[idx]
        labels_plot = labels[idx]
        print(f"[A2] 绘图采样: {n} -> {len(idx)}")
    else:
        Z_plot = Z
        labels_plot = labels

    # 为了避免类别过多导致图例过密，这里不画图例，只用颜色区分
    # 散点较多时使用较小点和透明度
    plt.figure(figsize=(7.5, 6))
    # 将标签编码到 [0..k-1]，使用默认 colormap
    le = LabelEncoder()
    c = le.fit_transform(labels_plot.astype(str))
    plt.scatter(Z_plot[:, 0], Z_plot[:, 1], c=c, s=6, alpha=0.6, edgecolors='none')
    plt.title(title)
    plt.xlabel("UMAP-1" if title.lower().startswith("umap") else "PC-1")
    plt.ylabel("UMAP-2" if title.lower().startswith("umap") else "PC-2")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_A2_global_feature_space(df_feat: pd.DataFrame,
                                targets=("type", "brand"),
                                out_dir=OUT_DIR,
                                view="idle"):
    """
    A2：整体特征空间可分性
    步骤：
      1) 选择数值特征列，删除含 NaN 的样本；可选裁尾 + 标准化
      2) PCA(n_components=0.95) -> UMAP(2D)（若无 UMAP 则退化到 PCA-2D）
      3) 对每个 target（type/brand）：
           - 用“真标签”在嵌入空间上计算 Silhouette/CH/DB
           - 在 UMAP 嵌入上做 KMeans(n_clusters=类别数)，算 NMI/ARI（与真标签比较）
           - 保存 UMAP 散点图（按该 target 着色）
      4) 记录指标到 A2_global_metrics.csv（可多次 append）
    """
    print(f"[A2] 开始：视角={view}，targets={targets}")

    # 选择数值特征列；丢弃方差为 0 的列，避免无信息列干扰
    num_cols = _numeric_feature_columns(df_feat)
    # 去掉常数列
    non_const_cols = [c for c in num_cols if pd.Series(df_feat[c]).nunique(dropna=True) > 1]
    if len(non_const_cols) < 2:
        print("[A2] 可用数值特征不足，跳过。")
        return
    X_df = df_feat[non_const_cols].copy()

    # 丢 NA 行
    mask_rows = np.all(np.isfinite(X_df.values), axis=1)
    X_df = X_df[mask_rows]
    df_used = df_feat.loc[mask_rows].copy()
    print(f"[A2] 清洗后样本数: {len(X_df)}，特征维度: {X_df.shape[1]}")

    # 若需要对计算限流（很大时），可在这里采样
    if A2_MAX_ROWS is not None and len(X_df) > A2_MAX_ROWS:
        print(f"[A2] 计算采样: {len(X_df)} -> {A2_MAX_ROWS}")
        X_df = X_df.sample(A2_MAX_ROWS, random_state=0)
        df_used = df_used.loc[X_df.index]

    # 可选裁尾（winsorize）
    if A2_USE_WINSORIZE:
        X_df = _winsorize_df(X_df, pct=A2_WINSORIZE_PCT)
        print(f"[A2] 已进行裁尾：±{int(A2_WINSORIZE_PCT*100)}%")

    # 标准化
    X, scaler = _standardize_df(X_df, mode=A2_STD_SCALER)
    print(f"[A2] 标准化完成：scaler={A2_STD_SCALER}")

    # PCA 保留 95% 方差
    pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
    X_pca = pca.fit_transform(X)
    var_covered = float(np.sum(pca.explained_variance_ratio_))
    n_pc = X_pca.shape[1]
    print(f"[A2] PCA 完成：PC数={n_pc}, 覆盖方差比={var_covered:.3f}")

    # UMAP 2D（若无 umap-learn，则退化到 PCA-2D）
    if _UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=0,
            n_components=2,
        )
        Z = reducer.fit_transform(X_pca)
        embed_name = "UMAP"
        print(f"[A2] UMAP 完成：n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}")
    else:
        # 退化：直接取 PCA-2D
        if X_pca.shape[1] >= 2:
            Z = X_pca[:, :2]
        else:
            # 极端情况：只有 1 个PC，就补零列
            Z = np.hstack([X_pca, np.zeros((X_pca.shape[0], 1))])
        embed_name = "PCA2"
        print("[A2] 未检测到 umap-learn，退化为 PCA-2D。若需 UMAP，请 pip install umap-learn")

    # 指标文件（追加）
    metrics_path = Path(out_dir) / "A2" / "A2_global_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    target_map = {"type": "type_label", "brand": "brand_label", "device": "device_label"}

    for tgt in targets:
        tgt_col = target_map[tgt]
        if tgt_col not in df_used.columns:
            print(f"[A2] 跳过目标 {tgt}：列 {tgt_col} 不存在。")
            continue

        y_raw = df_used[tgt_col].astype(str).values
        # 类别数
        y_le = LabelEncoder()
        y = y_le.fit_transform(y_raw)
        n_classes = len(y_le.classes_)
        n_samples = Z.shape[0]

        if n_classes < 2 or n_samples < 10:
            print(f"[A2] {tgt} 可用类别<2或样本过少，跳过。")
            continue

        # 1) 使用“真标签”评估嵌入空间的分离度
        try:
            sil = float(silhouette_score(Z, y))
        except Exception:
            sil = float("nan")
        try:
            ch = float(calinski_harabasz_score(Z, y))
        except Exception:
            ch = float("nan")
        try:
            db = float(davies_bouldin_score(Z, y))
        except Exception:
            db = float("nan")

        # 2) 在嵌入上做 KMeans(n_clusters=类别数)，与真标签比对 NMI/ARI
        try:
            km = KMeans(n_clusters=n_classes, random_state=0, n_init=10)
            pred = km.fit_predict(Z)
            nmi = float(normalized_mutual_info_score(y, pred))
            ari = float(adjusted_rand_score(y, pred))
            inertia = float(km.inertia_)
        except Exception:
            nmi = float("nan")
            ari = float("nan")
            inertia = float("nan")

        # 记录
        metrics_rows.append({
            "view": view,
            "label": tgt,
            "n_samples": n_samples,
            "n_features_used": X_df.shape[1],
            "embed": embed_name,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "nmi_kmeans": nmi,
            "ari_kmeans": ari,
            "kmeans_inertia": inertia,
            "pca_n_components": n_pc,
            "pca_var_covered": var_covered
        })

        # 画图（按该 target 着色）
        title = f"{embed_name} ({view}) colored by {tgt}"
        out_png = Path(out_dir) / "A2" / f"A2_umap_{view}_{tgt}.png"
        _scatter_umap(Z, y_raw, title, out_png, sample_cap=A2_PLOT_SAMPLE_CAP)
        print(f"[A2] 已保存嵌入图：{out_png}")

    # 追加写指标 CSV
    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        if metrics_path.exists():
            df_old = pd.read_csv(metrics_path)
            df_metrics = pd.concat([df_old, df_metrics], ignore_index=True)
        df_metrics.to_csv(metrics_path, index=False)
        print(f"[A2] 指标表已更新：{metrics_path}")
    else:
        print("[A2] 无可记录指标（可能是标签列缺失或类别不足）。")


# =========================
# A3：例证展示（序列/原始字节衍生）
# =========================

def _pick_showcase_features(df_feat: pd.DataFrame,
                            target: str,
                            view: str,
                            out_dir: str,
                            k: int,
                            prefs: List[str],
                            use_a1_rank: bool) -> List[str]:
    """
    为 A3 选择要展示的特征列表（长度 <= k）：
      优先：prefs 中存在于 df_feat 的列（按给定顺序）
      其次：若 use_a1_rank=True，尝试读取 A1 排名表，从高到低选取未重复的列
      兜底：按方差从高到低在数值列中补足
    """
    pick = []

    # 1) 按偏好列表挑选
    for f in prefs:
        if f in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[f]):
            pick.append(f)
        if len(pick) >= k:
            break

    # 2) 回退：读 A1 排名
    if use_a1_rank and len(pick) < k:
        rank_csv = Path(out_dir) / "A1" / f"A1_{target}_rank_{view}.csv"
        if rank_csv.exists():
            try:
                df_rank = pd.read_csv(rank_csv)
                for f in df_rank["feature"].tolist():
                    if f in pick:
                        continue
                    if f in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[f]):
                        pick.append(f)
                        if len(pick) >= k:
                            break
            except Exception:
                pass

    # 3) 最后兜底：高方差列中补足
    if len(pick) < k:
        num_cols = _numeric_feature_columns(df_feat)
        # 去掉已挑
        num_cols = [c for c in num_cols if c not in pick]
        if num_cols:
            variances = [(c, float(np.nanvar(df_feat[c].values))) for c in num_cols]
            variances.sort(key=lambda x: x[1], reverse=True)
            for c, _ in variances:
                pick.append(c)
                if len(pick) >= k:
                    break

    # 去重并截断
    uniq = []
    for f in pick:
        if f not in uniq:
            uniq.append(f)
    return uniq[:k]


def _plot_feature_by_label(df: pd.DataFrame,
                           feature: str,
                           target_col: str,
                           out_png: Path,
                           violin: bool = False,
                           max_classes: int = 20,
                           sample_cap_per_class: int | None = None):
    """
    单特征的分布对比图（箱线或小提琴），按 target 分组。
    会：
      - 过滤非数值/NaN
      - 只保留类别频次Top-N，以免图太挤
      - 每类可限流采样，避免超大数据绘图耗时
    """
    dfp = df[[feature, target_col]].copy()
    dfp = dfp[np.isfinite(dfp[feature])]
    if dfp.empty:
        return False

    # 只保留Top-N类别
    counts = dfp[target_col].value_counts()
    keep = set(counts.head(max_classes).index)
    dfp = dfp[dfp[target_col].isin(keep)]
    if dfp.empty:
        return False

    # 每类采样
    if sample_cap_per_class is not None:
        parts = []
        for cls, grp in dfp.groupby(target_col):
            if len(grp) > sample_cap_per_class:
                parts.append(grp.sample(sample_cap_per_class, random_state=0))
            else:
                parts.append(grp)
        dfp = pd.concat(parts, ignore_index=True)

    # 组装绘图数据
    classes = list(sorted(keep, key=lambda k: (-counts[k], k)))
    data = [dfp.loc[dfp[target_col] == cls, feature].values for cls in classes]

    plt.figure(figsize=(max(6, len(classes)*0.6), 4.5))
    if violin:
        plt.violinplot(data, showmeans=True, showextrema=True)
    else:
        plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(classes)+1), classes, rotation=45, ha="right")
    plt.title(f"{feature} by {target_col}  (n={len(dfp)})")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True



def run_A3_showcase_plots(df_feat: pd.DataFrame,
                          targets=("type", "brand"),
                          out_dir=OUT_DIR,
                          view="idle",
                          topk=4):
    """
    A3：序列/原始字节衍生指标的例证展示
    逻辑：
      - 针对每个 target（type/brand[/device]），优先用 A3_FEATURE_PREFS[target] 中的候选；
        若缺失则回退到 A1 排名表选 Top 特征，再兜底用高方差数值列补足，最终选 <= topk 个。
      - 为每个选中的特征生成一张图（箱线或小提琴），按 target 分组展示分布。
      - 文件命名：A3_{view}_{target}_{feature}_{box/violin}.png
    """
    print(f"[A3] 开始：视角={view}，targets={targets}")
    target_map = {"type": "type_label", "brand": "brand_label", "device": "device_label"}

    for tgt in targets:
        tgt_col = target_map[tgt]
        if tgt_col not in df_feat.columns:
            print(f"[A3] 跳过目标 {tgt}：列 {tgt_col} 不存在。")
            continue

        # 过滤缺失标签
        df_clean = df_feat[pd.notna(df_feat[tgt_col])].copy()
        if df_clean.empty:
            print(f"[A3] 跳过目标 {tgt}：无可用样本。")
            continue

        # 选择要展示的特征
        prefs = A3_FEATURE_PREFS.get(tgt, [])
        to_show = _pick_showcase_features(
            df_clean, target=tgt, view=view, out_dir=out_dir,
            k=topk, prefs=prefs, use_a1_rank=A3_USE_A1_RANK
        )
        print(f"[A3] 目标 {tgt} 选用展示特征：{to_show}")

        # 逐个特征绘图
        for feat in to_show:
            vis_kind = "violin" if A3_VIOLIN else "box"
            out_png = Path(out_dir) / "A3" / f"A3_{view}_{tgt}_{feat}_{vis_kind}.png"
            ok = _plot_feature_by_label(
                df_clean, feature=feat, target_col=tgt_col, out_png=out_png,
                violin=A3_VIOLIN, max_classes=A3_MAX_CLASSES,
                sample_cap_per_class=SAMPLE_CAP_PER_CLASS
            )
            if ok:
                print(f"[A3] 已保存：{out_png}")
            else:
                print(f"[A3] 跳过 {feat}（数据不足或类别过滤后为空）")


def run_A1_device_label_analysis(df_feat: pd.DataFrame,
                                 out_dir=OUT_DIR,
                                 view="idle",
                                 topk=4):
    print(f"[A1-device] 占位：型号标签分析（后续可复用 A1 子函数）")


# =========================
# 报告生成（Markdown）
# =========================

def _md_hdr(title: str, author: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    s = f"# {title}\n\n"
    s += f"- **Author**: {author}\n- **Generated**: {now}\n\n"
    return s

def _md_table_from_df(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "_(no data)_\n\n"
    d = df.head(max_rows)
    return d.to_markdown(index=False) + "\n\n"

def _rel_path_str(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)

def _append_images_if_exist(md: str, img_paths: List[Path], base: Path, title: str) -> str:
    imgs = [p for p in img_paths if p.exists()]
    if not imgs:
        return md + f"**{title}**: _(no figures)_\n\n"
    md += f"**{title}**:\n\n"
    for p in imgs:
        md += f"![{p.name}]({_rel_path_str(p, base)})\n\n"
    return md

def generate_report(out_dir: str,
                    views: List[str],
                    targets: List[str],
                    include_device: bool,
                    report_topk: int = 8,
                    title: str = REPORT_TITLE,
                    author: str = REPORT_AUTHOR):
    """
    汇总 A1/A2/A3 输出，生成 Markdown 报告：
      - A1：各视角/目标的 Top-K 排名表 + Top-K 箱/提琴图 + MI 热力图
      - A2：全局指标（汇总表）+ UMAP 图
      - A3：例证图
    """
    print("[REPORT] 生成报告 ...")
    base = Path(out_dir)
    rpt_dir = base / "report"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rpt_path = rpt_dir / f"ExpA_report_{stamp}.md"

    # 报告头
    md = _md_hdr(title, author)

    # 系统信息（可选）
    if REPORT_ADD_SYSTEM_INFO:
        md += "## System & Config\n\n"
        md += f"- Views: `{views}`\n- Targets: `{targets}`\n- Include device label: `{include_device}`\n"
        md += f"- Out dir: `{out_dir}`\n- UMAP: `{'enabled' if _UMAP_AVAILABLE else 'disabled (fallback to PCA2)'}`\n\n"

    # A1
    md += "## A1: Single-Feature Analysis\n\n"
    target_map = {"type": "type_label", "brand": "brand_label", "device": "device_label"}
    A1_dir = base / "A1"
    for view in views:
        md += f"### View: **{view}**\n\n"
        for tgt in targets + (["device"] if include_device and "device" not in targets else []):
            rank_csv = A1_dir / f"A1_{tgt}_rank_{view}.csv"
            md += f"#### Target: **{tgt}**\n\n"
            if rank_csv.exists():
                df_rank = pd.read_csv(rank_csv)
                md += _md_table_from_df(df_rank[["rank","feature","MI","method","F_or_H","p_value","effect_size","levene_p"]], max_rows=report_topk)
            else:
                md += "_(rank file not found)_\n\n"

            # 插图：Top-K 箱线/提琴 + 热力图
            figs = [
                A1_dir / f"A1_{view}_{tgt}_top{TOPK}_box.png",
                A1_dir / f"A1_{view}_{tgt}_top{TOPK}_violin.png",
            ]
            md = _append_images_if_exist(md, figs, base, title=f"Top-{TOPK} plots (box/violin)")
        # MI 热力图（按视角统一一张）
        md = _append_images_if_exist(md, [A1_dir / f"A1_{view}_feat_score_heatmap.png"], base, title="Feature × Target (MI) Heatmap")

    # A2
    md += "## A2: Global Separability (PCA→UMAP)\n\n"
    A2_dir = base / "A2"
    metrics_csv = A2_dir / "A2_global_metrics.csv"
    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        # 展示每视角/目标的最近一次记录（或全部）
        md += _md_table_from_df(df_metrics, max_rows=50)
    else:
        md += "_(metrics file not found)_\n\n"

    for view in views:
        md += f"### View: **{view}**\n\n"
        for tgt in targets + (["device"] if include_device and "device" not in targets else []):
            fig = A2_dir / f"A2_umap_{view}_{tgt}.png"
            md = _append_images_if_exist(md, [fig], base, title=f"Embedding colored by {tgt}")

    # A3
    md += "## A3: Showcase (Sequence/Raw-byte Derived)\n\n"
    A3_dir = base / "A3"
    for view in views:
        md += f"### View: **{view}**\n\n"
        for tgt in targets + (["device"] if include_device and "device" not in targets else []):
            # 选取最多 6 张 A3 图（按文件名前缀匹配）
            figs = sorted(list(A3_dir.glob(f"A3_{view}_{tgt}_*_{'violin' if A3_VIOLIN else 'box'}.png")))
            if figs:
                md = _append_images_if_exist(md, figs[:6], base, title=f"Showcase for {tgt}")
            else:
                md += f"**Showcase for {tgt}**: _(no figures)_\n\n"

    # 写文件
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] 报告已生成：{rpt_path}")


# =========================
# 主流程
# =========================

def main():
    ensure_out_dirs()
    label_maps = load_label_maps()

    df_all = load_split_csv(SPLIT_CSV)

    for view in VIEWS:
        print("\n" + "="*80)
        print(f"[MAIN] 视角开始：{view}")
        df_view = filter_view(df_all, view=view, exclude_unknown=EXCLUDE_UNKNOWN)

        # 1) 载入统计特征
        df_stats = load_and_merge_stat_features(df_view, STAT_ROOT)

        # 2) 计算序列衍生指标（处理 0 填充/掩码）
        df_seq_derived = compute_seq_derived(df_view, SEQ_ROOT)

        # 3) 计算原始字节衍生指标（处理 0 填充/掩码）
        df_raw_derived = compute_raw_derived(df_view, RAW_ROOT)

        # 4) 合并成完整特征表
        df_feat = merge_feature_tables(df_stats, df_seq_derived, df_raw_derived)

        # 5) A1：单特征统计检验与排名（含图表）
        run_A1_single_feature_analysis(
            df_feat,
            targets=tuple(TARGETS),
            out_dir=OUT_DIR,
            view=view,
            topk=TOPK,
            run_device_label=RUN_DEVICE_LABEL
        )

        # 6) A2：整体特征空间可分性
        run_A2_global_feature_space(
            df_feat, 
            targets=tuple(TARGETS), 
            out_dir=OUT_DIR, 
            view=view
        )

        # 7）A3：序列与原始字节衍生指标的针对性展示
        run_A3_showcase_plots(
            df_feat, 
            targets=tuple(TARGETS), 
            out_dir=OUT_DIR, 
            view=view, 
            topk=TOPK
        )

        # 可选：型号标签分析（如需在 A1 中单独跑 device）
        if RUN_DEVICE_LABEL and "device" not in TARGETS:
            run_A1_device_label_analysis(df_feat, out_dir=OUT_DIR, view=view, topk=TOPK)

    # 8) 报告生成（汇总以上各视角/目标的结果）
    generate_report(
        out_dir=OUT_DIR,
        views=VIEWS,
        targets=TARGETS,
        include_device=RUN_DEVICE_LABEL,
        report_topk=REPORT_TOPK_PER_TARGET,
        title=REPORT_TITLE,
        author=REPORT_AUTHOR
    )

    print("\n✅ Experiment A 完成（A1/A2/A3 + 报告已生成）。")


if __name__ == "__main__":
    main()
