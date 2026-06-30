#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_expC_splits.py

功能：
- 合并两套 split CSV（8_split_sample_info 与 11_multitask_training），
- 同时补充 9_learned_embeddings 的 SSL 路径（通过 CL 路径替换推断 & 存在性校验），
- 输出新的 3_uk_{train,test,unknown}.csv 到 12_expC_outputs/uk/，供实验C统一读取。

输入：
- A: /home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk/3_uk_{split}.csv
     列含：file_path, seq_feature_path, raw_feature_path, device, is_behavior, set_type, type_label, brand_label, device_label, sample_file, sample_base, behavior_type, idle_group
- B: /home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/3_uk_{split}.csv
     列含：stat_feature_path, seq_embed_feature_path, raw_embed_feature_path, device, is_behavior, set_type, type_label, brand_label, device_label, sample_file, sample_base, behavior_type, idle_group

输出：
- /home/hyj/unknownDeviceIdentification/dataset/12_expC_outputs/uk/3_uk_{split}.csv
  关键列：
    device,is_behavior,set_type,type_label,brand_label,device_label,sample_file,sample_base,behavior_type,idle_group
    stat_feature_path_orig, seq_feature_path, raw_feature_path
    stat_feature_path_cl,   seq_embed_feature_path_cl, raw_embed_feature_path_cl
    seq_embed_feature_path_ssl, raw_embed_feature_path_ssl
"""

import os
from pathlib import Path
import pandas as pd

DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"
SPLIT_A_DIR = f"{DATA_ROOT}/8_split_sample_info/uk"
SPLIT_B_DIR = f"{DATA_ROOT}/11_multitask_training/uk"
OUT_DIR     = f"{DATA_ROOT}/12_expC_outputs/uk"

SPLITS = ["train", "test", "unknown"]

# 替换规则：CL → SSL
REPL = {
    "seq": (
        "10_contrastive_embeddings/10_contrastive_sequence_embeddings",
        "9_learned_embeddings/9_learned_sequence_embeddings"
    ),
    "raw": (
        "10_contrastive_embeddings/10_contrastive_rawbyte_embeddings",
        "9_learned_embeddings/9_learned_rawbyte_embeddings"
    )
}

def infer_ssl_path_from_cl(cl_path: str, mode: str) -> str:
    """由 CL 路径推断 SSL 路径（若替换失败或为空，返回空字符串）。"""
    if not isinstance(cl_path, str) or len(cl_path) == 0:
        return ""
    src, dst = REPL[mode]
    return cl_path.replace(src, dst)

def merge_one_split(split: str):
    a_csv = Path(SPLIT_A_DIR) / f"3_uk_{split}.csv"
    b_csv = Path(SPLIT_B_DIR) / f"3_uk_{split}.csv"
    out_csv = Path(OUT_DIR) / f"3_uk_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not a_csv.exists():
        raise FileNotFoundError(f"A CSV not found: {a_csv}")
    if not b_csv.exists():
        raise FileNotFoundError(f"B CSV not found: {b_csv}")

    print(f"[INFO] Loading A(raw) from {a_csv}")
    dfA = pd.read_csv(a_csv)
    print(f"[INFO] Loading B(CL)  from {b_csv}")
    dfB = pd.read_csv(b_csv)

    # 统一用于 merge 的键（优先 sample_base）
    key_cols = ["sample_base"]
    # 兜底键（避免极少数缺失 sample_base 的情况）
    fallback_keys = ["device", "sample_file"]

    # 预处理：重命名 A 的原始路径列
    dfA = dfA.rename(columns={
        "file_path": "stat_feature_path_orig"
    })

    # 预处理：重命名 B 的 CL 列
    dfB = dfB.rename(columns={
        "stat_feature_path": "stat_feature_path_cl",
        "seq_embed_feature_path": "seq_embed_feature_path_cl",
        "raw_embed_feature_path": "raw_embed_feature_path_cl"
    })

    # 选择合并时保留的公共列（元信息）
    meta_cols = ["device","is_behavior","set_type","type_label","brand_label",
                 "device_label","sample_file","sample_base","behavior_type","idle_group"]

    # A 中用于输出的列
    cols_A = ["stat_feature_path_orig", "seq_feature_path", "raw_feature_path"] + meta_cols
    for c in cols_A:
        if c not in dfA.columns:
            raise ValueError(f"Column missing in A: {c}")

    # B 中用于输出的列
    cols_B = ["stat_feature_path_cl","seq_embed_feature_path_cl","raw_embed_feature_path_cl"] + meta_cols
    for c in cols_B:
        if c not in dfB.columns:
            raise ValueError(f"Column missing in B: {c}")

    # 先用 sample_base 精确 merge
    print("[INFO] Merging by 'sample_base' ...")
    merged = pd.merge(
        dfA[cols_A], dfB[cols_B],
        on=meta_cols if "sample_base" not in dfA.columns or "sample_base" not in dfB.columns else meta_cols,
        how="outer", suffixes=("_A","_B")
    )

    # 如果上面的合并没有把所有行对齐（少量 NA），尝试用 fallback 键补齐
    na_mask = merged["stat_feature_path_cl"].isna() | merged["stat_feature_path_orig"].isna()
    if na_mask.any():
        print(f"[WARN] Found {na_mask.sum()} rows with NA after primary merge. Trying fallback merge ...")
        # 左表用 A 的未配对行，与 B 做 (device, sample_file) 的补合并
        # 为简单稳妥，重新用 fallback 键 merge 一次：A join B on fallback_keys + is_behavior（以减少冲突）
        fb_keys = fallback_keys + ["is_behavior"]
        A_fb = dfA[cols_A].drop_duplicates(subset=fb_keys)
        B_fb = dfB[cols_B].drop_duplicates(subset=fb_keys)

        merged_fb = pd.merge(A_fb, B_fb, on=fb_keys, how="outer", suffixes=("_A","_B"))
        # 再根据 (device, sample_file, is_behavior) 对齐回 merged（这里用笛卡儿键合并）
        merged = pd.merge(
            merged.drop(columns=["stat_feature_path_orig","seq_feature_path","raw_feature_path",
                                 "stat_feature_path_cl","seq_embed_feature_path_cl","raw_embed_feature_path_cl"], errors="ignore"),
            merged_fb,
            on=fb_keys,
            how="left"
        )

    # 推断 SSL 路径
    print("[INFO] Inferring SSL paths from CL paths and verifying existence ...")
    def _infer_and_check(row, mode):
        cl_col = "seq_embed_feature_path_cl" if mode=="seq" else "raw_embed_feature_path_cl"
        cl_path = row.get(cl_col, "")
        ssl_path = infer_ssl_path_from_cl(cl_path, mode)
        if ssl_path and not os.path.exists(ssl_path):
            print(f"[WARN] SSL path not found ({mode}): {ssl_path}")
        return ssl_path

    merged["seq_embed_feature_path_ssl"] = merged.apply(lambda r: _infer_and_check(r, "seq"), axis=1)
    merged["raw_embed_feature_path_ssl"] = merged.apply(lambda r: _infer_and_check(r, "raw"), axis=1)

    # 列顺序（最终输出）
    out_cols = meta_cols + [
        "stat_feature_path_orig", "seq_feature_path", "raw_feature_path",
        "stat_feature_path_cl", "seq_embed_feature_path_cl", "raw_embed_feature_path_cl",
        "seq_embed_feature_path_ssl", "raw_embed_feature_path_ssl"
    ]
    # 去重并排序
    merged = merged[out_cols].drop_duplicates()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[SAVE] {split}: merged rows = {len(merged)} → {out_csv}")

def main():
    print("[INFO] Start merging splits for Experiment C ...")
    for sp in SPLITS:
        merge_one_split(sp)
    print("[DONE] All splits merged into 12_expC_outputs/uk/")

if __name__ == "__main__":
    main()
