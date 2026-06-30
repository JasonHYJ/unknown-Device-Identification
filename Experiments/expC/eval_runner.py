#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_runner.py — Experiment C unified evaluator (test + unknown + threshold sweep + PR/ROC)

WHAT THIS FILE DOES
-------------------
- 统一评估由 train_runner.py 训练好的模型（也兼容你手动放入的同命名 ckpt）
- 对 **test（已知设备）** 给出 Acc、macro-F1；可视化混淆矩阵
- 对 **unknown（未知设备）** 进行 **max-softmax 概率阈值扫**，报告覆盖率/拒识率/已知准确率等
- 额外提供 **PR/ROC 曲线绘制**（宏平均），默认开启；你可在 CONFIG 中关闭或注释
- 记录评估时长与吞吐（samples/sec），保存到 CSV

如何选择要评估的模型
--------------------
本脚本同样 **不使用命令行参数**，而是在文件顶部 CONFIG 中选择：
  CONFIG = {
    "model": "cnnseq",   # ["svm","rf","xgb","cnnseq","lstmseq","bytecnn","lp_seqssl","lp_seqcl","earlyconcat","attnfusion","ours"]
    "tasks": "T",        # 单任务：T|B|D；融合模型忽略此项
    "seed": 0,
    "cuda_visible_devices": "0",
    "plot_pr_roc": True,  # 生成 PR/ROC 曲线（宏平均 one-vs-rest）
  }
修改保存后直接运行： python eval_runner.py
"""

"""
MSP = Max Softmax Probability（最大 softmax 概率）
SPS = Samples Per Second（吞吐量，样本/秒）
详见脚本内注释。
"""

# ---------- GPU 选择（在导入 torch 之前设置） ----------
import os
CONFIG = {
    "model": "attnfusion",        # 待评估的模型["svm","rf","xgb","cnnseq","lstmseq","bytecnn","lp_seqssl","lp_seqcl","earlyconcat","attnfusion","ours"]
    "tasks": "T",             # 单任务评估的目标标签类型：T|B|D；融合：TBD（忽略此项）
    "seed": 0,
    "cuda_visible_devices": "3",
    "plot_pr_roc": True,      # 生成 PR/ROC（宏平均）；若想关闭设为 False
}
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]

import sys, json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_curve, roc_curve, auc)
import joblib

# 直接复用 train_runner 中的数据常量、数据集与模型定义，确保列名/路径一致
from train_runner import (
    DATA_ROOT, TRAIN_CSV, TEST_CSV, UNKN_CSV, LABEL_DIR, EXP_ROOT,
    StatOnlyDataset, SeqMatrixDataset, RawMatrixDataset, EmbeddingDataset, MultiModalFusionDataset,
    CNNSeq, LSTMSeq, ByteCNN, LinearProbe, EarlyConcatFusion, AttnFusion,
    TYPE2IDX, BRAND2IDX, DEVICE2IDX, ensure_dirs, reduce_128_to_64
)

# ---------------------- Utils ----------------------
def _macro_pr_roc(prob: np.ndarray, y_true: np.ndarray, save_prefix: Path):
    """
    计算多分类 one-vs-rest 的宏平均 PR/ROC，并保存图像。
    - prob: (N, C) 概率
    - y_true: (N,) 真实标签索引
    """
    classes = np.unique(y_true)
    # 若类别数过多，宏平均仍然可行；逐类绘制可很大，这里只画宏平均
    from sklearn.preprocessing import label_binarize
    Y = label_binarize(y_true, classes=classes)  # (N, C')

    # ---- Macro-PR ----
    precisions, recalls, pr_aucs = [], [], []
    for c in range(Y.shape[1]):
        p, r, _ = precision_recall_curve(Y[:,c], prob[:,c])
        precisions.append(p); recalls.append(r)
        pr_aucs.append(np.trapz(r, p))
    mean_prec = np.linspace(0,1,200)
    interp_recalls = []
    for p, r in zip(precisions, recalls):
        interp_recalls.append(np.interp(mean_prec, p[::-1], r[::-1]))
    mean_rec = np.mean(np.stack(interp_recalls, axis=0), axis=0)
    plt.figure(figsize=(5,4))
    plt.plot(mean_prec, mean_rec, label=f"Macro-PR (AUC≈{np.mean(pr_aucs):.3f})")
    plt.xlabel("Precision"); plt.ylabel("Recall"); plt.title("Macro PR (OvR)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{save_prefix}_macroPR.png"); plt.close()

    # ---- Macro-ROC ----
    fprs, tprs, rocs = [], [], []
    for c in range(Y.shape[1]):
        fpr, tpr, _ = roc_curve(Y[:,c], prob[:,c])
        fprs.append(fpr); tprs.append(tpr); rocs.append(auc(fpr,tpr))
    mean_fpr = np.linspace(0,1,200)
    interp_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(np.stack(interp_tprs, axis=0), axis=0)
    plt.figure(figsize=(5,4))
    plt.plot(mean_fpr, mean_tpr, label=f"Macro-ROC (AUC≈{np.mean(rocs):.3f})")
    plt.plot([0,1],[0,1],"--",alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Macro ROC (OvR)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{save_prefix}_macroROC.png"); plt.close()

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def evaluate_single_task(model_name: str, ckpt_path: Path, task: str, device: torch.device,
                         save_dir: Path, batch_size=128, num_workers=4, plot_pr_roc=True):
    """评估单任务模型（经典 ML + 单模态 DL + 线探针），含效率指标落盘。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Evaluating single-task model: {model_name} ({task})")
    print(f"[INFO] Using device: {device}")

    # --- 经典 ML ---
    if model_name in ["svm","rf","xgb"]:
        print(f"[INFO] Loading StatOnlyDataset for test & unknown ...")
        ds_te = StatOnlyDataset(TEST_CSV, task)
        Xte = []; yte = []
        for i in range(len(ds_te)):
            xi, yi = ds_te[i]; Xte.append(xi); yte.append(yi)
        Xte = np.vstack(Xte); yte = np.array(yte)

        ds_uk = StatOnlyDataset(UNKN_CSV, task)
        Xuk = []; yuk = []
        for i in range(len(ds_uk)):
            xi, yi = ds_uk[i]; Xuk.append(xi); yuk.append(yi)
        Xuk = np.vstack(Xuk); yuk = np.array(yuk)
        print(f"[INFO] Loaded {len(ds_te)} test samples, {len(ds_uk)} unknown samples.")

        model = joblib.load(ckpt_path)
        print(f"[INFO] Model loaded from {ckpt_path}")

        # closed-set metrics on test
        print("[INFO] Evaluating closed-set test performance ...")
        t0 = time.perf_counter()
        pred_te = model.predict(Xte)
        t1 = time.perf_counter()
        test_time = t1 - t0
        test_sps  = len(yte) / max(test_time, 1e-8)
        acc = accuracy_score(yte, pred_te)
        f1m = f1_score(yte, pred_te, average="macro")
        print(f"[RESULT] Closed-set test: acc={acc:.4f}, macro-F1={f1m:.4f} | time={test_time:.2f}s, sps={test_sps:.1f}")

        # PR/ROC（若支持概率）
        if plot_pr_roc and hasattr(model, "predict_proba"):
            print("[INFO] Generating PR/ROC curves ...")
            prob_te = model.predict_proba(Xte)
            _macro_pr_roc(prob_te, yte, save_dir/"test")
        else:
            prob_te = None

        # unknown via probability threshold (if supported)
        best_tau, best_f1, unk_time, unk_sps = 0., 0., 0., 0.
        if hasattr(model, "predict_proba"):
            print("[INFO] Sweeping probability thresholds for unknown detection ...")
            u0 = time.perf_counter()
            prob_uk = model.predict_proba(Xuk)
            u1 = time.perf_counter()
            unk_time = u1 - u0
            unk_sps  = len(ds_uk) / max(unk_time, 1e-8)
            msp_te = prob_te.max(axis=1); msp_uk = prob_uk.max(axis=1)
            taus = np.linspace(0.3,0.95,14)
            for tau in taus:
                known_mask = (msp_te >= tau)
                if known_mask.sum() == 0: continue
                macroF1 = f1_score(yte[known_mask], pred_te[known_mask], average="macro")
                if macroF1 > best_f1: best_tau, best_f1 = float(tau), float(macroF1)
            print(f"[RESULT] Best τ={best_tau:.2f}, macro-F1={best_f1:.4f} | unknown_time={unk_time:.2f}s, sps={unk_sps:.1f}")

        # 保存 metrics
        rows = [{
            "split":"test","test_acc":float(acc),"test_macro_f1":float(f1m),
            "test_time_sec":float(test_time),"test_sps":float(test_sps),
            "best_tau":float(best_tau),"best_macro_f1":float(best_f1),
            "unknown_time_sec":float(unk_time),"unknown_sps":float(unk_sps)
        }]
        pd.DataFrame(rows).to_csv(save_dir/"metrics_summary.csv", index=False)
        print(f"[SAVE] Results → {save_dir/'metrics_summary.csv'}")

        # 混淆矩阵
        cm = confusion_matrix(yte, pred_te)
        # 使用浅色背景 + 白格网线 + 蓝调色系
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            cmap=ListedColormap(sns.color_palette("Blues", as_cmap=True)(np.linspace(0.1, 1.0, 256))),
            annot=True,     # 如果不想在图里显示每个格子的数字（例如类别特别多）。设置为False
            fmt="d",
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="white"
        )
        plt.title("Confusion Matrix", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / "confmat.png", dpi=300, facecolor="white")
        plt.close()
        # 使用浅色背景 + 白格网线 + 蓝调色系
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            cmap=ListedColormap(sns.color_palette("Blues", as_cmap=True)(np.linspace(0.1, 1.0, 256))),
            annot=True,
            fmt="d",
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="white"
        )
        plt.title("Confusion Matrix", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / "confmat.png", dpi=300, facecolor="white")
        plt.close()
        print(f"[SAVE] Confusion → {save_dir/'confmat.png'}")
        print("[DONE] Evaluation complete.\n")
        return

    # --- DL & Linear probe ---
    print(f"[INFO] Loading test & unknown datasets ...")
    if model_name in ["cnnseq","lstmseq"]:
        te = SeqMatrixDataset(TEST_CSV, task); uk = SeqMatrixDataset(UNKN_CSV, task)
        dl_test = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dl_unk  = DataLoader(uk, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif model_name == "bytecnn":
        te = RawMatrixDataset(TEST_CSV, task); uk = RawMatrixDataset(UNKN_CSV, task)
        dl_test = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dl_unk  = DataLoader(uk, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else: # lp_seqssl / lp_seqcl
        mode = "ssl" if model_name=="lp_seqssl" else "cl"
        te = EmbeddingDataset(TEST_CSV, task, mode); uk = EmbeddingDataset(UNKN_CSV, task, mode)
        dl_test = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dl_unk  = DataLoader(uk, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"[INFO] Loaded {len(te)} test samples, {len(uk)} unknown samples.")

    # 推断类别数并构建网络
    if model_name == "cnnseq":
        _, y = te[0]; num_cls = len(set([y for _,y in te])); net = CNNSeq(num_cls)
    elif model_name == "lstmseq":
        _, y = te[0]; num_cls = len(set([y for _,y in te])); net = LSTMSeq(num_cls)
    elif model_name == "bytecnn":
        _, y = te[0]; num_cls = len(set([y for _,y in te])); net = ByteCNN(num_cls)
    else:
        _, y = te[0]; num_cls = len(set([y for _,y in te])); net = LinearProbe(num_cls, in_dim=64)
    print(f"[INFO] Initialized model ({model_name}) with {num_cls} classes")

    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device); net.eval()
    print(f"[INFO] Model loaded from {ckpt_path} and moved to {device}")

    # test metrics + PR/ROC
    print("[INFO] Evaluating closed-set test performance ...")
    y_true, y_pred, prob_arr = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, y in dl_test:
            logits = net(xb.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            prob_arr.append(prob)
            y_true.extend(y.tolist())
            y_pred.extend(np.argmax(prob,1).tolist())
    t1 = time.perf_counter()
    prob_arr = np.vstack(prob_arr)
    test_time = t1 - t0
    test_sps  = len(y_true) / max(test_time, 1e-8)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"[RESULT] Closed-set test: acc={acc:.4f}, macro-F1={f1m:.4f} | time={test_time:.2f}s, sps={test_sps:.1f}")

    if CONFIG["plot_pr_roc"]:
        print("[INFO] Generating macro-PR/ROC curves ...")
        _macro_pr_roc(prob_arr, np.array(y_true), save_dir/"test")

    # unknown with MSP sweep
    print("[INFO] Sweeping max-softmax thresholds for unknown detection ...")
    unk_msp = []
    u0 = time.perf_counter()
    with torch.no_grad():
        for xb, y in dl_unk:
            logits = net(xb.to(device))
            unk_msp.extend(torch.softmax(logits, dim=1).max(dim=1).values.cpu().tolist())
    u1 = time.perf_counter()
    unk_time = u1 - u0
    unk_sps  = len(unk_msp) / max(unk_time, 1e-8)
    
    taus = np.linspace(0.3,0.95,14)
    best_tau, best_f1 = 0., 0.
    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    msp = prob_arr.max(axis=1)
    for tau in taus:
        known_mask = (msp >= tau)
        if known_mask.sum()==0: continue
        macroF1 = f1_score(y_true_np[known_mask], y_pred_np[known_mask], average="macro")
        if macroF1 > best_f1:
            best_tau, best_f1 = float(tau), float(macroF1)
    print(f"[RESULT] Best τ={best_tau:.2f}, macro-F1={best_f1:.4f} | unknown_time={unk_time:.2f}s, sps={unk_sps:.1f}")

    # 保存 metrics 与混淆矩阵
    rows = [{
        "split":"test","test_acc":float(acc),"test_macro_f1":float(f1m),
        "test_time_sec":float(test_time),"test_sps":float(test_sps),
        "best_tau":float(best_tau),"best_macro_f1":float(best_f1),
        "unknown_time_sec":float(unk_time),"unknown_sps":float(unk_sps)
    }]
    pd.DataFrame(rows).to_csv(save_dir/"metrics_summary.csv", index=False)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # 使用浅色背景 + 白格网线 + 蓝调色系
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        cmap=ListedColormap(sns.color_palette("Blues", as_cmap=True)(np.linspace(0.1, 1.0, 256))),
        annot=True,
        fmt="d",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white"
    )
    plt.title("Confusion Matrix", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / "confmat.png", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVE] Results → {save_dir/'metrics_summary.csv'}; Confusion → {save_dir/'confmat.png'}")
    print("[DONE] Evaluation complete.\n")

# ---------------------- Multi-task eval ----------------------
def evaluate_multi_task(model_name: str, ckpt_path: Path, device: torch.device,
                        save_dir: Path, batch_size=128, num_workers=4):
    """评估多任务融合模型（EarlyConcat / AttentionFusion / ours-占位）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Evaluating multi-task model: {model_name}")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading MultiModalFusionDataset ...")
    te = MultiModalFusionDataset(TEST_CSV)
    uk = MultiModalFusionDataset(UNKN_CSV)
    print(f"[INFO] Loaded {len(te)} test samples, {len(uk)} unknown samples.")
    dl_test = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_unk  = DataLoader(uk, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_type = len(TYPE2IDX); num_brand = len(BRAND2IDX); num_device = len(DEVICE2IDX)
    if model_name == "earlyconcat":
        net = EarlyConcatFusion(num_type, num_brand, num_device)
    elif model_name == "attnfusion":
        net = AttnFusion(num_type, num_brand, num_device)
    else:
        net = EarlyConcatFusion(num_type, num_brand, num_device)

    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device); net.eval()
    print(f"[INFO] Model loaded from {ckpt_path} and moved to {device}")
    
    rows = []
    def _eval_dl(dloader, split_name):
        t0 = time.perf_counter()
        Yt, Yb, Yd = [], [], []
        Pt, Pb, Pd = [], [], []
        Mt, Mb = [], []
        with torch.no_grad():
            for (xs, xtup) in dloader:
                x_stat, x_seq, x_raw = xs
                yt, yb, yd = xtup
                logit_t, logit_b, logit_d = net(x_stat.to(device), x_seq.to(device), x_raw.to(device))
                Pt.append(torch.argmax(logit_t,1).cpu().numpy()); Yt.append(np.array(yt))
                Pb.append(torch.argmax(logit_b,1).cpu().numpy()); Yb.append(np.array(yb))
                Pd.append(torch.argmax(logit_d,1).cpu().numpy()); Yd.append(np.array(yd))
                Mt.append(torch.softmax(logit_t,1).max(dim=1).values.cpu().numpy())
                Mb.append(torch.softmax(logit_b,1).max(dim=1).values.cpu().numpy())
        t1 = time.perf_counter()
        split_time = t1 - t0
        # 汇总
        Yt = np.concatenate(Yt); Pt = np.concatenate(Pt); Mt = np.concatenate(Mt)
        Yb = np.concatenate(Yb); Pb = np.concatenate(Pb); Mb = np.concatenate(Mb)
        Yd = np.concatenate(Yd); Pd = np.concatenate(Pd)

        acc_t = accuracy_score(Yt, Pt); f1_t = f1_score(Yt, Pt, average="macro")
        acc_b = accuracy_score(Yb, Pb); f1_b = f1_score(Yb, Pb, average="macro")
        acc_d = accuracy_score(Yd, Pd); f1_d = f1_score(Yd, Pd, average="macro")
        rows.append({
            "split":split_name,
            "type_acc":float(acc_t), "type_f1":float(f1_t),
            "brand_acc":float(acc_b), "brand_f1":float(f1_b),
            "device_acc":float(acc_d), "device_f1":float(f1_d),
            f"{split_name}_time_sec": float(split_time),
            f"{split_name}_sps": float((len(Yt)+len(Yb)+len(Yd))/3.0/max(split_time,1e-8))
        })
        return Mt, Mb, (Yt, Pt), (Yb, Pb), (Yd, Pd), split_time

    print("[INFO] Evaluating test set across three tasks ...")
    mt_test, mb_test, (Yt, Pt), (Yb, Pb), (Yd, Pd), test_time = _eval_dl(dl_test, "test")
    print(f"[RESULT] Closed-set metrics: "
          f"type_acc={rows[-1]['type_acc']:.4f}, brand_acc={rows[-1]['brand_acc']:.4f}, device_acc={rows[-1]['device_acc']:.4f} | time={test_time:.2f}s")

    # confusion matrices (三任务各一张)
    for (M, name) in [(confusion_matrix(Yt, Pt), "type"),
                      (confusion_matrix(Yb, Pb), "brand"),
                      (confusion_matrix(Yd, Pd), "device")]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            M,
            cmap=ListedColormap(sns.color_palette("Blues", as_cmap=True)(np.linspace(0.1, 1.0, 256))),
            annot=True,
            fmt="d",
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="white"
        )
        plt.title(f"Confusion Matrix — {name}", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / f"confmat_{name}.png", dpi=300, facecolor="white")
        plt.close()
    print(f"[SAVE] Confusion matrices saved in {save_dir}")

    # unknown with MSP threshold sweep (type & brand)
    print("[INFO] Sweeping MSP thresholds for type/brand unknown detection ...")
    mt_unk, mb_unk, *_ = _eval_dl(dl_unk, "unknown")
    taus = np.linspace(0.3,0.95,14)
    def _sweep(msp_known, msp_unk):
        best = {"tau":None,"known_coverage":0.0,"unknown_reject":0.0,"score":-1}
        for tau in taus:
            known_mask = (msp_known >= tau)
            cov = float(np.mean(known_mask))
            rej = float(np.mean(msp_unk < tau))
            score = cov  # 若想兼顾 rej，可改为 score = 0.5*cov + 0.5*rej
            if score > best["score"]:
                best = {"tau":float(tau), "known_coverage":cov, "unknown_reject":rej, "score":score}
        return best
    best_t = _sweep(np.array(mt_test), np.array(mt_unk))
    best_b = _sweep(np.array(mb_test), np.array(mb_unk))
    rows.append({"split":"unknown@best_tau",
                 "type_tau":best_t["tau"], "type_cov":best_t["known_coverage"], "type_reject":best_t["unknown_reject"],
                 "brand_tau":best_b["tau"], "brand_cov":best_b["known_coverage"], "brand_reject":best_b["unknown_reject"]})

    pd.DataFrame(rows).to_csv(save_dir/"metrics_summary.csv", index=False)
    print(f"[SAVE] Metrics summary → {save_dir/'metrics_summary.csv'}")
    print("[DONE] Evaluation complete.\n")

def main():
    print(f"[INFO] Visible GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES')} (mapped as cuda:0)")
    print(f"[INFO] Config: {CONFIG}")
    group = "baselines" if CONFIG["model"] in ["svm","rf","xgb","cnnseq","lstmseq","bytecnn","lp_seqssl","lp_seqcl"] else "fusion"
    model_dir = Path(f"{EXP_ROOT}/{group}/models")
    eval_dir  = Path(f"{EXP_ROOT}/{group}/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 依命名规则找到模型文件（与 train_runner.py 完全一致）
    if CONFIG["model"] in ["svm","rf","xgb"]:
        name = f"{CONFIG['model']}_statCL__{CONFIG['tasks']}_seed{CONFIG['seed']}.pkl"
    elif CONFIG["model"] in ["cnnseq","lstmseq","bytecnn","lp_seqssl","lp_seqcl"]:
        name = f"{CONFIG['model']}__{CONFIG['tasks']}_seed{CONFIG['seed']}.pt"
    else:
        name = f"{CONFIG['model']}__TBD_seed{CONFIG['seed']}.pt"
    ckpt = model_dir/name
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    save_dir = eval_dir/f"{ckpt.stem}"
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint: {ckpt}")

    if CONFIG["model"] in ["earlyconcat","attnfusion","ours"]:
        evaluate_multi_task(CONFIG["model"], Path(ckpt), device, save_dir)
    else:
        evaluate_single_task(CONFIG["model"], Path(ckpt), CONFIG["tasks"], device, save_dir, plot_pr_roc=CONFIG["plot_pr_roc"])

    print(f"[DONE] All results saved under: {save_dir}\n")

if __name__ == "__main__":
    main()
