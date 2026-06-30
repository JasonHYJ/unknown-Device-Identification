# Re-import required modules due to kernel reset
import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 设置输入目录
input_dir = Path("/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/cicIoT2022")

# 初始化统计数据
stats = {
    "activity": {
        "sample_counts": 0,
        "packet_counts": [],
        "raw_bytes_lengths": []
    },
    "idle": {
        "sample_counts": 0,
        "packet_counts": [],
        "raw_bytes_lengths": []
    }
}

# 遍历目录
for device_dir in input_dir.iterdir():
    if not device_dir.is_dir():
        continue
    for mode in ["activity", "idle"]:
        mode_dir = device_dir / mode
        if not mode_dir.exists():
            continue
        for sub_dir in mode_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for csv_file in sub_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    stats[mode]["sample_counts"] += 1
                    stats[mode]["packet_counts"].append(len(df))
                    if 'raw_bytes' in df.columns:
                        raw_lengths = df['raw_bytes'].dropna().astype(str).apply(lambda x: len(x))
                        stats[mode]["raw_bytes_lengths"].extend(raw_lengths.tolist())
                except Exception as e:
                    print(f"读取失败: {csv_file}, 原因: {e}")

# 统计信息展示函数
def display_stats(mode, data):
    packet_array = np.array(data["packet_counts"])
    raw_byte_array = np.array(data["raw_bytes_lengths"])
    print(f"\n📊 {mode.upper()} 流量样本统计:")
    print(f"总样本数: {data['sample_counts']}")
    print(f"数据包数 - 最小: {packet_array.min()}, 最大: {packet_array.max()}, 均值: {packet_array.mean():.2f}, 中位数: {np.median(packet_array)}")
    for q in [90, 95, 99]:
        print(f"数据包数 - {q}分位: {np.percentile(packet_array, q)}")
    if len(raw_byte_array) > 0:
        print(f"原始字节长度 - 最小: {raw_byte_array.min()}, 最大: {raw_byte_array.max()}, 均值: {raw_byte_array.mean():.2f}, 中位数: {np.median(raw_byte_array)}")
        for q in [90, 95, 99]:
            print(f"原始字节长度 - {q}分位: {np.percentile(raw_byte_array, q)}")

# 显示统计结果
display_stats("activity", stats["activity"])
display_stats("idle", stats["idle"])
