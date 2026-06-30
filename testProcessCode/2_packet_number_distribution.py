import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

"""
脚本功能说明：
遍历给定的 input_dir；

对 activity/ 和 idle/ 下的所有 CSV 样本文件分别统计每个样本中的数据包数量；

分别绘制两个分布直方图（行为流量样本、闲时流量样本）；

显示基本统计量（如均值、中位数、最大值、最小值等）；

所得结果用于后续确定统一的序列长度。
"""


def collect_sample_packet_counts(input_dir):
    activity_counts = []
    idle_counts = []

    input_path = Path(input_dir)
    assert input_path.exists(), f"路径不存在：{input_path}"

    for device_dir in input_path.iterdir():
        if not device_dir.is_dir():
            continue

        for mode in ['activity', 'idle']:
            mode_path = device_dir / mode
            if not mode_path.exists():
                continue

            for subfolder in mode_path.iterdir():
                if not subfolder.is_dir():
                    continue

                for csv_file in subfolder.glob("*.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        pkt_count = len(df)
                        if mode == 'activity':
                            activity_counts.append(pkt_count)
                        else:
                            idle_counts.append(pkt_count)
                    except Exception as e:
                        print(f"❌ 文件读取失败: {csv_file.name}，错误: {e}")

    return activity_counts, idle_counts


def plot_distribution(counts, title, color):
    plt.figure(figsize=(10, 5))
    sns.histplot(counts, bins=40, kde=True, color=color)
    plt.title(f"{title} - 样本数据包数量分布")
    plt.xlabel("数据包数量")
    plt.ylabel("样本数量")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_statistics(name, counts):
    print(f"\n📊 {name} 样本统计:")
    print(f"总样本数: {len(counts)}")
    print(f"最小值: {min(counts)}")
    print(f"最大值: {max(counts)}")
    print(f"均值: {sum(counts) / len(counts):.2f}")
    print(f"中位数: {pd.Series(counts).median()}")
    print(f"90分位: {pd.Series(counts).quantile(0.9)}")
    print(f"95分位: {pd.Series(counts).quantile(0.95)}")
    print(f"99分位: {pd.Series(counts).quantile(0.99)}")


def main():
    input_dir = "/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/cicIoT2022"

    print(f"📁 正在分析目录: {input_dir}")
    activity_counts, idle_counts = collect_sample_packet_counts(input_dir)

    print_statistics("行为流量", activity_counts)
    print_statistics("闲时流量", idle_counts)

    plot_distribution(activity_counts, "行为流量", "skyblue")
    plot_distribution(idle_counts, "闲时流量", "salmon")


if __name__ == "__main__":
    main()
