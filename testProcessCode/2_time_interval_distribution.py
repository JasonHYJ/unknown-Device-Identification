import os
import pandas as pd
import matplotlib.pyplot as plt

# 修改为你的根目录路径
root_dir = "/home/hyj/unknownDeviceIdentification/dataset/test/csv_clip_time_interval_log1p"

# 用于收集所有 time_interval 值
all_time_intervals = []

# 遍历设备文件夹
for device_folder in os.listdir(root_dir):
    device_path = os.path.join(root_dir, device_folder)
    if not os.path.isdir(device_path):
        continue

    for mode in ['activity', 'idle']:
        mode_path = os.path.join(device_path, mode)
        if not os.path.isdir(mode_path):
            continue

        # 遍历行为/闲时文件夹
        for subfolder in os.listdir(mode_path):
            subfolder_path = os.path.join(mode_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # 遍历样本 CSV 文件
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(subfolder_path, file_name)
                    try:
                        df = pd.read_csv(file_path, usecols=["time_interval"])
                        all_time_intervals.extend(df["time_interval"].dropna().tolist())
                    except Exception as e:
                        print(f"[!] Failed to read {file_path}: {e}")

# 构建DataFrame
interval_df = pd.DataFrame(all_time_intervals, columns=["time_interval"])

# 打印统计信息
stats = interval_df["time_interval"].describe(percentiles=[0.90, 0.95, 0.98, 0.99])
print("Time Interval Statistics:\n", stats)

# 绘图：分布直方图，clip上限为5秒
plt.figure(figsize=(10, 6))
interval_df["time_interval"].clip(upper=5).hist(bins=100)
plt.title("Distribution of time_interval (Clipped at 5s)")
plt.xlabel("Time Interval (seconds)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
