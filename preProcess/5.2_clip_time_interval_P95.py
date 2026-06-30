import os
import pandas as pd
import numpy as np

"""
功能描述：
- 遍历 IoT 数据集中每个行为/闲时文件夹
- 设置每个样本的第一个 time_interval 为 0
- 统计该文件夹中所有样本的 time_interval 的 95% 分位数
- 作为该文件夹统一的剪裁阈值，应用于其所有样本
- 将处理结果保存到新的输出目录中，保持目录结构不变
- 最后打印 clip 阈值记录及样本数量统计
"""


def collect_time_intervals(folder_path):
    """
    从一个行为/闲时文件夹中收集所有样本的 time_interval（首包设为0）
    """
    all_intervals = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path, usecols=["time_interval"])
                if len(df) > 0:
                    df.loc[df.index[0], "time_interval"] = 0  # 修正首包时间间隔
                    all_intervals.extend(df["time_interval"].dropna().tolist())
            except Exception as e:
                print(f"[错误] 无法读取 {file_path}：{e}")
    return all_intervals


def clip_and_save_csv(file_path, output_path, clip_threshold):
    """
    剪裁 CSV 文件中的 time_interval 并保存到输出路径
    """
    try:
        df = pd.read_csv(file_path)
        if "time_interval" not in df.columns or len(df) == 0:
            print(f"[跳过] 缺少 time_interval 列: {file_path}")
            return False

        original_max = df["time_interval"].max()
        df.loc[df.index[0], "time_interval"] = 0  # 修正首包时间间隔
        df["time_interval"] = df["time_interval"].clip(upper=clip_threshold)
        new_max = df["time_interval"].max()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"[完成] {output_path} | 裁剪: {clip_threshold:.3f}s | 原最大: {original_max:.3f}, 剪后: {new_max:.3f}")
        return True

    except Exception as e:
        print(f"[错误] 无法处理 {file_path}：{e}")
        return False


def main():
    # ==== 修改为你的实际路径 ====
    input_root = "/home/hyj/unknownDeviceIdentification/dataset/test/csv_filter"
    output_root = "/home/hyj/unknownDeviceIdentification/dataset/test/csv_clip_time_interval_P95"

    # === 全局统计 ===
    total_count = 0
    activity_count = 0
    idle_count = 0
    clip_threshold_record = {}

    # === 遍历设备目录 ===
    for device_name in os.listdir(input_root):
        device_path = os.path.join(input_root, device_name)
        if not os.path.isdir(device_path):
            continue

        for mode in ['activity', 'idle']:
            mode_path = os.path.join(device_path, mode)
            if not os.path.isdir(mode_path):
                continue

            # 每个行为文件夹或闲时文件夹
            for subfolder_name in os.listdir(mode_path):
                subfolder_path = os.path.join(mode_path, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    continue

                print(f"\n📂 正在处理文件夹: {subfolder_path}")

                # 1. 收集 time_interval 统计值
                all_intervals = collect_time_intervals(subfolder_path)
                if len(all_intervals) == 0:
                    print(f"[跳过] 没有有效 time_interval：{subfolder_path}")
                    continue

                clip_threshold = np.percentile(all_intervals, 95)
                clip_threshold_record[subfolder_path] = clip_threshold

                # 2. 遍历样本进行剪裁并保存
                folder_sample_count = 0
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        input_file_path = os.path.join(subfolder_path, file)

                        relative_path = os.path.relpath(input_file_path, input_root)
                        output_file_path = os.path.join(output_root, relative_path)

                        success = clip_and_save_csv(input_file_path, output_file_path, clip_threshold)
                        if success:
                            folder_sample_count += 1
                            total_count += 1
                            if mode == 'activity':
                                activity_count += 1
                            elif mode == 'idle':
                                idle_count += 1
                print(f"[完成] {subfolder_path} | 样本数: {folder_sample_count} | 剪裁阈值: {clip_threshold:.3f}s")

        # === 汇总结果 ===
        print("\n📊 文件夹剪裁阈值记录（95%分位数）：")
        for folder, thresh in clip_threshold_record.items():
            print(f"- {folder}: {thresh:.3f}s")

        print("\n✅ 处理完成")
        print(f"总样本数: {total_count}")
        print(f"行为样本数 (activity): {activity_count}")
        print(f"闲时样本数 (idle): {idle_count}")


if __name__ == "__main__":
    main()
