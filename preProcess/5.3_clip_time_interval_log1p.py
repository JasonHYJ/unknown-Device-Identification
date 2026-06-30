import os
import pandas as pd
import numpy as np

"""
（当前时间间隔处理方式！！！应该算是最佳的处理包间时间间隔数值偏大的方式）
功能描述：
- 遍历 IoT 数据集中的所有行为/闲时文件夹
- 对每个样本的 time_interval 列：
    1. 设置首个包时间间隔为 0
    2. 应用 log1p() 变换（即 log(1 + x)）
- 将变换后的 CSV 文件保存到新的输出目录中，保持原始目录结构不变
- 输出每个文件夹处理状态以及全局样本计数
"""


def log_transform_and_save_csv(input_path, output_path):
    try:
        df = pd.read_csv(input_path)

        if "time_interval" not in df.columns or df.empty:
            print(f"[跳过] 缺少 time_interval 列或空文件: {input_path}")
            return False

        original_max = df["time_interval"].max()
        df.loc[df.index[0], "time_interval"] = 0  # 修正首包时间间隔

        # 应用 log1p 变换
        df["time_interval"] = np.log1p(df["time_interval"])
        new_max = df["time_interval"].max()

        # 保存文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"[完成] {output_path} | 原最大: {original_max:.3f}, log1p后最大: {new_max:.3f}")
        return True

    except Exception as e:
        print(f"[错误] 无法处理 {input_path}：{e}")
        return False


def main():
    input_root = "/home/hyj/unknownDeviceIdentification/dataset/4_csv_withProtocolFilter/cicIoT2022"
    output_root = "/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/cicIoT2022"

    total_count = 0
    activity_count = 0
    idle_count = 0

    for device_name in os.listdir(input_root):
        device_path = os.path.join(input_root, device_name)
        if not os.path.isdir(device_path):
            continue

        for mode in ['activity', 'idle']:
            mode_path = os.path.join(device_path, mode)
            if not os.path.isdir(mode_path):
                continue

            for subfolder_name in os.listdir(mode_path):
                subfolder_path = os.path.join(mode_path, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    continue

                print(f"\n📂 正在处理文件夹: {subfolder_path}")
                folder_sample_count = 0

                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        input_file_path = os.path.join(subfolder_path, file)
                        relative_path = os.path.relpath(input_file_path, input_root)
                        output_file_path = os.path.join(output_root, relative_path)

                        success = log_transform_and_save_csv(input_file_path, output_file_path)
                        if success:
                            folder_sample_count += 1
                            total_count += 1
                            if mode == 'activity':
                                activity_count += 1
                            elif mode == 'idle':
                                idle_count += 1

                print(f"[完成] 文件夹处理完毕: {folder_sample_count} 个样本")

    print("\n✅ 全部处理完成")
    print(f"总样本数: {total_count}")
    print(f"行为样本数 (activity): {activity_count}")
    print(f"闲时样本数 (idle): {idle_count}")


if __name__ == "__main__":
    main()
