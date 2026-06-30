import os

"""
作用：统计输入目录结构中 .csv 文件的数量，包括：
1. 所有 csv 总数
2. 所有 activity 文件夹下的行为样本数
3. 所有 idle 文件夹下的闲时样本数

示例输出格式（假设统计结果）：
总共的 CSV 文件数：4350
所有 activity 文件夹中的行为样本 CSV 数量：3050
所有 idle 文件夹中的闲时样本 CSV 数量：1300
"""


def count_csv_files(root_dir):
    total_csv = 0
    activity_csv = 0
    idle_csv = 0

    # 遍历所有设备文件夹
    for device_name in os.listdir(root_dir):
        device_path = os.path.join(root_dir, device_name)
        if not os.path.isdir(device_path):
            continue  # 跳过非文件夹

        # 分别处理 activity 和 idle 文件夹
        for mode in ['activity', 'idle']:
            mode_path = os.path.join(device_path, mode)
            if not os.path.isdir(mode_path):
                continue

            # 遍历 mode 文件夹下的所有子文件夹（行为/闲时）
            for subfolder in os.listdir(mode_path):
                subfolder_path = os.path.join(mode_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                # 统计该行为/闲时子文件夹下的所有 csv 文件
                csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
                count = len(csv_files)
                total_csv += count
                if mode == 'activity':
                    activity_csv += count
                elif mode == 'idle':
                    idle_csv += count

    print(f"总共的 CSV 文件数：{total_csv}")
    print(f"所有 activity 文件夹中的行为样本 CSV 数量：{activity_csv}")
    print(f"所有 idle 文件夹中的闲时样本 CSV 数量：{idle_csv}")


# 替换为你的根目录路径
root_dir = '/home/hyj/unknownDeviceIdentification/dataset/2_csv/cicIoT2022'
count_csv_files(root_dir)
