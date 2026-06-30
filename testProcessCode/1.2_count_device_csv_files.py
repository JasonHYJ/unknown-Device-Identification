import os

"""
说明：更加细粒度的统计csv文件数目：
1. 每个设备文件夹中的 CSV 文件数量会被统计，并输出。
2. 每个设备的 activity 文件夹中的行为文件夹以及其每个文件夹中的 CSV 文件数量也会被统计。
3. 每个设备的 idle 文件夹中的闲时文件夹及其每个文件夹中的 CSV 文件数量同样会被统计。
4. 最后会输出所有设备的总计信息。

示例输出格式（假设统计结果）：
统计设备: allure-speaker
  行为文件夹 play1 CSV 文件数：100
  行为文件夹 play2 CSV 文件数：120
  闲时文件夹 idle1 CSV 文件数：150
  闲时文件夹 idle2 CSV 文件数：80
设备 allure-speaker 总共 CSV 文件数：450
设备 allure-speaker activity 文件夹中的 CSV 文件数：220
设备 allure-speaker idle 文件夹中的 CSV 文件数：230

统计设备: appletv
  行为文件夹 play1 CSV 文件数：200
  行为文件夹 play2 CSV 文件数：150
  闲时文件夹 idle1 CSV 文件数：180
  闲时文件夹 idle2 CSV 文件数：70
设备 appletv 总共 CSV 文件数：600
设备 appletv activity 文件夹中的 CSV 文件数：350
设备 appletv idle 文件夹中的 CSV 文件数：250

总共的 CSV 文件数：1050
所有 activity 文件夹中的行为样本 CSV 数量：570
所有 idle 文件夹中的闲时样本 CSV 数量：480
"""

import os


def count_device_csv_files(root_dir):
    total_csv = 0
    activity_csv = 0
    idle_csv = 0

    # 遍历所有设备文件夹
    for device_name in os.listdir(root_dir):
        device_path = os.path.join(root_dir, device_name)
        if not os.path.isdir(device_path):
            continue  # 跳过非文件夹

        device_total_csv = 0
        device_activity_csv = 0
        device_idle_csv = 0

        print(f"\n统计设备: {device_name}")

        # 统计该设备的总 csv 文件数
        for mode in ['activity', 'idle']:
            mode_path = os.path.join(device_path, mode)
            if not os.path.isdir(mode_path):
                continue

            # 分别处理 activity 和 idle 文件夹
            for subfolder in os.listdir(mode_path):
                subfolder_path = os.path.join(mode_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                # 统计该行为/闲时子文件夹下的所有 csv 文件
                csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npz') or f.endswith('.csv')]
                count = len(csv_files)
                device_total_csv += count

                if mode == 'activity':
                    device_activity_csv += count
                    print(f"  行为文件夹 {subfolder} CSV 文件数：{count}")
                elif mode == 'idle':
                    device_idle_csv += count
                    print(f"  闲时文件夹 {subfolder} CSV 文件数：{count}")

        total_csv += device_total_csv
        activity_csv += device_activity_csv
        idle_csv += device_idle_csv

        print(f"设备 {device_name} 总共 CSV 文件数：{device_total_csv}")
        print(f"设备 {device_name} activity 文件夹中的 CSV 文件数：{device_activity_csv}")
        print(f"设备 {device_name} idle 文件夹中的 CSV 文件数：{device_idle_csv}")

    print(f"\n总共的 CSV 文件数：{total_csv}")
    print(f"所有 activity 文件夹中的行为样本 CSV 数量：{activity_csv}")
    print(f"所有 idle 文件夹中的闲时样本 CSV 数量：{idle_csv}")


# 替换为你的根目录路径
root_dir = '/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings/train/cicIoT2022'
count_device_csv_files(root_dir)
