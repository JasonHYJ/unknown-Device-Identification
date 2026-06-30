import os
import pandas as pd

def count_all_datasets(base_dirs):
    all_device_stats = []
    print("开始统计所有数据集的设备样本情况...\n")

    for base_dir in base_dirs:
        dataset_name = os.path.basename(base_dir.rstrip('/'))
        print(f"\n📁 数据集：{dataset_name}")
        dataset_behavior_total = 0
        dataset_idle_total = 0

        for device_name in os.listdir(base_dir):
            device_path = os.path.join(base_dir, device_name)
            if not os.path.isdir(device_path):
                continue

            activity_path = os.path.join(device_path, 'activity')
            idle_path = os.path.join(device_path, 'idle')

            behavior_total = 0
            idle_total = 0
            behavior_details = {}
            idle_details = {}

            # 统计行为样本
            if os.path.exists(activity_path):
                for behavior_name in os.listdir(activity_path):
                    behavior_folder = os.path.join(activity_path, behavior_name)
                    if os.path.isdir(behavior_folder):
                        files = [f for f in os.listdir(behavior_folder) if f.endswith('.csv')]
                        behavior_details[behavior_name] = len(files)
                        behavior_total += len(files)

            # 统计闲时样本
            if os.path.exists(idle_path):
                for idle_name in os.listdir(idle_path):
                    idle_folder = os.path.join(idle_path, idle_name)
                    if os.path.isdir(idle_folder):
                        files = [f for f in os.listdir(idle_folder) if f.endswith('.csv')]
                        idle_details[idle_name] = len(files)
                        idle_total += len(files)

            dataset_behavior_total += behavior_total
            dataset_idle_total += idle_total

            print(f"  📦 设备：{device_name:25s} | 行为样本: {behavior_total:4d} | 闲时样本: {idle_total:5d} | 行为种类: {len(behavior_details):2d} | 闲时来源: {len(idle_details):2d}")

            all_device_stats.append({
                'dataset': dataset_name,
                'device': device_name,
                'behavior_total': behavior_total,
                'idle_total': idle_total,
                'behavior_details': behavior_details,
                'idle_details': idle_details
            })

        print(f"✅ 总结 - 数据集 {dataset_name}: 行为样本总数 = {dataset_behavior_total}, 闲时样本总数 = {dataset_idle_total}")

    return pd.DataFrame(all_device_stats)


# 修改为你本地的绝对路径
dataset_paths = [
    "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature/cicIoT2022",
    "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature/uk",
    "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature/us"
]

df = count_all_datasets(dataset_paths)
df.to_csv("/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/all_dataset_stats.csv", index=False)
print("\n✅ 统计完成，已保存为 all_dataset_stats.csv")