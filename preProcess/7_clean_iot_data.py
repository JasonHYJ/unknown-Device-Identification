import os
import shutil
from collections import defaultdict

"""
目前正在使用的脚本

脚本功能说明：
本脚本用于清洗 IoT 流量特征样本数据。
输入为三个特征目录（统计特征、序列特征、原始字节特征），每个目录结构一致，包含多个数据集及设备。
清洗规则：
    - 保留所有设备的闲时流量样本（idle/），但仅当闲时样本总数 ≥ 300
    - 仅保留行为样本总数 ≥ 10 的设备的行为文件夹（activity/）

清洗后的数据将分别保存至新建的输出目录中（保持原目录结构不变），用于后续阶段训练。
这个代码是通过stat特征目录的统计情况，对三个特征的目录进行处理
"""

# === 1. 配置输入输出路径 ===
input_dirs = {
    'stat': '/home/hyj/unknownDeviceIdentification/dataset/6_extracted_features/6_csv_statistical_feature',
    'seq': '/home/hyj/unknownDeviceIdentification/dataset/6_extracted_features/6_csv_sequence_feature_matrix',
    'raw': '/home/hyj/unknownDeviceIdentification/dataset/6_extracted_features/6_csv_rawByte_feature_matrix'
}
output_dirs = {
    'stat': '/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature',
    'seq': '/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_sequence_feature_matrix',
    'raw': '/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_rawByte_feature_matrix'
}
behavior_threshold = 10  # 每台设备所有行为样本总数 ≥ 10 时保留其 activity/
idle_threshold = 300    # 每台设备所有闲时样本总数 ≥ 300 时保留其 idle/

# === 2. 收集每台设备的行为样本和闲时样本数量 ===
def collect_behavior_counts(stat_dir):
    behavior_counts = defaultdict(int)
    idle_counts = defaultdict(int)
    print(f"正在检查目录: {stat_dir}")
    if not os.path.exists(stat_dir):
        print(f"错误：输入目录 {stat_dir} 不存在！")
        return behavior_counts, idle_counts

    for dataset in os.listdir(stat_dir):
        dataset_path = os.path.join(stat_dir, dataset)
        print(f"检查数据集: {dataset_path}")
        if not os.path.isdir(dataset_path):
            print(f"跳过非目录: {dataset_path}")
            continue
        for device in os.listdir(dataset_path):
            # Check activity folder
            act_dir = os.path.join(dataset_path, device, 'activity')
            print(f"检查设备 {device} 的 activity 路径: {act_dir}, 是否存在: {os.path.isdir(act_dir)}")
            if os.path.isdir(act_dir):
                for behavior in os.listdir(act_dir):
                    behavior_path = os.path.join(act_dir, behavior)
                    if os.path.isdir(behavior_path):
                        files = os.listdir(behavior_path)
                        csv_files = [f for f in files if f.lower().endswith('.csv')]
                        count = len(csv_files)
                        behavior_counts[(dataset, device)] += count
                        print(f"设备 {device} 行为 {behavior} 包含 CSV 文件数量: {count}")
                    else:
                        print(f"跳过非目录: {behavior_path}")
            else:
                print(f"设备 {device} 的 activity 文件夹不存在")
            
            # Check idle folder
            idle_dir = os.path.join(dataset_path, device, 'idle')
            print(f"检查设备 {device} 的 idle 路径: {idle_dir}, 是否存在: {os.path.isdir(idle_dir)}")
            if os.path.isdir(idle_dir):
                for idle in os.listdir(idle_dir):
                    idle_path = os.path.join(idle_dir, idle)
                    if os.path.isdir(idle_path):
                        files = os.listdir(idle_path)
                        csv_files = [f for f in files if f.lower().endswith('.csv')]
                        count = len(csv_files)
                        idle_counts[(dataset, device)] += count
                        print(f"设备 {device} 的 idle 子文件夹 {idle} 包含 CSV 文件数量: {count}")
                    else:
                        print(f"跳过非目录: {idle_path}")
            else:
                print(f"设备 {device} 的 idle 文件夹不存在")
                
    return behavior_counts, idle_counts

# === 3. 将符合条件的样本拷贝到输出目录 ===
def copy_valid_structure(input_dir, output_dir, valid_behavior_devices, valid_idle_devices):
    for dataset in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for device in os.listdir(dataset_path):
            device_path = os.path.join(dataset_path, device)
            if not os.path.isdir(device_path):
                continue
            out_device_path = os.path.join(output_dir, dataset, device)
            os.makedirs(out_device_path, exist_ok=True)  # 确保输出目录存在

            # 仅当闲时样本足够时保留 idle 文件夹
            if (dataset, device) in valid_idle_devices:
                idle_dir = os.path.join(device_path, 'idle')
                if os.path.isdir(idle_dir):
                    shutil.copytree(idle_dir, os.path.join(out_device_path, 'idle'), dirs_exist_ok=True)
                    print(f"已拷贝设备 {device} 的 idle 文件夹到: {os.path.join(out_device_path, 'idle')}")

            # 如果行为样本足够，则保留 activity 文件夹
            if (dataset, device) in valid_behavior_devices:
                act_dir = os.path.join(device_path, 'activity')
                if os.path.isdir(act_dir):
                    shutil.copytree(act_dir, os.path.join(out_device_path, 'activity'), dirs_exist_ok=True)
                    print(f"已拷贝设备 {device} 的 activity 文件夹到: {os.path.join(out_device_path, 'activity')}")

# === 4. 主执行函数 ===
def main():
    print("🔍 正在统计每台设备的行为样本和闲时样本总数...")
    behavior_counts, idle_counts = collect_behavior_counts(input_dirs['stat'])

    # 设备满足样本总数 ≥ 阈值 的集合
    valid_behavior_devices = {k for k, v in behavior_counts.items() if v >= behavior_threshold}
    valid_idle_devices = {k for k, v in idle_counts.items() if v >= idle_threshold}
    print(f"✅ 满足行为样本总数 ≥ {behavior_threshold} 的设备数量：{len(valid_behavior_devices)}")
    print(f"✅ 满足闲时样本总数 ≥ {idle_threshold} 的设备数量：{len(valid_idle_devices)}")
    # 打印每个设备的样本数量，方便验证
    print("\n行为样本统计：")
    for (dataset, device), count in behavior_counts.items():
        print(f"数据集 {dataset}, 设备 {device}: {count} 个行为样本")
    print("\n闲时样本统计：")
    for (dataset, device), count in idle_counts.items():
        print(f"数据集 {dataset}, 设备 {device}: {count} 个闲时样本")

    # 对所有输入目录执行清洗并保存输出
    for key in input_dirs:
        print(f"\n📁 正在处理特征目录：{input_dirs[key]}")
        copy_valid_structure(input_dirs[key], output_dirs[key], valid_behavior_devices, valid_idle_devices)
        print(f"✅ 输出保存至：{output_dirs[key]}")

if __name__ == "__main__":
    main()