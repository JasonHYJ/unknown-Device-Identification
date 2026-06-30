import os
import shutil
from collections import defaultdict

"""
目前不用的脚本

脚本功能说明：
本脚本用于清洗 IoT 流量特征样本数据。
输入为三个特征目录（统计特征、序列特征、原始字节特征），每个目录结构一致，包含多个数据集及设备。
清洗规则：
    - 保留所有设备的闲时流量样本（idle/）
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

# === 2. 收集每台设备的行为样本数量 ===
def collect_behavior_counts(stat_dir):
    behavior_counts = defaultdict(int)
    for dataset in os.listdir(stat_dir):
        dataset_path = os.path.join(stat_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for device in os.listdir(dataset_path):
            act_dir = os.path.join(dataset_path, device, 'activity')
            if not os.path.isdir(act_dir):
                continue
            for behavior in os.listdir(act_dir):
                behavior_path = os.path.join(act_dir, behavior)
                if os.path.isdir(behavior_path):
                    # 累加该行为文件夹下的CSV样本数量
                    count = len([f for f in os.listdir(behavior_path) if f.endswith('.csv')])
                    behavior_counts[(dataset, device)] += count
    return behavior_counts

# === 3. 将符合条件的样本拷贝到输出目录 ===
def copy_valid_structure(input_dir, output_dir, valid_devices):
    for dataset in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for device in os.listdir(dataset_path):
            device_path = os.path.join(dataset_path, device)
            if not os.path.isdir(device_path):
                continue
            out_device_path = os.path.join(output_dir, dataset, device)

            # 保留所有 idle 文件夹
            idle_dir = os.path.join(device_path, 'idle')
            if os.path.isdir(idle_dir):
                shutil.copytree(idle_dir, os.path.join(out_device_path, 'idle'), dirs_exist_ok=True)

            # 如果行为样本足够，则保留 activity 文件夹
            if (dataset, device) in valid_devices:
                act_dir = os.path.join(device_path, 'activity')
                if os.path.isdir(act_dir):
                    shutil.copytree(act_dir, os.path.join(out_device_path, 'activity'), dirs_exist_ok=True)

# === 4. 主执行函数 ===
def main():
    print("🔍 正在统计每台设备的行为样本总数...")
    behavior_counts = collect_behavior_counts(input_dirs['stat'])

    # 设备满足样本总数 ≥ 阈值 的集合
    valid_devices = {k for k, v in behavior_counts.items() if v >= behavior_threshold}
    print(f"✅ 满足行为样本总数 ≥ {behavior_threshold} 的设备数量：{len(valid_devices)}")

    # 对所有输入目录执行清洗并保存输出
    for key in input_dirs:
        print(f"\n📁 正在处理特征目录：{input_dirs[key]}")
        copy_valid_structure(input_dirs[key], output_dirs[key], valid_devices)
        print(f"✅ 输出保存至：{output_dirs[key]}")

if __name__ == "__main__":
    main()
