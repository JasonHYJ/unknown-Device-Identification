import os

"""
📌 功能说明：
本脚本用于重命名 IoT 行为流量数据集中所有设备的行为 pcap 文件，使其命名规范、统一、简洁。

适配目录结构如下：

input_root/
├── DeviceA/
│   └── activity/
│       ├── BEHAVIOR_X/
│       │   ├── raw_file1.pcap
│       │   └── ...
│       └── BEHAVIOR_Y/
│           └── ...

🔁 重命名规则：
将每个行为 pcap 文件改名为：
  DeviceName__behavior_folder_name_lowercase__00001.pcap

示例：
  原始：echodot1LANVOLUMEOFF_1.pcap
  目标：AmazonAlexaEchoDot1__lan_volume_off__00001.pcap

📦 操作说明：
- 保留原目录结构，仅重命名文件；
- 行为名称转为小写；
- 编号从 00001 开始，格式统一为 5 位数字。
"""


def rename_activity_files(input_root):
    """
    遍历 input_root 下每个设备的 activity 目录中的所有行为文件夹，
    对每个 .pcap 文件进行统一重命名。
    """
    renamed_count = 0  # 统计总重命名数

    for device_name in os.listdir(input_root):
        device_path = os.path.join(input_root, device_name)
        activity_path = os.path.join(device_path, 'activity')

        if not os.path.isdir(activity_path):
            continue  # 如果没有 activity 文件夹则跳过

        for behavior_folder in os.listdir(activity_path):
            behavior_path = os.path.join(activity_path, behavior_folder)
            if not os.path.isdir(behavior_path):
                continue  # 确保是行为文件夹

            behavior_name = behavior_folder.lower()  # 行为名统一转小写

            # 获取所有 .pcap 文件并排序，确保编号稳定
            pcap_files = [f for f in os.listdir(behavior_path) if f.endswith(".pcap")]
            pcap_files.sort()

            for idx, old_filename in enumerate(pcap_files, 1):
                ext = os.path.splitext(old_filename)[1]  # 获取扩展名（.pcap）
                new_filename = f"{device_name}__{behavior_name}__{idx:05d}{ext}"

                old_path = os.path.join(behavior_path, old_filename)
                new_path = os.path.join(behavior_path, new_filename)

                os.rename(old_path, new_path)
                renamed_count += 1

    print(f"✅ 行为文件重命名完成，共处理 {renamed_count} 个 pcap 文件。")


# 示例调用：请替换为你的实际路径
rename_activity_files("/home/hyj/unknownDeviceIdentification/dataset/1_splited_pcap/us")
