import os

"""
📌 功能说明：
本脚本用于批量重命名 IoT 流量划分样本文件，适配不同数据集（如 cicIoT2022 和 monIoTr）。
在这些数据集中，划分后的文件名较长或结构冗余，例如：
  原始：DeviceName__idle__DeviceName_YYYY_MM_DD_Idle__00001.pcap
期望：
  新名：DeviceName__YYYY_MM_DD_Idle__00001.pcap

✅ 关键步骤如下：
1. 遍历所有设备文件夹；
2. 定位每个设备的 idle 流量子目录；
3. 遍历每个原始 pcap 日期目录下的样本；
4. 对符合特定命名格式的文件进行重命名；
5. 根据不同数据集来源（如 cicIoT2022 或 monIoTr），决定是否剥除冗余设备名前缀；
6. 生成简洁规范的文件名格式，保存回原目录。
"""


def rename_idle_samples(output_root):
    """
    对 output_root 下所有设备的 idle 子目录中的样本文件进行重命名：
    原始文件名格式为：
        DeviceName__idle__DeviceName_YYYY_MM_DD_Idle__00001.pcap
    修改为：
        DeviceName__YYYY_MM_DD_Idle__00001.pcap
    """
    renamed_count = 0  # 记录重命名的文件数

    for device_name in os.listdir(output_root):
        device_path = os.path.join(output_root, device_name)
        idle_path = os.path.join(device_path, 'idle')

        if not os.path.isdir(idle_path):
            continue  # 如果 idle 路径不存在，跳过该设备

        for subdir in os.listdir(idle_path):
            subdir_path = os.path.join(idle_path, subdir)
            if not os.path.isdir(subdir_path):
                continue  # 只处理文件夹（日期目录）

            for filename in os.listdir(subdir_path):
                if not filename.endswith(".pcap"):
                    continue  # 只处理 .pcap 文件

                parts = filename.split("__")
                if len(parts) != 4:
                    continue  # 文件名结构不符合要求，跳过

                device = parts[0]

                # ⚠️ 两种数据集处理逻辑（只保留一条生效）：
                # ① 适用于 cicIoT2022：去掉冗余的设备名前缀
                # date_part = parts[2].replace(device + "_", "")

                # ② 适用于 monIoTr：直接保留原始字段
                date_part = parts[2]

                index_part = parts[3]

                # 构建新的文件名
                new_filename = f"{device}__{date_part}__{index_part}"

                # 拼接文件路径
                old_path = os.path.join(subdir_path, filename)
                new_path = os.path.join(subdir_path, new_filename)

                # 执行重命名
                os.rename(old_path, new_path)
                renamed_count += 1

    print(f"✅ 重命名完成，共处理 {renamed_count} 个样本文件。")


# 示例调用：修改为你的输出路径
rename_idle_samples("/home/hyj/unknownDeviceIdentification/dataset/1_splited_pcap/us")
