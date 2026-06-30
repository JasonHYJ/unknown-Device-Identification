import os
import shutil
from scapy.all import rdpcap, wrpcap, PacketList


# 功能说明：
# 将 uk 目录下各设备的 idle 流量 pcap 文件，按 3 分钟窗口、1 分钟滑步划分成多个样本；
# 若样本中数据包数 ≥ 10，则保留为新的 pcap 文件，结构与原目录一致；
# 命名格式为：DeviceName__idle__日期__样本编号.pcap

def get_pkt_timestamp(pkt):
    """提取每个数据包的时间戳（单位：秒）"""
    return float(pkt.time)


def process_idle_pcap(input_root, output_root, window_minutes=3, stride_minutes=1, min_packets_per_sample=10):
    """
    主处理函数：
    - 对所有设备的 idle pcap 文件进行窗口划分；
    - 将所有 activity 文件夹内容一并复制；
    - 输出结构与输入结构保持一致；
    - 样本文件命名方式统一为 Device__idle__Date__Index.pcap。
    """
    print("🚀 开始处理所有设备...\n")
    window_size = window_minutes * 60
    stride = stride_minutes * 60

    for device_name in os.listdir(input_root):
        device_path = os.path.join(input_root, device_name)
        idle_path = os.path.join(device_path, 'idle')
        activity_path = os.path.join(device_path, 'activity')

        print(f"\n📂 处理设备：{device_name}")

        # === 1. 处理 idle 流量 ===
        if os.path.isdir(idle_path):
            print("  ▶ 处理 idle 流量样本...")
            for pcap_file in os.listdir(idle_path):
                if not pcap_file.endswith('.pcap'):
                    continue

                pcap_path = os.path.join(idle_path, pcap_file)
                print(f"    ➤ 正在处理文件：{pcap_file}")

                try:
                    packets = rdpcap(pcap_path)
                except Exception as e:
                    print(f"    ❌ 无法读取 {pcap_file}，错误：{e}")
                    continue

                if len(packets) == 0:
                    print(f"    ⚠️ 文件为空，跳过：{pcap_file}")
                    continue

                file_date = os.path.splitext(pcap_file)[0]
                base_output_dir = os.path.join(output_root, device_name, 'idle', file_date)
                os.makedirs(base_output_dir, exist_ok=True)

                timestamps = [get_pkt_timestamp(pkt) for pkt in packets]
                start_time = min(timestamps)
                end_time = max(timestamps)

                sample_index = 0
                current_start = start_time

                while current_start + window_size <= end_time:
                    current_end = current_start + window_size
                    window_packets = PacketList(
                        [pkt for pkt in packets if current_start <= get_pkt_timestamp(pkt) < current_end]
                    )

                    if len(window_packets) >= min_packets_per_sample:
                        sample_index += 1
                        filename = f"{device_name}__idle__{file_date}__{sample_index:05d}.pcap"
                        save_path = os.path.join(base_output_dir, filename)

                        try:
                            wrpcap(save_path, window_packets)
                        except Exception as e:
                            print(f"    ❌ 保存失败 {filename}：{e}")

                    current_start += stride

                print(f"    ✅ 完成文件处理：生成 {sample_index} 个样本")

        # === 2. 复制 activity 文件夹 ===
        if os.path.isdir(activity_path):
            print("  ▶ 正在复制 activity 文件夹...")
            output_activity_path = os.path.join(output_root, device_name, 'activity')
            try:
                shutil.copytree(activity_path, output_activity_path, dirs_exist_ok=True)
                print(f"    ✅ 已复制 activity 到：{output_activity_path}")
            except Exception as e:
                print(f"    ❌ activity 文件夹复制失败：{e}")
        else:
            print("  ⚠️ 未找到 activity 文件夹，跳过。")

    print("\n🎉 所有设备处理完成。")


if __name__ == "__main__":
    input_root = "/home/hyj/unknownDeviceIdentification/dataset/originalFile/cicIoT2022"  # 输入根目录
    output_root = "/home/hyj/unknownDeviceIdentification/dataset/1_splited_pcap/cicIoT2022"  # 输出根目录
    window_minutes = 3  # 样本时间窗口（分钟）
    stride_minutes = 1  # 滑动步长（分钟）
    min_packets_per_sample = 10  # 最小数据包数

    process_idle_pcap(
        input_root=input_root,
        output_root=output_root,
        window_minutes=window_minutes,
        stride_minutes=stride_minutes,
        min_packets_per_sample=min_packets_per_sample
    )
