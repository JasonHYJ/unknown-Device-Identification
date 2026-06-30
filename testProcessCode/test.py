import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---------------------- 功能说明 ----------------------
# 本脚本从IoT流量CSV样本中提取原始字节序列(raw_bytes)，
# 屏蔽MAC/IP/端口/DNS域名等唯一字段，
# 将每个样本生成定长 (N包, M字节) 的特征矩阵，
# 并以 .npz 格式保存，包含mask和标签字段。
# ------------------------------------------------------

# 设备标签映射字典（用户自定义）
device_label_map = {
    "allure-speaker": {"brand": "allure", "type": "speaker"},
    "appletv": {"brand": "apple", "type": "tv"},
    "mi-plug": {"brand": "xiaomi", "type": "plug"},
    "huawei-camera": {"brand": "huawei", "type": "camera"},
}

# DNS域名部分：匹配ASCII域名字符（包括点）
DNS_PATTERN = re.compile(r'(?:[\x20-\x7E]{2,})')

# MAC/IP/PORT 预编译正则（16进制字符串）
HEX_PATTERN_MAP = {
    'mac': re.compile(r'([0-9a-f]{12})'),     # 连续12位hex，可能是mac
    'ipv4': re.compile(r'(c0a8[0-9a-f]{4}|0a[0-9a-f]{6})'),  # 常见私有IP段匹配（例：192.168.x.x）
    'port': re.compile(r'([0-9a-f]{4})'),     # 连续4位hex可能是端口（不精确）
}

def sanitize_raw_byte(hex_str):
    """对原始16进制字符串中的MAC/IP/端口/DNS明文字段进行屏蔽"""
    # 屏蔽DNS域名明文
    hex_str = DNS_PATTERN.sub('00', hex_str)
    # 屏蔽MAC/IP/端口
    for key, pattern in HEX_PATTERN_MAP.items():
        hex_str = pattern.sub('00', hex_str)
    return hex_str

def extract_raw_matrix(df, max_packets, max_bytes):
    """从样本数据帧中提取定长的原始字节特征矩阵"""
    matrix = np.zeros((max_packets, max_bytes), dtype=np.uint8)
    mask = np.zeros((max_packets,), dtype=np.float32)

    for i, hex_str in enumerate(df['raw_bytes'].values[:max_packets]):
        try:
            clean_hex = sanitize_raw_byte(hex_str)
            byte_arr = bytearray.fromhex(clean_hex)
            cut_bytes = byte_arr[:max_bytes]
            matrix[i, :len(cut_bytes)] = np.frombuffer(cut_bytes, dtype=np.uint8)
            mask[i] = 1.0
        except Exception as e:
            continue
    return matrix, mask

def process_all_samples(input_root, output_root, behavior_maxpkt=256, idle_maxpkt=128, max_bytes=128):
    input_root = Path(input_root)
    output_root = Path(output_root)

    total_samples = total_behavior = total_idle = 0
    print(f"📁 开始处理目录: {input_root}")

    for device_dir in tqdm(list(input_root.iterdir()), desc="设备遍历"):
        if not device_dir.is_dir():
            continue

        device_name = device_dir.name
        if device_name not in device_label_map:
            print(f"⚠️ 未找到标签: {device_name}，跳过")
            continue

        label = device_label_map[device_name]
        for mode in ['activity', 'idle']:
            mode_dir = device_dir / mode
            if not mode_dir.exists():
                continue
            is_behavior = 1 if mode == "activity" else 0
            max_pkt = behavior_maxpkt if is_behavior else idle_maxpkt

            for subdir in mode_dir.iterdir():
                if not subdir.is_dir():
                    continue

                for csv_file in sorted(subdir.glob("*.csv")):
                    try:
                        print(f"📄 正在处理: {csv_file.name}")
                        df = pd.read_csv(csv_file)
                        matrix, mask = extract_raw_matrix(df, max_pkt, max_bytes)

                        out_rel_path = csv_file.relative_to(input_root).with_suffix('.npz')
                        out_path = output_root / out_rel_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)

                        np.savez_compressed(out_path,
                                            raw_matrix=matrix,
                                            mask=mask,
                                            is_behavior=is_behavior,
                                            type_label=label["type"],
                                            brand_label=label["brand"],
                                            device_label=device_name,
                                            sample_file=csv_file.name)

                        total_samples += 1
                        if is_behavior: total_behavior += 1
                        else: total_idle += 1
                    except Exception as e:
                        print(f"❌ 错误: {csv_file}, 原因: {e}")

    print("\n✅ 所有样本处理完成")
    print(f"📊 总样本数: {total_samples}, 行为: {total_behavior}, 闲时: {total_idle}")

def main():
    input_root = "/home/hyj/unknownDeviceIdentification/dataset/test/csv_clip_time_interval_log1p"
    output_root = "/home/hyj/unknownDeviceIdentification/dataset/test/raw_byte_feature_matrix"
    process_all_samples(input_root, output_root)

if __name__ == "__main__":
    main()
