from pathlib import Path
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# -----------------------------
# 脚本功能说明：
# 本脚本用于从csv样本文件中提取IoT原始字节特征矩阵。
# 处理流程：
# 1. 每个样本保留前N个数据包，每个数据包保留前M个原始字节（不足填充0）；
# 2. 屏蔽raw_bytes中的敏感字段（如MAC地址、IP地址、端口号、DNS/mDNS域名等）；
# 3. 保存为.npz文件，包含：raw_matrix, mask, original_len, is_behavior, labels, sample_file
# -----------------------------

def mask_sensitive_fields(df, max_byte_len):
    masked_bytes_list = []

    for idx, row in df.iterrows():
        try:
            raw_hex = row['raw_bytes']
            raw_bytes = bytearray.fromhex(str(raw_hex))

            # MAC 地址（前12字节）
            if len(raw_bytes) >= 12:
                raw_bytes[0:6] = b'\x00' * 6
                raw_bytes[6:12] = b'\x00' * 6

            # IP 地址（14字节偏移处）
            if len(raw_bytes) >= 34:
                raw_bytes[26:30] = b'\x00' * 4  # ip.src
                raw_bytes[30:34] = b'\x00' * 4  # ip.dst

            # 屏蔽端口号（TCP 或 UDP，偏移 34~37）
            if ("tcp.srcport" in row and pd.notna(row["tcp.srcport"])) or \
                    ("udp.srcport" in row and pd.notna(row["udp.srcport"])):
                if len(raw_bytes) >= 38:
                    raw_bytes[34:36] = b'\x00\x00'  # src port
                    raw_bytes[36:38] = b'\x00\x00'  # dst port

            # DNS/mDNS域名屏蔽（源或目的端口为 53 / 5353）
            src_port = int(row["udp.srcport"]) if pd.notna(row["udp.srcport"]) else -1
            dst_port = int(row["udp.dstport"]) if pd.notna(row["udp.dstport"]) else -1
            if (src_port in [53, 5353] or dst_port in [53, 5353]) and len(raw_bytes) > 54:
                dns_start = 42
                qname_start = dns_start + 12
                i = qname_start

                # 清零 QNAME（按照 DNS 格式逐个label跳跃）
                while i < len(raw_bytes) and raw_bytes[i] != 0:
                    length = raw_bytes[i]
                    if i + length + 1 >= len(raw_bytes):
                        break  # 安全保护：防止越界
                    raw_bytes[i:i + length + 1] = b'\x00' * (length + 1)
                    i += length + 1

                # 跳过结尾的0字节
                if i < len(raw_bytes):
                    raw_bytes[i] = 0x00
                    i += 1

                # 此时 i 应该指向 QTYPE 的起始位置，保留 QTYPE 和 QCLASS
                # 域名结束符（0x00）后是 QTYPE（2字节）+ QCLASS（2字节）
                # 保留这 4 字节，其余清零
                qtype_qclass_end = i + 4

                # 从 QCLASS 之后全部置零
                if qtype_qclass_end < len(raw_bytes):
                    raw_bytes[qtype_qclass_end:] = b'\x00' * (len(raw_bytes) - qtype_qclass_end)

            # 补零或截断
            if len(raw_bytes) >= max_byte_len:
                raw_bytes = raw_bytes[:max_byte_len]
            else:
                raw_bytes.extend(b'\x00' * (max_byte_len - len(raw_bytes)))

            masked_bytes_list.append(raw_bytes)
        except Exception as e:
            print(f"❌ 字节处理失败: 第 {idx} 行, 错误: {e}")
            masked_bytes_list.append(bytearray(max_byte_len))

    return masked_bytes_list


def extract_raw_matrix(df, max_pkt_len=256, max_byte_len=128):
    masked_raw_list = mask_sensitive_fields(df, max_byte_len)
    actual_len = len(masked_raw_list)
    valid_len = min(actual_len, max_pkt_len)

    matrix = np.zeros((max_pkt_len, max_byte_len), dtype=np.uint8)
    mask = np.zeros((max_pkt_len,), dtype=np.float32)

    for i in range(valid_len):
        matrix[i] = np.frombuffer(masked_raw_list[i], dtype=np.uint8)
        mask[i] = 1.0

    return matrix, mask, actual_len


def process_all_raw_samples(input_dir, output_dir, label_map, max_behavior_pkt=256, max_idle_pkt=128, byte_len=128):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    total_samples = 0
    total_behavior = 0
    total_idle = 0

    print(f"📁 正在处理输入目录: {input_root.resolve()}")

    for device_dir in tqdm(list(input_root.iterdir()), desc="设备遍历"):
        if not device_dir.is_dir():
            continue

        device_name = device_dir.name
        print(f"\n🚧 正在处理设备: {device_name}")
        if device_name not in label_map:
            print(f"⚠️ 缺失标签：{device_name}，跳过")
            continue

        labels = label_map[device_name]
        for mode in ["activity", "idle"]:
            mode_dir = device_dir / mode
            if not mode_dir.exists():
                continue
            is_behavior = 1 if mode == "activity" else 0
            max_pkt = max_behavior_pkt if is_behavior else max_idle_pkt

            for subfolder in mode_dir.iterdir():
                if not subfolder.is_dir():
                    continue

                for csv_file in subfolder.glob("*.csv"):
                    try:
                        print(f"📄 正在处理文件: {csv_file}")
                        df = pd.read_csv(csv_file)
                        raw_matrix, mask, actual_len = extract_raw_matrix(df, max_pkt_len=max_pkt,
                                                                          max_byte_len=byte_len)

                        relative_path = csv_file.relative_to(input_root)
                        relative_path = relative_path.with_name(relative_path.stem + "_raw.npz")
                        output_path = output_root / relative_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        np.savez_compressed(output_path,
                                            raw_matrix=raw_matrix,
                                            mask=mask,
                                            original_len=actual_len,
                                            is_behavior=is_behavior,
                                            type_label=labels["type"],
                                            brand_label=labels["brand"],
                                            device_label=device_name,
                                            sample_file=csv_file.name)

                        total_samples += 1
                        if is_behavior:
                            total_behavior += 1
                        else:
                            total_idle += 1
                        # print(f"✅ 处理完成: {csv_file.name}")

                    except Exception as e:
                        print(f"❌ 读取失败: {csv_file}, 原因: {e}")

    print("\n✅ 所有样本处理完成")
    print(f"📊 总样本数: {total_samples}")
    print(f"🔹 行为样本数: {total_behavior}")
    print(f"🔹 闲时样本数: {total_idle}")


def main():
    input_dir = "/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/cicIoT2022"
    output_dir = "/home/hyj/unknownDeviceIdentification/dataset/6_csv_rawByte_feature_matrix/cicIoT2022"

    label_map = {
        # uk数据集
        "allure-speaker": {"brand": "HarmanKardon", "type": "Audio"},
        "appletv": {"brand": "Apple", "type": "TV"},
        "blink-camera": {"brand": "Blink", "type": "Camera"},
        "blink-security-hub": {"brand": "Blink", "type": "Camera"},
        "bosiwo-camera-wired": {"brand": "Bosiwo", "type": "Camera"},
        "charger-camera": {"brand": "None", "type": "Camera"},
        "echodot": {"brand": "Amazon", "type": "Audio"},
        "echoplus": {"brand": "Amazon", "type": "Audio"},
        "echospot": {"brand": "Amazon", "type": "Audio"},
        "firetv": {"brand": "Amazon", "type": "TV"},
        "google-home": {"brand": "Google", "type": "Audio"},
        "google-home-mini": {"brand": "Google", "type": "Audio"},
        "honeywell-thermostat": {"brand": "Honeywell", "type": "HomeAutomation"},
        "lightify-hub": {"brand": "Osram", "type": "SmartHubs"},
        "magichome-strip": {"brand": "Magichone", "type": "HomeAutomation"},
        "nest-tstat": {"brand": "Nest", "type": "HomeAutomation"},
        "netatmo-weather-station": {"brand": "Netatmo", "type": "Appliances1"},
        "ring-doorbell": {"brand": "Ring", "type": "Camera"},
        "roku-tv": {"brand": "Roku", "type": "TV"},
        "samsungtv-wired": {"brand": "Samsung", "type": "TV"},
        "sengled-hub": {"brand": "Sengled", "type": "SmartHubs"},
        "smarter-coffee-mach": {"brand": "Smarter", "type": "Appliances2"},
        "smartthings-hub": {"brand": "Smartthings", "type": "SmartHubs"},
        "sousvide": {"brand": "Anova", "type": "Appliances3"},
        "t-philips-hub": {"brand": "PhilipsHue", "type": "SmartHubs"},
        "tplink-bulb": {"brand": "Tp-link", "type": "HomeAutomation"},
        "tplink-plug": {"brand": "Tp-link", "type": "HomeAutomation"},
        "t-wemo-plug": {"brand": "Belkin", "type": "HomeAutomation"},
        "wansview-cam-wired": {"brand": "Wansview", "type": "Camera"},
        "xiaomi-cam2": {"brand": "Xiaomi", "type": "Camera"},
        "xiaomi-cleaner": {"brand": "Xiaomi", "type": "Appliances4"},
        "xiaomi-hub": {"brand": "Xiaomi", "type": "SmartHubs"},
        "yi-camera": {"brand": "Yi", "type": "Camera"},

        # us数据集
        "amcrest-cam-wired": {"brand": "Amcrest", "type": "Camera"},
        # "appletv": {"brand": "Apple", "type": "TV"},
        # "blink-camera": {"brand": "Blink", "type": "Camera"},
        # "blink-security-hub": {"brand": "Blink", "type": "Camera"},
        "brewer": {"brand": "Behmor", "type": "Appliances5"},
        "bulb1": {"brand": "Flux", "type": "HomeAutomation"},
        "cloudcam": {"brand": "Amazon", "type": "Camera"},
        "dlink-mov": {"brand": "Dlink", "type": "HomeAutomation"},
        "dryer": {"brand": "Samsung", "type": "Appliances6"},
        # "echodot": {"brand": "Amazon", "type": "Audio"},
        # "echoplus": {"brand": "Amazon", "type": "Audio"},
        # "echospot": {"brand": "Amazon", "type": "Audio"},
        # "firetv": {"brand": "Amazon", "type": "TV"},
        "fridge": {"brand": "Samsung", "type": "Appliances7"},
        # "google-home-mini": {"brand": "Google", "type": "Audio"},
        "ikettle": {"brand": "Smarter", "type": "Appliances8"},
        "insteon-hub": {"brand": "Insteon", "type": "SmartHubs"},
        "invoke": {"brand": "HarmanKardon", "type": "Audio"},
        "lefun-cam-wired": {"brand": "Lefun", "type": "Camera"},
        "lgtv-wired": {"brand": "LG", "type": "TV"},
        # "lightify-hub": {"brand": "Osram", "type": "SmartHubs"},
        "luohe-spycam": {"brand": "MediaTek", "type": "Camera"},
        # "magichome-strip": {"brand": "Magichone", "type": "HomeAutomation"},
        "microseven-camera": {"brand": "Microseven", "type": "Camera"},
        "microwave": {"brand": "GE", "type": "Appliances9"},
        # "nest-tstat": {"brand": "Nest", "type": "HomeAutomation"},
        "philips-bulb": {"brand": "PhilipsHue", "type": "HomeAutomation"},
        # "ring-doorbell": {"brand": "Ring", "type": "Camera"},
        # "roku-tv": {"brand": "Roku", "type": "TV"},
        # "samsungtv-wired": {"brand": "Samsung", "type": "TV"},
        # "sengled-hub": {"brand": "Sengled", "type": "SmartHubs"},
        # "smartthings-hub": {"brand": "Smartthings", "type": "SmartHubs"},
        # "sousvide": {"brand": "Anova", "type": "Appliances3"},
        # "t-philips-hub": {"brand": "PhilipsHue", "type": "SmartHubs"},
        # "tplink-bulb": {"brand": "Tp-link", "type": "HomeAutomation"},
        # "tplink-plug": {"brand": "Tp-link", "type": "HomeAutomation"},
        # "t-wemo-plug": {"brand": "Belkin", "type": "HomeAutomation"},
        # "wansview-cam-wired": {"brand": "Wansview", "type": "Camera"},
        "washer": {"brand": "Samsung", "type": "Appliances10"},
        "wink-hub2": {"brand": "Wink", "type": "SmartHubs"},
        # "xiaomi-hub": {"brand": "Xiaomi", "type": "SmartHubs"},
        "xiaomi-ricecooker": {"brand": "Xiaomi", "type": "Appliances11"},
        "xiaomi-strip": {"brand": "Xiaomi", "type": "HomeAutomation"},
        # "yi-camera": {"brand": "Yi", "type": "Camera"},
        "zmodo-doorbell": {"brand": "Zmodo", "type": "Camera"},

        # cicIoT2022数据集
        "AmazonAlexaEchoDot1": {"brand": "AmazonAlexa", "type": "Audio"},
        "AmazonAlexaEchoDot2": {"brand": "AmazonAlexa", "type": "Audio"},
        "AmazonAlexaEchoSpot": {"brand": "AmazonAlexa", "type": "Audio"},
        "AmazonAlexaEchoStudio": {"brand": "AmazonAlexa", "type": "Audio"},
        "AmazonPlug": {"brand": "SmartLife", "type": "HomeAutomation"},
        "AMCREST-WiFiCamera": {"brand": "AmcrestViewPro", "type": "Camera"},
        "ArloBaseStation": {"brand": "Arlo", "type": "Camera"},
        "ArloQCamera": {"brand": "Arlo", "type": "Camera"},
        "AtomiCoffeeMaker": {"brand": "AtomiSmart", "type": "HomeAutomation"},
        "Borun-Sichuan-AICamera": {"brand": "Y380", "type": "Camera"},
        "DCS8000LHA1D-LinkMiniCamera": {"brand": "mydlink", "type": "Camera"},
        "D-LinkDCHS-161WaterSensor": {"brand": "mydlink", "type": "Appliances12"},
        "EufyHomeBase2": {"brand": "eufySecurity", "type": "HomeAutomation"},
        "GlobeLampESP_B1680C": {"brand": "GlobeSuite", "type": "HomeAutomation"},
        "GoogleNestMini": {"brand": "GoogleHome", "type": "Audio"},
        "GosundESP_0C3994Plug": {"brand": "SmartLife", "type": "HomeAutomation"},
        "GosundESP_1ACEE1Socket": {"brand": "SmartLife", "type": "HomeAutomation"},
        "GosundESP_10ACD8Plug": {"brand": "SmartLife", "type": "HomeAutomation"},
        "GosundESP_039AAFSocket": {"brand": "SmartLife", "type": "HomeAutomation"},
        "GosundESP_147FF9Plug": {"brand": "SmartLife", "type": "HomeAutomation"},
        "GosundESP_10098FSocket": {"brand": "SmartLife", "type": "HomeAutomation"},
        "GosundESP_032979Plug": {"brand": "SmartLife", "type": "HomeAutomation"},
        "HeimVisionSmartLifeRadio-Lamp": {"brand": "SmartLife", "type": "HomeAutomation"},
        "HeimVisionSmartWiFiCamera": {"brand": "HeimLink", "type": "Camera"},
        "HomeEyeCamera": {"brand": "HomeEye", "type": "Camera"},
        "iRobotRoomba": {"brand": "iRobot", "type": "HomeAutomation"},
        "LuoheCamDog": {"brand": "CamDog", "type": "Camera"},
        "NestIndoorCamera": {"brand": "Nest", "type": "Camera"},
        "NetatmoCamera": {"brand": "Security", "type": "Camera"},
        "NetatmoWeatherStation": {"brand": "Netatmo", "type": "Appliances13"},
        "PhilipsHueBridge": {"brand": "Hue", "type": "HomeAutomation"},
        "RingBaseStationAC": {"brand": "Ring", "type": "HomeAutomation"},
        "SIMCAM1SAMPAKTec": {"brand": "SimHome", "type": "Camera"},
        "SmartBoard": {"brand": "None", "type": "HomeAutomation"},
        "SonosOneSpeaker": {"brand": "Sonos", "type": "Audio"},
        "TeckinPlug1": {"brand": "SmartLife", "type": "HomeAutomation"},
        "TeckinPlug2": {"brand": "SmartLife", "type": "HomeAutomation"},
        "YutronPlug1": {"brand": "SmartLife", "type": "HomeAutomation"},
        "YutronPlug2": {"brand": "SmartLife", "type": "HomeAutomation"},

    }

    process_all_raw_samples(input_dir, output_dir, label_map)


if __name__ == "__main__":
    main()
