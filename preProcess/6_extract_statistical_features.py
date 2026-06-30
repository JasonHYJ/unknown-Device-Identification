# 添加样本计数统计的功能
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------
# 统计特征提取脚本
# 功能说明：
# - 遍历输入目录中的所有设备文件夹；
# - 对每个行为/闲时文件夹中的每个样本 CSV 文件提取统计特征；
# - 添加 is_behavior 标记（1=行为，0=闲时）；
# - 添加标签字段（type_label, brand_label, device_label）；
# - 每个样本的统计特征保存为单独的CSV文件（xxx_stat.csv）
# - 保留输入目录的结构层级
# ---------------------------------------------

# 设备标签字典：用于为每个设备添加类型和品牌信息标签
device_label_map = {
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

# 定义输出的统计特征字段名称顺序，用于统一输出格式（可用于后续建模）
stat_feature_names = [
    "packet_count", "avg_pkt_len", "std_pkt_len", "max_pkt_len", "min_pkt_len", "total_bytes", "payload_bytes_total",
    "payload_bytes_ratio",
    "up_pkt_count", "down_pkt_count", "up_bytes", "down_bytes", "up_down_pkt_ratio", "up_down_byte_ratio", "udp_ratio",
    "avg_iat", "std_iat", "min_iat", "max_iat", "pkt_rate", "pkt_interval_entropy", "burst_count",
    "heartbeat_period_fft", "active_ratio", "burstiness",
    "tcp_count", "udp_count", "session_count",
    "unique_dst_ports", "entropy_pkt_size",
    "is_behavior", "type_label", "brand_label", "device_label", "sample_file"
]


# 计算离散熵的辅助函数，用于计算包大小分布或间隔时间的离散程度，分布越离散，熵越高
def safe_entropy(values):
    if len(values) == 0:
        return 0
    value_counts = pd.Series(values).value_counts(normalize=True)
    return -(value_counts * np.log2(value_counts)).sum()


# 新增：基于协议和方向的会话数提取函数
def extract_sessions(df):
    session_set = set()

    # TCP 会话
    tcp_df = df.dropna(subset=["tcp.srcport", "tcp.dstport"])
    for _, row in tcp_df.iterrows():
        ip1, ip2 = sorted([row["ip.src"], row["ip.dst"]])
        port1, port2 = sorted([row["tcp.srcport"], row["tcp.dstport"]])
        session_set.add((ip1, ip2, port1, port2, "TCP"))

    # UDP 会话
    udp_df = df.dropna(subset=["udp.srcport", "udp.dstport"])
    for _, row in udp_df.iterrows():
        ip1, ip2 = sorted([row["ip.src"], row["ip.dst"]])
        port1, port2 = sorted([row["udp.srcport"], row["udp.dstport"]])
        session_set.add((ip1, ip2, port1, port2, "UDP"))

    return len(session_set)


# 主特征提取函数：接收一个样本数据帧 df，输出字典形式的统计特征
def extract_stat_features(df, is_behavior):
    features = {}
    features["packet_count"] = len(df)  # 包总数
    features["avg_pkt_len"] = df["frame.len"].mean()  # 平均包长
    features["std_pkt_len"] = df["frame.len"].std()  # 包长标准差
    features["max_pkt_len"] = df["frame.len"].max()  # 最大包长
    features["min_pkt_len"] = df["frame.len"].min()  # 最小包长
    features["total_bytes"] = df["frame.len"].sum()  # 总流量字节数

    # TCP 与 UDP payload 长度相加（注意：若某协议为空将为 NaN，因此先转为 str 处理）
    tcp_payload_len = df["tcp.payload"].astype(str).str.len()
    udp_payload_len = df["udp.payload"].astype(str).str.len()
    df["payload_len"] = tcp_payload_len + udp_payload_len

    # payload 总字节数
    features["payload_bytes_total"] = df["payload_len"].sum()
    # payload 占整体字节的比例（衡量实际有效载荷比例）
    features["payload_bytes_ratio"] = features["payload_bytes_total"] / features["total_bytes"] if features[
                                                                                                       "total_bytes"] > 0 else 0
    # 区分上行下行统计
    up_df = df[df["direction"] == 1]
    down_df = df[df["direction"] == -1]
    features["up_pkt_count"] = len(up_df)  # 上行包数（设备 → 网关）
    features["down_pkt_count"] = len(down_df)  # 下行包数（网关 → 设备）
    features["up_bytes"] = up_df["frame.len"].sum()  # 上行字节数
    features["down_bytes"] = down_df["frame.len"].sum()  # 下行字节数
    features["up_down_pkt_ratio"] = features["up_pkt_count"] / features["down_pkt_count"] if features[
                                                                                                 "down_pkt_count"] > 0 else 0   # 上行包数 / 下行包数
    features["up_down_byte_ratio"] = features["up_bytes"] / features["down_bytes"] if features["down_bytes"] > 0 else 0     # 上行字节 / 下行字节
    features["udp_ratio"] = df["udp.length"].notna().sum() / len(df) if len(df) > 0 else 0      # UDP 包数 / 总包数

    # 时间间隔特征（inter-arrival time）
    iats = df["time_interval"].dropna().astype(float).values
    features["avg_iat"] = np.mean(iats) if len(iats) > 0 else 0     # 平均包间隔时间
    features["std_iat"] = np.std(iats) if len(iats) > 0 else 0      # 包间隔时间标准差
    features["min_iat"] = np.min(iats) if len(iats) > 0 else 0      # 最短包间隔时间
    features["max_iat"] = np.max(iats) if len(iats) > 0 else 0      # 最长包间隔时间
    features["pkt_rate"] = len(iats) / (np.sum(iats) + 1e-6) if len(iats) > 0 else 0    # 平均每秒包数
    features["pkt_interval_entropy"] = safe_entropy(np.round(iats, 3)) if len(iats) > 0 else 0  # 包间时间序列的熵
    features["burst_count"] = np.sum(iats < 1.0)  # 小于1秒间隔的 burst 包数量

    # 频域分析周期（仅对闲时流量进行） fft 提取主频周期，仅对闲时流量进行（行为流量无周期）
    if is_behavior:
        features["heartbeat_period_fft"] = -1  # 默认is_behavior=1为行为流量，不适用，填充为 -1
    else:
        try:
            fft = np.fft.fft(iats)
            fft_power = np.abs(fft[1:len(fft) // 2])
            dominant_freq = np.argmax(fft_power) + 1
            features["heartbeat_period_fft"] = 1 / dominant_freq if dominant_freq > 0 else -1
        except:
            features["heartbeat_period_fft"] = -1

    features["active_ratio"] = np.sum(iats < 1.0) / len(iats) if len(iats) > 0 else 0   # 包间时间 <1s 的包数占比
    features["burstiness"] = features["burst_count"] / len(iats) if len(iats) > 0 else 0    # 最大 burst 中包数 / 总包数

    # 协议与端口特征
    features["tcp_count"] = df["tcp.len"].notna().sum()     # TCP 数据包数
    features["udp_count"] = df["udp.length"].notna().sum()  # UDP 数据包数
    features["session_count"] = extract_sessions(df)    # 样本中会话（五元组）数量

    features["unique_dst_ports"] = df["tcp.dstport"].fillna(df["udp.dstport"]).dropna().nunique()   # 目的端口的去重数量
    features["entropy_pkt_size"] = safe_entropy(df["frame.len"])    # 包大小分布的熵

    # 行为标识
    features["is_behavior"] = int(is_behavior)  # 0 表示闲时，1 表示行为
    return features


def main():
    input_root = Path("/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/cicIoT2022")
    output_root = Path("/home/hyj/unknownDeviceIdentification/dataset/6_csv_statistical_feature/cicIoT2022")

    total_behavior = 0
    total_idle = 0
    total_samples = 0

    print(f"📁 输入目录: {input_root.resolve()}")
    print(f"📁 输出目录: {output_root.resolve()}")

    # 显示设备遍历的进度条
    for device_dir in tqdm(list(input_root.iterdir()), desc="设备遍历"):
        if not device_dir.is_dir():
            continue

        device_name = device_dir.name
        print(f"\n🚧 正在处理设备: {device_name}")

        if device_name not in device_label_map:
            print(f"⚠️ 未找到标签映射: {device_name}，跳过")
            continue

        labels = device_label_map[device_name]
        type_label = labels["type"]
        brand_label = labels["brand"]
        device_label = device_name

        for mode in ["activity", "idle"]:
            mode_dir = device_dir / mode
            if not mode_dir.exists():
                continue

            is_behavior = (mode == "activity")

            for subfolder in mode_dir.iterdir():
                if not subfolder.is_dir():
                    continue

                all_feature_rows = []
                for sample_file in sorted(subfolder.glob("*.csv")):
                    try:
                        df = pd.read_csv(sample_file)
                        stat = extract_stat_features(df.copy(), is_behavior)
                        stat["type_label"] = type_label
                        stat["brand_label"] = brand_label
                        stat["device_label"] = device_label
                        stat["sample_file"] = sample_file.name

                        # 构造输出路径
                        relative_path = sample_file.relative_to(input_root).parent
                        out_folder = output_root / relative_path
                        out_folder.mkdir(parents=True, exist_ok=True)
                        sample_base = os.path.splitext(sample_file.name)[0]
                        sample_name = sample_base + "_stat.csv"
                        out_path = out_folder / sample_name

                        out_df = pd.DataFrame([stat])[stat_feature_names]
                        out_df.to_csv(out_path, index=False)

                        # 更新样本计数
                        total_samples += 1
                        if is_behavior:
                            total_behavior += 1
                        else:
                            total_idle += 1

                    except Exception as e:
                        print(f"❌ 读取失败: {sample_file}，原因: {type(e).__name__}: {e}")
                        continue


    print(f"\n📊 总样本数: {total_samples}")
    print(f"🔹 行为样本数: {total_behavior}")
    print(f"🔹 闲时样本数: {total_idle}")


if __name__ == "__main__":
    main()
