import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------
# 脚本功能说明：
# 本脚本用于提取IoT设备流量样本中的序列特征（包大小、包间时间、方向），
# 对每个样本生成定长的特征矩阵，并输出为 .npz 文件，供后续模型训练使用。
# - 输入：csv 文件（每行为一个数据包的特征）
# - 输出：.npz 文件（包含特征矩阵、有效mask、原始长度、行为标记）
# - 行为流量最大长度默认 256，闲时流量默认 128
# - 按设备目录、行为/闲时文件夹、子文件夹组织输出目录结构
# ---------------------------------------------------------

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


def extract_sequence_features(df, max_len, is_behavior):
    """
    提取三维序列特征矩阵：包大小、包间时间、方向。
    返回：特征矩阵、mask、原始包数、行为标记
    """

    pkt_lens = df["frame.len"].values
    iats = df["time_interval"].values
    directions = df["direction"].values

    actual_len = len(pkt_lens)
    valid_len = min(actual_len, max_len)

    feature_matrix = np.zeros((max_len, 3), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)

    feature_matrix[:valid_len, 0] = pkt_lens[:valid_len]
    feature_matrix[:valid_len, 1] = iats[:valid_len]
    feature_matrix[:valid_len, 2] = directions[:valid_len]
    mask[:valid_len] = 1.0

    return feature_matrix, mask, valid_len, is_behavior


def process_all_samples(input_root, output_root, behavior_len=256, idle_len=128):
    """
    主处理函数，遍历输入目录所有样本，按行为/闲时分别提取特征矩阵并保存
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    total_samples = 0
    total_behavior = 0
    total_idle = 0

    print(f"📁 正在处理输入目录: {input_root.resolve()}")

    # 遍历每个设备文件夹
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

            is_behavior = 1 if mode == "activity" else 0
            max_len = behavior_len if is_behavior else idle_len

            for sub_dir in mode_dir.iterdir():
                if not sub_dir.is_dir():
                    continue

                for csv_file in sub_dir.glob("*.csv"):
                    try:
                        print(f"📄 正在处理文件: {csv_file}")
                        df = pd.read_csv(csv_file)
                        feature_matrix, mask, original_len, is_behavior_flag = extract_sequence_features(df, max_len,
                                                                                                         is_behavior)

                        # 构建输出路径
                        relative_path = csv_file.relative_to(input_root)
                        relative_path = relative_path.with_name(relative_path.stem + "_seq.npz")
                        output_path = output_root / relative_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # 保存 .npz 文件，包括标签和样本名
                        np.savez_compressed(
                            output_path,
                            feature_matrix=feature_matrix,
                            mask=mask,
                            original_len=original_len,
                            is_behavior=is_behavior_flag,
                            type_label=type_label,
                            brand_label=brand_label,
                            device_label=device_label,
                            sample_file=csv_file.name
                        )

                        total_samples += 1
                        if is_behavior:
                            total_behavior += 1
                        else:
                            total_idle += 1

                    except Exception as e:
                        print(f"❌ 错误: {csv_file}, 原因: {e}")

    print("\n✅ 处理完成")
    print(f"📊 总样本数: {total_samples}")
    print(f"🔹 行为样本数: {total_behavior}")
    print(f"🔹 闲时样本数: {total_idle}")


def main():
    input_root = "/home/hyj/unknownDeviceIdentification/dataset/5_csv_clip_time_interval_log1p/us"
    output_root = "/home/hyj/unknownDeviceIdentification/dataset/6_csv_sequence_feature_matrix/us"

    process_all_samples(input_root, output_root)


if __name__ == "__main__":
    main()
