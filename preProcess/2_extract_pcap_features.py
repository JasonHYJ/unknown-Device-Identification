# -- coding: utf-8 --

"""
功能说明：
本脚本用于从大量 .pcap 网络抓包文件中提取结构化的流量特征数据，
并在输出的 CSV 文件中添加以下两类增强特征：
  1. direction（方向字段）：用于表示数据包是设备发出的（1）、接收的（-1）还是未知（0）；
  2. raw_bytes（十六进制原始字节）：用于保存每个数据包的原始二进制内容，以十六进制字符串表示。

该脚本结合了 tshark（用于提取结构化字段）与 scapy（用于读取原始字节内容）两个工具进行处理。

适用于物联网设备流量分析、指纹识别、深度学习输入等研究任务场景。

处理步骤：

1. 遍历输入目录下所有子目录，查找所有 .pcap 文件。
2. 使用 tshark 提取每个 pcap 文件的结构化字段（如时间戳、MAC、IP、TTL、payload等），生成初步的 .csv 文件。
3. 使用 scapy 逐包读取原始 pcap 文件，提取每个数据包的十六进制原始字节内容（raw_bytes）。
4. 对每条数据记录，根据 eth.src 和 eth.dst 判断其是否属于指定设备，添加 direction 字段。
   - 如果 eth.src 匹配设备 MAC → direction = 1（设备发出）
   - 如果 eth.dst 匹配设备 MAC → direction = -1（设备接收）
   - 否则 → direction = 0
5. 将所有字段和两列新增特征拼接后保存到最终输出的 CSV 文件中，覆盖原文件。
6. 保留与原始 pcap 相同的目录结构，保存于目标目录下。

注意事项：
- 依赖 tshark 和 scapy 库，请确保它们已正确安装。
- 若原始 .pcap 包数与 tshark 行数不一致（罕见情况），将只处理对齐部分。
- tshark 字段提取默认以英文逗号分隔，已进行字段长度检查。
"""


import os
import subprocess
import shlex
import csv
from scapy.all import rdpcap

# 设备 MAC 地址字典，将设备名称与其对应的 MAC 地址进行映射
device_mac_mapping = {
    # uk数据集
    "uk_allure-speaker": "b0:f1:ec:d4:26:ae",
    "uk_appletv": "50:32:37:b8:c7:0f",
    "uk_blink-camera": "f4:b8:5e:68:8f:35",
    "uk_blink-security-hub": "00:03:7f:96:d8:ec",
    "uk_bosiwo-camera-wired": "ae:ca:06:0e:ec:89",
    "uk_charger-camera": "fc:ee:e6:2e:23:a3",
    "uk_echodot": "cc:f7:35:49:f4:05",
    "uk_echoplus": "00:fc:8b:84:22:10",
    "uk_echospot": "5c:41:5a:29:ad:97",
    "uk_firetv": "cc:f7:35:25:af:4d",
    "uk_google-home": "54:60:09:6f:32:84",
    "uk_google-home-mini": "20:df:b9:13:e5:2e",
    "uk_honeywell-thermostat": "b8:2c:a0:28:3e:6b",
    "uk_lightify-hub": "84:18:26:7c:1a:56",
    "uk_magichome-strip": "dc:4f:22:89:fc:e7",
    "uk_nest-tstat": "64:16:66:2a:98:62",
    "uk_netatmo-weather-station": "70:ee:50:36:98:da",
    "uk_ring-doorbell": "f0:45:da:36:e6:23",
    "uk_roku-tv": "c8:3a:6b:fa:1c:00",
    "uk_samsungtv-wired": "fc:03:9f:93:22:62",
    "uk_sengled-hub": "b0:ce:18:20:43:bf",
    "uk_smarter-coffee-mach": "0c:2a:69:11:01:ba",
    "uk_smartthings-hub": "d0:52:a8:a4:e6:46",
    "uk_sousvide": "68:c6:3a:ba:c2:6b",
    "uk_t-philips-hub": "ec:b5:fa:00:98:da",
    "uk_tplink-bulb": "50:c7:bf:ca:3f:9d",
    "uk_tplink-plug": "50:c7:bf:b1:d2:78",
    "uk_t-wemo-plug": "58:ef:68:99:7d:ed",
    "uk_wansview-cam-wired": "78:a5:dd:28:a1:b7",
    "uk_xiaomi-cam2": "78:11:dc:76:69:b0",
    "uk_xiaomi-cleaner": "78:11:dc:ec:a3:ab",
    "uk_xiaomi-hub": "7c:49:eb:88:da:82",
    "uk_yi-camera": "0c:8c:24:0b:be:fb",

    # us数据集
    "us_amcrest-cam-wired": "9c:8e:cd:0a:33:1b",
    "us_appletv": "08:66:98:a2:21:9e",
    "us_blink-camera": "f4:b8:5e:31:73:db",
    "us_blink-security-hub": "00:03:7f:4f:c6:b5",
    "us_brewer": "20:f8:5e:cc:18:1f",
    "us_bulb1": "ec:fa:bc:82:20:bb",
    "us_cloudcam": "b0:fc:0d:c9:00:4c",
    "us_dlink-mov": "6c:72:20:c5:0a:3f",
    "us_dryer": "c0:97:27:73:aa:38",
    "us_echodot": "18:74:2e:41:4d:35",
    "us_echoplus": "fc:a1:83:38:e0:2d",
    "us_echospot": "00:71:47:c0:91:93",
    "us_firetv": "6c:56:97:35:39:f4",
    "us_fridge": "70:2c:1f:3b:36:53",
    "us_google-home-mini": "20:df:b9:5f:41:7e",
    "us_ikettle": "0c:2a:69:0e:91:16",
    "us_insteon-hub": "00:0e:f3:3b:85:e5",
    "us_invoke": "d8:f7:10:c3:34:e4",
    "us_lefun-cam-wired": "ae:ca:06:08:d3:e6",
    "us_lgtv-wired": "38:8c:50:68:d7:5c",
    "us_lightify-hub": "84:18:26:7d:cf:a2",
    "us_luohe-spycam": "00:0c:43:20:32:bb",
    "us_magichome-strip": "dc:4f:22:c1:58:05",
    "us_microseven-camera": "00:fc:5c:e0:81:86",
    "us_microwave": "d8:28:c9:10:b5:60",
    "us_nest-tstat": "18:b4:30:c8:d8:28",
    "us_philips-bulb": "34:ce:00:99:9b:83",
    "us_ring-doorbell": "98:84:e3:e4:35:bd",
    "us_roku-tv": "88:de:a9:08:03:b9",
    "us_samsungtv-wired": "84:c0:ef:2f:42:cc",
    "us_sengled-hub": "b0:ce:18:27:9f:e4",
    "us_smartthings-hub": "24:fd:5b:04:1b:75",
    "us_sousvide": "dc:4f:22:28:b6:5b",
    "us_t-philips-hub": "00:17:88:68:5f:61",
    "us_tplink-bulb": "50:c7:bf:a0:f3:76",
    "us_tplink-plug": "50:c7:bf:5a:2e:a0",
    "us_t-wemo-plug": "14:91:82:b4:4b:5f",
    "us_wansview-cam-wired": "78:a5:dd:1a:15:19",
    "us_washer": "c0:97:27:81:67:99",
    "us_wink-hub2": "00:21:cc:4d:ce:8c",
    "us_xiaomi-hub": "34:ce:00:83:99:35",
    "us_xiaomi-ricecooker": "7c:49:eb:35:7a:49",
    "us_xiaomi-strip": "34:ce:00:8b:22:74",
    "us_yi-camera": "b0:d5:9d:b9:f0:b4",
    "us_zmodo-doorbell": "7c:c7:09:56:6e:48",

    # ciciot数据集
    "ciciot_AmazonAlexaEchoDot1": "1c:fe:2b:98:16:dd",
    "ciciot_AmazonAlexaEchoDot2": "a0:d0:dc:c4:08:ff",
    "ciciot_AmazonAlexaEchoSpot": "1c:12:b0:9b:0c:ec",
    "ciciot_AmazonAlexaEchoStudio": "08:7c:39:ce:6e:2a",
    "ciciot_AmazonPlug": "b8:5f:98:d0:76:e6",
    "ciciot_AMCREST-WiFiCamera": "9c:8e:cd:1d:ab:9f",
    "ciciot_ArloBaseStation": "3c:37:86:6f:b9:51",
    "ciciot_ArloQCamera": "40:5d:82:35:14:c8",
    "ciciot_AtomiCoffeeMaker": "68:57:2d:56:ac:47",
    "ciciot_Borun-Sichuan-AICamera": "c0:e7:bf:0a:79:d1",
    "ciciot_D-LinkDCHS-161WaterSensor": "f0:b4:d2:f9:60:95",
    "ciciot_DCS8000LHA1D-LinkMiniCamera": "b0:c5:54:59:2e:99",
    "ciciot_EufyHomeBase2": "8c:85:80:6c:b6:47",
    "ciciot_GlobeLampESP_B1680C": "50:02:91:b1:68:0c",
    "ciciot_GoogleNestMini": "cc:f4:11:9c:d0:00",
    "ciciot_GosundESP_032979Plug": "b8:f0:09:03:29:79",
    "ciciot_GosundESP_039AAFSocket": "b8:f0:09:03:9a:af",
    "ciciot_GosundESP_0C3994Plug": "c4:dd:57:0c:39:94",
    "ciciot_GosundESP_10098FSocket": "50:02:91:10:09:8f",
    "ciciot_GosundESP_10ACD8Plug": "50:02:91:10:ac:d8",
    "ciciot_GosundESP_147FF9Plug": "24:a1:60:14:7f:f9",
    "ciciot_GosundESP_1ACEE1Socket": "50:02:91:1a:ce:e1",
    "ciciot_HeimVisionSmartLifeRadio-Lamp": "d4:a6:51:30:64:b7",
    "ciciot_HeimVisionSmartWiFiCamera": "44:01:bb:ec:10:4a",
    "ciciot_HomeEyeCamera": "34:75:63:73:f3:36",
    "ciciot_iRobotRoomba": "50:14:79:37:80:18",
    "ciciot_LuoheCamDog": "7c:a7:b0:cd:18:32",
    "ciciot_NestIndoorCamera": "44:bb:3b:00:39:07",
    "ciciot_NetatmoCamera": "70:ee:50:68:0e:32",
    "ciciot_NetatmoWeatherStation": "70:ee:50:6b:a8:1a",
    "ciciot_PhilipsHueBridge": "00:17:88:60:d6:4f",
    "ciciot_RingBaseStationAC": "b0:09:da:3e:82:6c",
    "ciciot_SIMCAM1SAMPAKTec": "10:2c:6b:1b:43:be",
    "ciciot_SmartBoard": "00:02:75:f6:e3:cb",
    "ciciot_SonosOneSpeaker": "48:a6:b8:f9:1b:88",
    "ciciot_TeckinPlug1": "d4:a6:51:76:06:64",
    "ciciot_TeckinPlug2": "d4:a6:51:78:97:4e",
    "ciciot_YutronPlug1": "d4:a6:51:20:91:d1",
    "ciciot_YutronPlug2": "d4:a6:51:21:6c:29",

}


def extract_features_from_pcap(source_root, destination_root, device_mac_mapping):
    """
    从 pcap 文件中提取特征信息，添加方向字段和原始字节字段，并保存为 CSV。
    """

    processed_file_count = 0
    matched_macs = set(mac.lower() for mac in device_mac_mapping.values())

    for root, _, files in os.walk(source_root, topdown=False):
        for name in files:
            if not name.endswith('.pcap'):
                continue

            file_path = os.path.join(root, name)
            csv_name = name.replace('.pcap', '.csv')
            relative_path = os.path.relpath(root, source_root)
            destination_dir = os.path.join(destination_root, relative_path)
            os.makedirs(destination_dir, exist_ok=True)
            destination_csv = os.path.join(destination_dir, csv_name)

            print(f"\n📁 正在处理: {file_path}")
            print(f"📄 输出 CSV: {destination_csv}")

            # 运行 tshark 提取基本字段
            tshark_cmd = (
                    'tshark -r ' + shlex.quote(file_path) +
                    ' -T fields -e frame.time_epoch -e frame.protocols -e frame.len -e eth.src -e eth.dst ' +
                    '-e ip.src -e ip.dst -e ip.len -e tcp.len -e udp.length -e ip.ttl -e tcp.srcport -e tcp.dstport ' +
                    '-e udp.srcport -e udp.dstport -e tcp.flags -e tls.record.content_type -e tcp.window_size ' +
                    '-e _ws.expert.message -e tcp.payload -e udp.payload ' +
                    '-E header=y -E separator=, -E quote=d -E occurrence=f > ' +
                    shlex.quote(destination_csv)
            )
            try:
                subprocess.run(tshark_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ tshark 错误: {e}")
                continue

            # 使用 scapy 提取原始字节数据
            try:
                packets = rdpcap(file_path)
                raw_bytes_list = [''.join(f"{byte:02x}" for byte in bytes(pkt)) for pkt in packets]
            except Exception as e:
                print(f"❌ scapy 读取错误: {e}")
                raw_bytes_list = []

            # 合并 direction 字段和 raw_bytes 字段
            temp_csv = destination_csv + ".temp"
            with open(destination_csv, 'r') as infile, open(temp_csv, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                headers = next(reader)
                headers += ['direction', 'raw_bytes']
                writer.writerow(headers)

                for i, row in enumerate(reader):
                    if len(row) < 5:
                        continue
                    src_mac = row[3].lower()
                    dst_mac = row[4].lower()
                    direction = 0
                    if src_mac in matched_macs:
                        direction = 1
                    elif dst_mac in matched_macs:
                        direction = -1

                    raw_bytes = raw_bytes_list[i] if i < len(raw_bytes_list) else ''
                    row += [str(direction), raw_bytes]
                    writer.writerow(row)

            os.replace(temp_csv, destination_csv)
            processed_file_count += 1

    print(f"\n✅ 所有文件处理完成，共处理 {processed_file_count} 个 pcap 文件。")


def main():
    # 示例路径（请替换为你自己的路径）
    source_root = "/home/hyj/unknownDeviceIdentification/dataset/1_splited_pcap/cicIoT2022"
    destination_root = "/home/hyj/unknownDeviceIdentification/dataset/2_csv/cicIoT2022"

    extract_features_from_pcap(source_root, destination_root, device_mac_mapping)


if __name__ == "__main__":
    main()
