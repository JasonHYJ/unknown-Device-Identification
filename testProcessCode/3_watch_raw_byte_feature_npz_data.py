import numpy as np
import os


def inspect_npz(npz_path, print_full_matrix=False):
    data = np.load(npz_path, allow_pickle=True)

    print(f"\n📂 正在查看文件: {npz_path}")
    print(f"✅ 字段列表: {list(data.keys())}")

    raw_matrix = data["raw_matrix"]
    mask = data["mask"]
    original_len = data["original_len"]
    is_behavior = data["is_behavior"]
    type_label = data["type_label"]
    brand_label = data["brand_label"]
    device_label = data["device_label"]
    sample_file = data["sample_file"]

    print(f"🔢 raw_matrix shape: {raw_matrix.shape}")
    print(f"🔢 mask: {mask}")
    print(f"🔢 mask shape: {mask.shape}")
    print(f"📏 original_len: {original_len}")
    print(f"📌 is_behavior: {is_behavior}")
    print(f"📁 type_label: {type_label}")
    print(f"📁 brand_label: {brand_label}")
    print(f"📁 device_label: {device_label}")
    print(f"📁 sample_file: {sample_file}")

    if print_full_matrix:
        print("\n📋 raw_matrix 内容（按包行显示，字节值为十六进制）:")
        for i in range(min(raw_matrix.shape[0], int(original_len))):
            hex_bytes = ' '.join([f"{b:02x}" for b in raw_matrix[i] if mask[i]])
            print(f"第 {i + 1} 个数据包: {hex_bytes}")

    else:
        # 可选：打印前几个非空包（便于观察）
        print("\n📋 raw_matrix 前5个非空数据包内容:")
        count = 0
        for i in range(raw_matrix.shape[0]):
            if mask[i] > 0 and count < 5:
                hex_bytes = ' '.join([f"{b:02x}" for b in raw_matrix[i]])
                print(f"第 {i + 1} 个包: {hex_bytes}")
                count += 1


if __name__ == "__main__":
    # 示例路径（请修改为你自己的文件路径）
    npz_path = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_rawByte_feature_matrix/uk/allure-speaker/activity/android_lan_audio_off/allure-speaker__android_lan_audio_off__00001_raw.npz"
    # npz_path = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_rawByte_feature_matrix/uk/allure-speaker/idle/2019-04-25_idle/allure-speaker__2019-04-25_idle__00001_raw.npz"

    # 设置是否要完整打印 256*128 或 128*128 的矩阵内容（True 慎用，输出很长）
    inspect_npz(npz_path, print_full_matrix=False)
