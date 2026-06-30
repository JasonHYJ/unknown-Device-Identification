import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm

# ------------------------------------------------------------
# 功能说明：
# 本脚本用于对清洗后特征数据（统计/序列/原始字节）进行样本划分。
# - 按设备随机选取20%为未知设备，其所有样本标注为 "unknown"
# - 剩余设备：
#   - 所有行为样本划为 train
#   - 每台设备最多选 300 条闲时样本为 train，其余为 test
# 标签缺失时默认填充为: unknownType / unknownBrand / unknownDevice
# ------------------------------------------------------------

def generate_sample_split_with_full_path(stat_root_dir, output_csv_path,
                                         unknown_ratio=0.2, idle_train_sample_limit=300):
    stat_root_dir = Path(stat_root_dir).resolve()
    seq_root_dir = Path(str(stat_root_dir).replace("statistical_feature", "sequence_feature_matrix")).resolve()
    raw_root_dir = Path(str(stat_root_dir).replace("statistical_feature", "rawByte_feature_matrix")).resolve()

    all_devices = [d for d in stat_root_dir.iterdir() if d.is_dir()]
    print(f"📁 正在处理数据集目录: {stat_root_dir}")
    print(f"🔍 共发现设备数量: {len(all_devices)}")

    unknown_device_count = max(1, int(len(all_devices) * unknown_ratio))
    unknown_devices = set(random.sample([d.name for d in all_devices], unknown_device_count))
    print(f"🎯 随机选择未知设备 ({unknown_device_count} 台): {list(unknown_devices)}")

    # ===== 指定未知设备 =====
    # predefined_unknown_devices = {
    #     'tplink-plug', 'blink-camera', 'ikettle', 'firetv', 'sousvide', 'insteon-hub', 'xiaomi-strip', 'washer'
    # }
    # unknown_devices = set(d.name for d in all_devices if d.name in predefined_unknown_devices)
    # print(f"🎯 使用预设未知设备 ({len(unknown_devices)} 台): {list(unknown_devices)}")

    all_records = []

    for device_dir in tqdm(all_devices, desc="📦 处理设备"):
        device = device_dir.name
        is_unknown = device in unknown_devices

        for mode in ["activity", "idle"]:
            mode_dir = device_dir / mode
            if not mode_dir.exists():
                print(f"❌ 模式目录不存在: {mode_dir}")
                continue

            is_behavior = int(mode == "activity")
            all_samples = []

            for behavior_group in mode_dir.iterdir():
                if not behavior_group.is_dir():
                    continue
                for stat_file in behavior_group.glob("*_stat.csv"):
                    try:
                        df = pd.read_csv(stat_file)
                        if df.empty:
                            continue
                        row = df.iloc[0]
                        sample_file = row["sample_file"]
                        sample_base = sample_file.replace(".csv", "").replace("_stat", "")

                        # ===== 标签缺失处理 =====
                        type_label = row["type_label"] if pd.notna(row["type_label"]) else "unknownType"
                        brand_label = row["brand_label"] if pd.notna(row["brand_label"]) else "unknownBrand"
                        device_label = row["device_label"] if pd.notna(row["device_label"]) else "unknownDevice"

                        relative_dir = stat_file.parent.relative_to(stat_root_dir)
                        seq_file = seq_root_dir / relative_dir / f"{sample_base}_seq.npz"
                        raw_file = raw_root_dir / relative_dir / f"{sample_base}_raw.npz"

                        record = {
                            "file_path": str(stat_file.resolve()),
                            "device": device,
                            "is_behavior": is_behavior,
                            "set_type": "unknown" if is_unknown else "",
                            "type_label": type_label,
                            "brand_label": brand_label,
                            "device_label": device_label,
                            "sample_file": sample_file,
                            "sample_base": sample_base,
                            "seq_feature_path": str(seq_file.resolve()),
                            "raw_feature_path": str(raw_file.resolve()),
                            "behavior_type": behavior_group.name if is_behavior else "",
                            "idle_group": behavior_group.name if not is_behavior else ""
                        }
                        all_samples.append(record)
                    except Exception as e:
                        print(f"⚠️ 无法读取文件 {stat_file}: {e}")

            # 根据设备状态进行样本划分
            if is_unknown:
                for r in all_samples:
                    r["set_type"] = "unknown"
                all_records.extend(all_samples)
            else:
                if is_behavior:
                    for r in all_samples:
                        r["set_type"] = "train"
                    all_records.extend(all_samples)
                else:
                    random.shuffle(all_samples)
                    for i, r in enumerate(all_samples):
                        r["set_type"] = "train" if i < idle_train_sample_limit else "test"
                    all_records.extend(all_samples)

    if not all_records:
        print("❌ 未收集到任何样本，可能路径或结构设置错误。")
        return

    df_out = pd.DataFrame(all_records)
    df_out.to_csv(output_csv_path, index=False)

    print(f"\n✅ 样本划分完成，保存至: {output_csv_path}")
    print(f"📊 总样本数: {len(df_out)}")
    print(f"📈 训练样本数: {len(df_out[df_out.set_type == 'train'])}")
    print(f"🧪 测试样本数: {len(df_out[df_out.set_type == 'test'])}")
    print(f"❓ 未知设备样本数: {len(df_out[df_out.set_type == 'unknown'])}")


def main():
    stat_root = "/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_statistical_feature/us"
    output_csv = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/us/3_us_full_split.csv"
    generate_sample_split_with_full_path(stat_root, output_csv)

if __name__ == "__main__":
    main()
