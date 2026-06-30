import os
import shutil
from pathlib import Path
from tqdm import tqdm

"""
目前不用的脚本
"""
# -----------------------------
# 清洗特征样本脚本
# 功能说明：
# - 针对统计特征（CSV）、序列特征（.npz）、原始字节特征（.npz）进行统一清洗；
# - 清洗条件：设备下所有行为样本总数 >= min_behavior_samples，才保留其 activity 文件夹；
# - 所有 idle 文件夹均保留；
# - 清洗后文件保存至新目录：结构保持一致。
# - 这个代码是对每一个activity目录进行处理，依次遍历，没有另一个代码快
# -----------------------------


def is_valid_behavior_dir(behavior_dir: Path, min_behavior_samples: int = 10) -> bool:
    """判断某个 activity 文件夹是否满足保留条件"""
    all_csv_files = list(behavior_dir.rglob("*.csv")) + list(behavior_dir.rglob("*.npz"))
    return len(all_csv_files) >= min_behavior_samples


def clean_dataset(input_dirs, output_dirs, min_behavior_samples=10):
    """主清洗函数，对所有类型的特征数据进行同步清洗"""
    assert len(input_dirs) == len(output_dirs), "输入输出路径数量不一致"

    for input_root, output_root in zip(input_dirs, output_dirs):
        input_root = Path(input_root)
        output_root = Path(output_root)

        if not input_root.exists():
            print(f"❌ 输入目录不存在: {input_root}")
            continue
        
        # if not output_root.exists():
        #     output_root.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 正在清洗数据集: {input_root.name}")
        dataset_names = [d for d in input_root.iterdir() if d.is_dir()]

        for dataset_dir in tqdm(dataset_names, desc=f"数据集遍历 - {input_root.name}"):
            for device_dir in dataset_dir.iterdir():
                if not device_dir.is_dir():
                    continue

                device_name = device_dir.name
                behavior_dir = device_dir / "activity"
                idle_dir = device_dir / "idle"

                # 是否保留行为数据
                keep_behavior = behavior_dir.exists() and is_valid_behavior_dir(behavior_dir, min_behavior_samples)

                # 拷贝行为数据
                if keep_behavior:
                    for file in behavior_dir.rglob("*.*"):
                        rel_path = file.relative_to(input_root)
                        out_path = output_root / rel_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, out_path)

                # 拷贝闲时数据（全部保留）
                if idle_dir.exists():
                    for file in idle_dir.rglob("*.*"):
                        rel_path = file.relative_to(input_root)
                        out_path = output_root / rel_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, out_path)

        print(f"✅ 清洗完成: {input_root.name} → {output_root.name}")


def main():
    # 输入目录：三种类型的特征
    input_dirs = [
        "/home/hyj/unknownDeviceIdentification/dataset/test/stat_feature",
        "/home/hyj/unknownDeviceIdentification/dataset/test/sequence_feature_matrix",
        "/home/hyj/unknownDeviceIdentification/dataset/test/raw_bytes_feature_matrix",
    ]

    # 输出目录：清洗后的结果保存
    output_dirs = [
        "/home/hyj/unknownDeviceIdentification/dataset/test/7_cleaned_stat_features",
        "/home/hyj/unknownDeviceIdentification/dataset/test/7_cleaned_sequence_features",
        "/home/hyj/unknownDeviceIdentification/dataset/test/7_cleaned_rawbytes_features",
    ]

    clean_dataset(input_dirs, output_dirs, min_behavior_samples=10)


if __name__ == "__main__":
    main()
