import pandas as pd
from pathlib import Path

def split_dataset_csv(main_csv_path: str):
    """
    功能：
    将主样本划分文件（包含 set_type 字段）拆分成 train/test/unknown 三个子文件。
    
    参数：
    main_csv_path: 主 CSV 文件路径
    """
    main_csv_path = Path(main_csv_path)
    df = pd.read_csv(main_csv_path)

    output_dir = main_csv_path.parent
    dataset_name = main_csv_path.stem.replace('_full_split', '')

    for set_type in ['train', 'test', 'unknown']:
        subset = df[df['set_type'] == set_type]
        output_path = output_dir / f"{dataset_name}_{set_type}.csv"
        subset.to_csv(output_path, index=False)
        print(f"✅ 已保存 {set_type} 子集样本 {len(subset)} 条 → {output_path}")

def main():
    # 修改为你的实际路径
    split_dataset_csv("/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/cicIoT2022/3_cicIoT2022_full_split.csv")

if __name__ == "__main__":
    main()
