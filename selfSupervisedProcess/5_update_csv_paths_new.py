import os
import re
import pandas as pd

"""
更健壮的路径更新脚本：
- 不再写死数据集名，通过原始 stat_feature_path 自动解析 {dataset_name}
- 将三种特征路径重定向到 10_contrastive_embeddings 下对应的 train/test/unknown 子树
- 可用于 uk/us/cicIoT2022 等多个数据集
"""

BASE_DIR = "/home/hyj/unknownDeviceIdentification/dataset"
OUT_DIR  = f"{BASE_DIR}/11_multitask_training"   # 每个数据集会再加一层子目录

BASE_PATHS = {
    "stat": f"{BASE_DIR}/10_contrastive_embeddings/10_contrastive_statistical_embeddings",
    "seq":  f"{BASE_DIR}/10_contrastive_embeddings/10_contrastive_sequence_embeddings",
    "raw":  f"{BASE_DIR}/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings",
}

# 在原始路径中解析数据集名与子路径：
# 允许匹配 .../7_cleaned_features/*/<dataset_name>/<device>/<view>/...
DATASET_PATTERN = re.compile(r"/7_cleaned_features/[^/]+/(?P<dataset>[^/]+)/(?P<subpath>.+)/[^/]+_stat\.csv$")

def infer_dataset_and_subpath(old_stat_path: str):
    m = DATASET_PATTERN.search(old_stat_path)
    if not m:
        # 尝试匹配 10_contrastive_statistical_embeddings 的老路径（如果之前更新过）
        alt = re.compile(r"/10_contrastive_statistical_embeddings/(?P<settype>train|test|unknown)/(?P<dataset>[^/]+)/(?P<subpath>.+)/[^/]+_stat\.csv$")
        m2 = alt.search(old_stat_path)
        if m2:
            return m2.group("dataset"), m2.group("subpath")
        raise ValueError(f"无法从路径中解析数据集/子路径: {old_stat_path}")
    return m.group("dataset"), m.group("subpath")

def rebuild_path(set_type: str, dataset: str, subpath: str, sample_base: str, feature: str):
    if feature == "stat":
        return f"{BASE_PATHS['stat']}/{set_type}/{dataset}/{subpath}/{sample_base}_stat.csv"
    if feature == "seq":
        return f"{BASE_PATHS['seq']}/{set_type}/{dataset}/{subpath}/{sample_base}_seq_embed.npy"
    if feature == "raw":
        return f"{BASE_PATHS['raw']}/{set_type}/{dataset}/{subpath}/{sample_base}_raw_embed.npy"
    raise ValueError(f"Unknown feature type: {feature}")

def process_csv(input_file: str, output_dir: str):
    print(f"[INFO] Processing: {input_file}")
    df = pd.read_csv(input_file)

    # 统一列名（兼容老CSV）
    df = df.rename(columns={
        "file_path": "stat_feature_path",
        "seq_feature_path": "seq_embed_feature_path",
        "raw_feature_path": "raw_embed_feature_path",
    })

    new_stat, new_seq, new_raw = [], [], []
    for _, row in df.iterrows():
        set_type = row["set_type"]               # train/test/unknown
        sample_base = row["sample_base"]

        dataset, subpath = infer_dataset_and_subpath(row["stat_feature_path"])
        new_stat.append(rebuild_path(set_type, dataset, subpath, sample_base, "stat"))
        new_seq.append(rebuild_path(set_type, dataset, subpath, sample_base, "seq"))
        new_raw.append(rebuild_path(set_type, dataset, subpath, sample_base, "raw"))

    df["stat_feature_path"]      = new_stat
    df["seq_embed_feature_path"] = new_seq
    df["raw_embed_feature_path"] = new_raw

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_file))
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")

def main():
    # 你可以按需改成 uk/us/cicIoT2022
    dataset_name = "uk"
    in_dir  = f"{BASE_DIR}/8_split_sample_info/{dataset_name}"
    out_dir = f"{BASE_DIR}/11_multitask_training/{dataset_name}"

    for name in (f"3_{dataset_name}_train.csv", f"3_{dataset_name}_test.csv", f"3_{dataset_name}_unknown.csv", f"3_{dataset_name}_full_split.csv"):
        in_file = os.path.join(in_dir, name)
        if os.path.exists(in_file):
            process_csv(in_file, out_dir)
        else:
            print(f"[WARN] Not found: {in_file}")

if __name__ == "__main__":
    main()
