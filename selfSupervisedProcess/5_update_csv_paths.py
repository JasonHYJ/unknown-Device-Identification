import pandas as pd
import os

"""
功能描述
目的：处理 uk_train.csv、uk_test.csv 和 uk_unknown.csv，修改指定列名并更新特征路径，保存到目标目录。
输入：三个 CSV 文件，包含原始特征路径和样本信息。
输出：更新后的 CSV 文件，保存到 /home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk。
主要步骤：
读取每个 CSV 文件。
重命名列：file_path → stat_feature_path, seq_feature_path → seq_embed_feature_path, raw_feature_path → raw_embed_feature_path。
更新路径：
统计特征路径：指向 10_contrastive_statistical_embeddings/{set_type}/uk/.../_stat.csv。
序列特征路径：指向 10_contrastive_sequence_embeddings/{set_type}/uk/.../_seq_embed.npy。
原始字节特征路径：指向 10_contrastive_rawbyte_embeddings/{set_type}/uk/.../_raw_embed.npy。
保存更新后的 CSV 文件到指定目录。
错误处理：检查文件存在性，捕获读写错误并打印提示信息。
日志：打印每个步骤的状态信息，便于跟踪执行过程。
"""

def update_path(row, feature_type, base_paths):
    """Update file paths for statistical, sequence, or raw byte features."""
    set_type = row["set_type"]  # train/test/unknown
    sample_base = row["sample_base"]  # e.g., ring-doorbell__2019-04-28_idle__00082
    # Extract sub-path (e.g., ring-doorbell/idle/2019-04-28_idle) from original stat_feature_path
    sub_path = row["stat_feature_path"].split("/uk/")[1].rsplit("/", 1)[0]
    
    if feature_type == "stat":
        return f"{base_paths['stat']}/{set_type}/uk/{sub_path}/{sample_base}_stat.csv"
    elif feature_type == "seq":
        return f"{base_paths['seq']}/{set_type}/uk/{sub_path}/{sample_base}_seq_embed.npy"
    elif feature_type == "raw":
        return f"{base_paths['raw']}/{set_type}/uk/{sub_path}/{sample_base}_raw_embed.npy"
    return None

def process_csv(input_file, output_dir, base_paths):
    """Process a single CSV file: rename columns and update feature paths."""
    print(f"Processing file: {input_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # Rename columns
    df = df.rename(columns={
        "file_path": "stat_feature_path",
        "seq_feature_path": "seq_embed_feature_path",
        "raw_feature_path": "raw_embed_feature_path"
    })
    print(f"Renamed columns in {input_file}")
    
    # Update paths for each feature type
    df["stat_feature_path"] = df.apply(lambda row: update_path(row, "stat", base_paths), axis=1)
    df["seq_embed_feature_path"] = df.apply(lambda row: update_path(row, "seq", base_paths), axis=1)
    df["raw_embed_feature_path"] = df.apply(lambda row: update_path(row, "raw", base_paths), axis=1)
    print(f"Updated feature paths in {input_file}")
    
    # Save updated CSV
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    try:
        df.to_csv(output_file, index=False)
        print(f"Saved updated file to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

def main():
    """Main function to process uk_train.csv, uk_test.csv, and uk_unknown.csv."""
    # Define input and output paths
    input_dir = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk"  # Adjust to your actual input directory
    output_dir = "/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")
    
    # Define base paths for new feature directories
    base_paths = {
        "stat": "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings",
        "seq": "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings",
        "raw": "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings"
    }
    
    # List of input CSV files
    input_files = [
        "3_uk_train.csv",
        "3_uk_test.csv",
        "3_uk_unknown.csv"
    ]
    
    # Process each CSV file
    for file_name in input_files:
        input_file = os.path.join(input_dir, file_name)
        if os.path.exists(input_file):
            process_csv(input_file, output_dir, base_paths)
        else:
            print(f"Input file not found: {input_file}")

if __name__ == "__main__":
    print("Starting CSV processing...")
    main()
    print("CSV processing completed.")