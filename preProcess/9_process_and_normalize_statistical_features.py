"""
（对uk数据集使用了）
功能说明：
本脚本用于对设备网络流量统计特征进行归一化处理，并保持原始目录结构保存到新的输出目录。
- 读取 train, test, unknown 三个 CSV 文件（uk_train.csv, uk_test.csv, uk_unknown.csv），获取每个样本的统计特征 CSV 文件路径。
- 对每个 CSV 文件的前 30 维统计特征进行归一化处理（使用 MinMaxScaler），并处理 NaN 值。
- 保持原始目录结构，将处理后的 CSV 文件保存到指定的输出目录（train, test, unknown 下的 uk 子目录）。
- NaN 处理策略：
  - 计数类特征（如 packet_count）填充为 0。
  - 统计类特征（如 avg_iat）填充为训练集该特征的中位数。
- 确保归一化后特征值在 [0, 1] 范围内，保留原始 CSV 的后 5 列元信息。

输入：
- train, test, unknown CSV 文件路径，包含 file_path 等信息。
- 统计特征 CSV 文件，包含 30 维特征 + 5 列元信息。

输出：
- 归一化后的 CSV 文件，保存到 /home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings/{set_type}/uk/ 下的对应目录。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# 定义特征列（前 30 维统计特征）
FEATURE_COLUMNS = [
    'packet_count', 'avg_pkt_len', 'std_pkt_len', 'max_pkt_len', 'min_pkt_len',
    'total_bytes', 'payload_bytes_total', 'payload_bytes_ratio', 'up_pkt_count',
    'down_pkt_count', 'up_bytes', 'down_bytes', 'up_down_pkt_ratio',
    'up_down_byte_ratio', 'udp_ratio', 'avg_iat', 'std_iat', 'min_iat',
    'max_iat', 'pkt_rate', 'pkt_interval_entropy', 'burst_count',
    'heartbeat_period_fft', 'active_ratio', 'burstiness', 'tcp_count',
    'udp_count', 'session_count', 'unique_dst_ports', 'entropy_pkt_size'
]

# 定义计数类特征（NaN 填充为 0）
COUNT_FEATURES = [
    'packet_count', 'total_bytes', 'up_pkt_count', 'down_pkt_count',
    'up_bytes', 'down_bytes', 'tcp_count', 'udp_count', 'session_count',
    'unique_dst_ports', 'burst_count'
]

def handle_nan(features, train_medians=None):
    """
    处理特征中的 NaN 值。
    - 计数类特征填充为 0。
    - 统计类特征填充为训练集的中位数（若提供），否则用当前特征的中位数。
    
    Args:
        features: numpy 数组，形状为 (n_samples, n_features)。
        train_medians: 训练集每列的中位数（用于 test/unknown）。
    
    Returns:
        处理后的特征数组。
    """
    features = features.copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        if np.any(np.isnan(features[:, i])):
            print(f"检测到 {col} 列存在 NaN，进行处理...")
            if col in COUNT_FEATURES:
                features[:, i] = np.nan_to_num(features[:, i], nan=0)
            else:
                median = train_medians[i] if train_medians is not None else np.nanmedian(features[:, i])
                features[:, i] = np.nan_to_num(features[:, i], nan=median)
    return features

def process_dataset(csv_path, set_type, scaler, train_medians, output_base):
    """
    处理单个数据集（train/test/unknown）的 CSV 文件。
    
    Args:
        csv_path: 包含 file_path 的 CSV 文件路径。
        set_type: 数据集类型（train/test/unknown）。
        scaler: 已拟合的 MinMaxScaler。
        train_medians: 训练集每列的中位数。
        output_base: 输出基础目录。
    """
    print(f"开始处理 {set_type} 数据集：{csv_path}")
    df = pd.read_csv(csv_path)
    print(f"读取 {set_type} 数据集，包含 {len(df)} 个样本")

    for idx, row in df.iterrows():
        file_path = row['file_path']
        try:
            # 读取样本 CSV
            sample_df = pd.read_csv(file_path)
            features = sample_df[FEATURE_COLUMNS].values

            # 检查 NaN
            if np.any(np.isnan(features)):
                print(f"样本 {file_path} 包含 NaN，进行处理...")
            features = handle_nan(features, train_medians)

            # 归一化
            features_normalized = scaler.transform(features)
            
            # 验证归一化结果
            if not (np.all(features_normalized >= 0) and np.all(features_normalized <= 1)):
                print(f"警告：样本 {file_path} 的归一化特征超出 [0, 1] 范围！")

            # 构造新的 DataFrame
            new_df = pd.DataFrame(features_normalized, columns=FEATURE_COLUMNS)
            new_df[['is_behavior', 'type_label', 'brand_label', 'device_label', 'sample_file']] = \
                sample_df[['is_behavior', 'type_label', 'brand_label', 'device_label', 'sample_file']]

            # 构造输出路径
            relative_path = file_path.split('/7_cleaned_features/7_cleaned_statistical_feature/uk/')[-1]
            output_path = os.path.join(output_base, set_type, 'uk', relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存新的 CSV
            new_df.to_csv(output_path, index=False)
            print(f"已保存归一化后的样本：{output_path}")

        except Exception as e:
            print(f"处理样本 {file_path} 时出错：{e}")

def main():
    # 输入路径
    csv_path_train = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk/uk_train.csv"
    csv_path_test = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk/uk_test.csv"
    csv_path_unknown = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/uk/uk_unknown.csv"
    output_base = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings"

    # 确保输出根目录存在
    os.makedirs(output_base, exist_ok=True)

    print("开始归一化处理...")

    # 读取训练集以拟合 MinMaxScaler
    print("读取训练集以拟合 MinMaxScaler...")
    train_df = pd.read_csv(csv_path_train)
    train_features = []
    for file_path in train_df['file_path']:
        try:
            df = pd.read_csv(file_path)
            train_features.append(df[FEATURE_COLUMNS].values)
        except Exception as e:
            print(f"读取训练样本 {file_path} 时出错：{e}")
            continue
    train_features = np.vstack(train_features)
    print(f"训练集特征形状：{train_features.shape}")

    # 处理训练集中的 NaN
    train_medians = []
    for i, col in enumerate(FEATURE_COLUMNS):
        nan_count = np.sum(np.isnan(train_features[:, i]))
        if nan_count > 0:
            print(f"训练集 {col} 列有 {nan_count} 个 NaN 值")
        if col in COUNT_FEATURES:
            train_medians.append(0)
        else:
            train_medians.append(np.nanmedian(train_features[:, i]))
    train_features = handle_nan(train_features)

    # 拟合 MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_features)
    print("MinMaxScaler 拟合完成")

    # 处理每个数据集
    for set_type, csv_path in [('train', csv_path_train), ('test', csv_path_test), ('unknown', csv_path_unknown)]:
        process_dataset(csv_path, set_type, scaler, train_medians, output_base)

    print("所有数据集归一化处理完成！")

if __name__ == "__main__":
    main()