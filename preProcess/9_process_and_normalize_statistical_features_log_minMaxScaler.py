"""
（当前暂未使用）
功能说明：
本脚本用于对设备网络流量统计特征进行归一化处理，并保持原始目录结构保存到新的输出目录。
- 读取 train, test, unknown 三个 CSV 文件（uk_train.csv, uk_test.csv, uk_unknown.csv），获取每个样本的统计特征 CSV 文件路径。
- 对每个 CSV 文件的前 30 维统计特征进行归一化处理（先使用log变换处理极端值，再使用 MinMaxScaler），并处理 NaN 值。
- 检测并记录超出 [0, 1] 范围的特征，裁剪到 [0, 1]。
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

# 定义需要 Log 变换的特征
LOG_TRANSFORM_FEATURES = [
    'packet_count', 'total_bytes', 'payload_bytes_total',
    'up_pkt_count', 'down_pkt_count', 'up_bytes', 'down_bytes',
    'up_down_pkt_ratio', 'up_down_byte_ratio', 'pkt_rate', 'burst_count', 'tcp_count', 'udp_count'
]

# 定义元信息列
META_COLUMNS = ['is_behavior', 'type_label', 'brand_label', 'device_label', 'sample_file']

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

def process_dataset(csv_path, set_type, scaler, train_medians, train_min, train_max, output_base):
    """
    处理数据集，进行 Log 变换、归一化，并保存结果。
    
    Args:
        csv_path: 数据集 CSV 文件路径。
        set_type: 数据集类型（train, test, unknown）。
        scaler: 预拟合的 MinMaxScaler。
        train_medians: 训练集特征中位数。
        train_min: 训练集特征最小值。
        train_max: 训练集特征最大值。
        output_base: 输出根目录。
    """
    print(f"开始处理 {set_type} 数据集：{csv_path}")
    df = pd.read_csv(csv_path)
    print(f"读取 {set_type} 数据集，包含 {len(df)} 个样本")

    log_file = os.path.join(output_base, f"{set_type}_outliers.log")
    stats_file = os.path.join(output_base, f"{set_type}_outlier_stats.txt")
    with open(log_file, 'w') as f:
        f.write(f"超出 [0, 1] 范围的样本记录 - {set_type}\n")

    out_of_range_count = {col: 0 for col in FEATURE_COLUMNS}
    total_samples = 0
    outlier_data = []

    for idx, row in df.iterrows():
        file_path = row['file_path']
        total_samples += 1
        try:
            sample_df = pd.read_csv(file_path)
            features = sample_df[FEATURE_COLUMNS].values
            features = handle_nan(features, train_medians)

            # 保存原始特征以记录超出范围信息
            original_features = features.copy()

            # 应用 Log 变换
            for i, col in enumerate(FEATURE_COLUMNS):
                if col in LOG_TRANSFORM_FEATURES:
                    features[:, i] = np.log1p(features[:, i])

            # 使用 MinMaxScaler 进行归一化并裁剪到 [0, 1]
            features_normalized = np.clip(scaler.transform(features), 0, 1)

            out_of_range = False
            for i, col in enumerate(FEATURE_COLUMNS):
                col_values = features_normalized[:, i]
                if np.any(col_values < -0.0001) or np.any(col_values > 1.0001):
                    out_of_range = True
                    out_of_range_count[col] += 1
                    min_val = np.min(col_values)
                    max_val = np.max(col_values)
                    with open(log_file, 'a') as f:
                        f.write(f"样本 {file_path} 特征 {col} 超出范围：归一化值 最小 = {min_val}, 最大 = {max_val}, 原始值 = {original_features[:, i]}\n")
                    outlier_data.append({
                        'file_path': file_path,
                        'feature': col,
                        'normalized_value': col_values[0],
                        'original_value': original_features[:, i][0]
                    })
                else:
                    min_val = np.min(col_values)
                    max_val = np.max(col_values)
                    with open(log_file, 'a') as f:
                        f.write(f"样本 {file_path} 特征 {col} 正常：归一化值 最小 = {min_val}, 最大 = {max_val}, 原始值 = {original_features[:, i]}\n")

            if out_of_range:
                print(f"警告：样本 {file_path} 的归一化特征超出 [0, 1] 范围！已记录到 {log_file}")
            else:
                print(f"样本 {file_path} 的归一化特征全部在 [0, 1] 范围内")

            # 创建输出 DataFrame，仅包含 30 个统计特征 + is_behavior + 元信息
            new_df = pd.DataFrame(features_normalized, columns=FEATURE_COLUMNS)
            new_df['is_behavior'] = sample_df['is_behavior']
            new_df[['type_label', 'brand_label', 'device_label', 'sample_file']] = \
                sample_df[['type_label', 'brand_label', 'device_label', 'sample_file']]

            # 确保列顺序正确
            output_columns = FEATURE_COLUMNS + ['is_behavior'] + ['type_label', 'brand_label', 'device_label', 'sample_file']
            new_df = new_df[output_columns]

            relative_path = file_path.split('/7_cleaned_features/7_cleaned_statistical_feature/us/')[-1]
            output_path = os.path.join(output_base, set_type, 'uk', relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_df.to_csv(output_path, index=False)
            print(f"已保存归一化后的样本：{output_path}")
        except Exception as e:
            print(f"处理样本 {file_path} 时出错：{e}")

    if outlier_data:
        pd.DataFrame(outlier_data).to_csv(os.path.join(output_base, f"{set_type}_outlier_values.csv"), index=False)
        print(f"已保存超出范围的原始值到 {set_type}_outlier_values.csv")

    with open(stats_file, 'w') as f:
        f.write(f"{set_type} 数据集超出范围统计：\n")
        f.write(f"总样本数：{total_samples}\n")
        f.write(f"超出范围样本数：{len(outlier_data)}\n")
        for col, count in out_of_range_count.items():
            f.write(f"特征 {col} 超出范围样本比例：{count / total_samples:.2%}\n")
    print(f"已保存统计信息到 {stats_file}")

def main():
    """
    主函数，处理 train, test, unknown 数据集的归一化。
    """
    # 输入路径
    csv_path_train = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/us/us_train.csv"
    csv_path_test = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/us/us_test.csv"
    csv_path_unknown = "/home/hyj/unknownDeviceIdentification/dataset/8_split_sample_info/us/us_unknown.csv"
    output_base = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_statistical_embeddings"

    # 确保输出根目录存在
    os.makedirs(output_base, exist_ok=True)

    print("开始归一化处理...")
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

    train_medians = []
    for i, col in enumerate(FEATURE_COLUMNS):
        nan_count = np.sum(np.isnan(train_features[:, i]))
        if nan_count > 0:
            print(f"训练集 {col} 列有 {nan_count} 个 NaN 值")
        if col in COUNT_FEATURES:
            train_medians.append(0)
        else:
            train_medians.append(np.nanmedian(train_features[:, i]))
    train_features = handle_nan(train_features, train_medians)

    # 计算训练集的 min 和 max（在 Log 变换前）
    train_min = np.nanmin(train_features, axis=0)
    train_max = np.nanmax(train_features, axis=0)

    # 对训练集应用 Log 变换
    for i, col in enumerate(FEATURE_COLUMNS):
        if col in LOG_TRANSFORM_FEATURES:
            train_features[:, i] = np.log1p(train_features[:, i])

    scaler = MinMaxScaler()
    scaler.fit(train_features)
    print("MinMaxScaler 拟合完成")

    for set_type, csv_path in [('train', csv_path_train), ('test', csv_path_test), ('unknown', csv_path_unknown)]:
        process_dataset(csv_path, set_type, scaler, train_medians, train_min, train_max, output_base)

    print("所有数据集归一化处理完成！")

if __name__ == "__main__":
    main()