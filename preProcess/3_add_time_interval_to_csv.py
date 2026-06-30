"""
对于得到的csv文件，添加时间间隔之后保存在新路径，再进行后续处理。相比于process3更加优化简洁，两个代码功能一样，都可用
实现的功能：
1、遍历源文件夹中的 CSV 文件。
2、复制 CSV 文件到目标文件夹，同时保持原始目录结构。
3、为每个 CSV 文件计算时间间隔，并保存结果到新的路径中。

1. 遍历与复制 CSV 文件
    遍历源目录中的所有子目录，找到 .csv 文件。
    将 .csv 文件复制到目标目录，同时保持原始目录结构。
2. 计算并添加时间间隔
    读取 CSV 文件，检查是否包含 frame.time_epoch 列。
    如果存在，将其转换为浮点数，并计算相邻数据包的时间间隔。
    将计算结果添加为新列 time_interval，其中第一个数据包的时间间隔为 0。
3. 保存处理后的文件
    处理完成后，将结果保存到目标目录。
4. 进度和统计信息
    在终端打印每个文件的处理路径、时间间隔计算状态、保存路径。
    最后统计并输出总共处理的文件数量。
"""

import os
import pandas as pd


def process_csv_file(file_path, new_file_path):
    # 打印正在处理的文件路径
    print(f"Processing file: {file_path}")

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 检查 'frame.time_epoch' 列是否存在
    if 'frame.time_epoch' in df.columns:
        # 转换为浮点型并计算时间间隔
        df['frame.time_epoch'] = df['frame.time_epoch'].astype(float)
        df['time_interval'] = df['frame.time_epoch'].diff().fillna(0)  # 计算时间间隔
        print(f"Added 'time_interval' column for {file_path}")
    else:
        print(f"'frame.time_epoch' column not found in {file_path}")

    # 保存处理后的CSV文件到新的路径
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    df.to_csv(new_file_path, index=False)
    print(f"Saved processed file to: {new_file_path}\n")


def copy_and_process_csv(root_folder, new_root_folder):
    file_count = 0  # 用于统计处理的文件数量

    # 遍历总文件夹
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                # 构建新的文件路径
                relative_path = os.path.relpath(file_path, root_folder)
                new_file_path = os.path.join(new_root_folder, relative_path)

                # 打印复制文件的路径信息
                print(f"Copying and processing {file_path} to {new_file_path}")

                # 处理CSV文件并保存到新路径
                process_csv_file(file_path, new_file_path)

                # 文件处理完成后增加计数
                file_count += 1

    # 打印总共处理的文件数
    print(f"Total CSV files processed: {file_count}")


if __name__ == "__main__":
    root_folder = "/home/hyj/unknownDeviceIdentification/dataset/2_csv/cicIoT2022"  # 原始总文件夹路径
    new_root_folder = "/home/hyj/unknownDeviceIdentification/dataset/3_csvAddTime/cicIoT2022"  # 新的总路径

    print(f"Starting to process files from {root_folder} to {new_root_folder}\n")
    copy_and_process_csv(root_folder, new_root_folder)
    print("Processing completed!")
