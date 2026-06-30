# -- coding: utf-8 --

"""
功能说明：
本脚本用于从大量 .pcap 文件生成的 CSV 文件中，筛选符合特定协议的数据包，并删除不符合条件的数据包。
处理步骤如下：

1. 复制文件：
    - 遍历源目录，将源目录下的所有 CSV 文件及子目录结构复制到目标目录中。

2. 过滤 CSV 文件：
    - 遍历目标目录下的每个 CSV 文件，根据以下规则筛选保留的数据包：
        a. TLS 应用数据包（tls.record.content_type == 23）
        b. 有效 TCP 数据包（tcp.len ≠ 0 且不包含 tls 协议）
        c. TLS 握手数据包（frame.protocols 包含 tls 且 tls.record.content_type 为空）
        d. UDP 数据包（frame.protocols 包含 udp）
    - 对于过滤后为空的 CSV 文件，直接删除该文件。

3. 清理目录：
    - 对每个文件夹检查过滤后的 CSV 文件数量，若少于 3 个，则删除该文件夹；否则保留该文件夹。

4. 输出进度：
    - 在终端实时输出每个文件的处理进度和最终统计结果。

步骤概述：
- 遍历指定目录中的 `.csv` 文件并应用过滤规则。
- 将每个文件的结果保存为新 CSV 文件，并在无有效数据包时删除文件。
- 保留符合条件的文件夹和 CSV 文件，删除不合格的目录和文件。
"""

import os
import shutil
import pandas as pd


def copy_directory(src_folder, dst_folder):
    """
    复制整个 CSV 文件目录及其内容到目标目录
    - 使用 os.walk 遍历源目录中的文件和子目录
    - 保证目标目录结构完整，复制每个文件
    """
    for root, dirs, files in os.walk(src_folder):
        # 计算相对路径以保留目录结构
        rel_path = os.path.relpath(root, src_folder)
        dst_path = os.path.join(dst_folder, rel_path)

        # 创建目标目录结构
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # 复制文件
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")


def filter_csv_files(input_dir):
    """
    过滤指定目录下的所有 CSV 文件，删除无关协议的数据包，并保存过滤后的数据。
    过滤规则：
    - TLS 应用数据包
    - 有效 TCP 数据包
    - TLS 握手数据包
    - UDP 数据包
    """
    processed_files_count = 0

    for root, dirs, files in os.walk(input_dir, topdown=False):
        # 获取本目录中的CSV文件
        csv_files = [f for f in files if f.endswith('.csv')]

        for name in csv_files:
            filename = os.path.join(root, name)
            print(f"Processing file: {filename}")

            # 读取 CSV 文件
            df = pd.read_csv(filename, encoding='utf-8', header=0, keep_default_na=False)
            pd.set_option('expand_frame_repr', False)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            # 确保字段类型正确
            df['tcp.len'] = pd.to_numeric(df['tcp.len'], errors='coerce')

            # 定义过滤条件
            condition1 = (df['tls.record.content_type'] == '23') | (df['tls.record.content_type'] == '23.0')  # TLS 数据包
            condition2 = (df['tcp.len'] != 0) & (df['tcp.len'] != '0.0') & (
                ~df['frame.protocols'].str.contains('tls'))  # 有效 TCP 数据包
            condition3 = df['frame.protocols'].str.contains('tls') & (df['tls.record.content_type'] == '')  # TLS 握手数据包
            condition_udp = df['frame.protocols'].str.contains('udp', na=False)  # UDP 数据包

            # 保留协议：DNS、mDNS、NTP
            # 排除无关协议：ARP, icmp, igmp, smtp, nbns, ftp, nd, smb, dhcp, ssdp等
            condition_no_relevant_protocols = ~df['frame.protocols'].str.contains('arp|icmp|igmp|smtp|nbns|ftp|nd|smb|dhcp|ssdp|gquic|stun|llc|eapol|wg', na=False)

            # 合并过滤条件（包括排除无关协议）
            result = df[(condition1 | condition2 | condition3 | condition_udp) & condition_no_relevant_protocols]
            # 合并过滤条件（不排除无关协议）
            # result = df[(condition1 | condition2 | condition3 | condition_udp)]

            # 删除空 CSV 文件或保存过滤后的数据
            try:
                if result.empty:
                    os.remove(filename)
                    print(f"Deleted empty file: {filename}")
                else:
                    result.to_csv(filename, index=False)
                    print(f"Saved filtered file: {filename}")
            except Exception as e:
                print(f"❌ 错误处理文件 {filename}: {e}")

            processed_files_count += 1
            print(f"Processed files count: {processed_files_count}")

        # 检查每个文件夹内剩余的 CSV 文件数量，若少于 3 个，则删除该文件夹
        if csv_files:
            remaining_csv = [f for f in os.listdir(root) if f.endswith('.csv')]
            if len(remaining_csv) < 3:
                try:
                    shutil.rmtree(root)
                    print(f"Deleted session folder (too few CSV files): {root}")
                except Exception as e:
                    print(f"❌ 删除目录 {root} 时出错: {e}")
            else:
                print(f"Retained session folder: {root}")

    print(f"Total processed CSV files: {processed_files_count}")


def main():
    """
    主函数，定义源目录和目标目录，依次执行文件复制和过滤操作
    """
    # 定义源文件夹路径和目标文件夹路径
    src_folder = '/home/hyj/unknownDeviceIdentification/dataset/test/csv'  # 原始 CSV 文件目录
    dst_folder = '/home/hyj/unknownDeviceIdentification/dataset/test/csv_filter'  # 复制后的 CSV 文件目录

    # 复制文件目录
    copy_directory(src_folder, dst_folder)

    # 对复制后的目录进行 CSV 文件过滤
    filter_csv_files(dst_folder)


if __name__ == "__main__":
    main()
