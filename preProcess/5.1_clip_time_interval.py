import os
import pandas as pd

"""
功能描述：
本脚本用于批量处理 IoT 样本数据中的 CSV 文件，对每个 CSV 文件中的 time_interval 列进行异常值剪裁（上限设为 1 秒），
然后将处理后的 CSV 文件保存到一个新的输出目录中。新目录结构与输入目录完全一致，支持行为流量和闲时流量两种模式。

使用说明：
- 将 input_root 设置为原始样本根目录，例如 "/path/to/uk"
- 将 output_root 设置为目标输出根目录，例如 "/path/to/out_uk"
- 运行脚本后，所有处理后的 CSV 文件将自动保存在对应路径下
"""


def clip_and_save(csv_path, output_csv_path, clip_threshold=1.0):
    """
    剪裁 CSV 中的 time_interval 列并保存到新路径。
    """
    try:
        df = pd.read_csv(csv_path)
        if "time_interval" not in df.columns:
            print(f"[跳过] 缺少 time_interval 列: {csv_path}")
            return

        original_max = df["time_interval"].max()
        df["time_interval"] = df["time_interval"].clip(upper=clip_threshold)
        new_max = df["time_interval"].max()

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)

        print(f"[处理完成] {output_csv_path} | 原最大: {original_max:.2f}, 剪裁后最大: {new_max:.2f}")

    except Exception as e:
        print(f"[错误] 处理文件失败 {csv_path}：{e}")


def main():
    # ==== 修改为你实际的输入输出路径 ====
    input_root = "/home/hyj/unknownDeviceIdentification/dataset/test/csv_filter"
    output_root = "/home/hyj/unknownDeviceIdentification/dataset/test/csv_clip_time_interval"
    clip_threshold = 1.0  # 秒

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(".csv"):
                input_csv_path = os.path.join(root, file)

                # 构建输出路径：将 input_root 替换为 output_root
                relative_path = os.path.relpath(input_csv_path, input_root)
                output_csv_path = os.path.join(output_root, relative_path)

                # 剪裁并保存
                clip_and_save(input_csv_path, output_csv_path, clip_threshold)


if __name__ == "__main__":
    main()
