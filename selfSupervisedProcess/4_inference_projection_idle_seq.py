# inference_projection_idle_seq.py
"""
📌 功能说明：
本脚本使用已训练好的 projection_idle_seq.pt 模型，
将 UK 数据集中“闲时流量的序列嵌入向量（128维）”映射为 64 维向量，
并保存至指定目录（结构保持一致）。
"""

import os
import numpy as np
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ==========================
# 📌 配置参数
# ==========================
INPUT_ROOT = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_sequence_embeddings/unknown/cicIoT2022"
OUTPUT_ROOT = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_sequence_embeddings/unknown/cicIoT2022"
MODEL_PATH = "/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_projection_idle_seq.pt"
EMBEDDING_DIM = 128
PROJECTED_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Step 1: 模型结构定义
# ==========================
class ProjectionMLP(torch.nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==========================
# Step 2: 嵌入生成函数
# ==========================\n
def generate_embeddings():
    model = ProjectionMLP(EMBEDDING_DIM, PROJECTED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("🚀 开始生成闲时流量的序列 64维嵌入向量...")

    # 收集所有输入文件路径
    all_input_paths = []
    for device in os.listdir(INPUT_ROOT):
        device_dir = os.path.join(INPUT_ROOT, device, "idle")
        if not os.path.isdir(device_dir):
            continue
        for group in os.listdir(device_dir):
            group_dir = os.path.join(device_dir, group)
            for file in os.listdir(group_dir):
                if file.endswith("_seq_embed.npy"):
                    input_path = os.path.join(group_dir, file)
                    all_input_paths.append(input_path)

    print(f"📦 待处理文件数: {len(all_input_paths)}")

    # tqdm 进度条处理每个文件
    for input_path in tqdm(all_input_paths, desc="📤 生成嵌入中"):
        vec = np.load(input_path)
        input_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            projected = model(input_tensor).squeeze(0).cpu().numpy()

        rel_path = os.path.relpath(input_path, INPUT_ROOT)
        output_path = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, projected)

    print(f"✅ 嵌入生成完成，输出路径：{OUTPUT_ROOT}")

# ==========================
# 主程序入口
# ==========================
if __name__ == "__main__":
    generate_embeddings()
