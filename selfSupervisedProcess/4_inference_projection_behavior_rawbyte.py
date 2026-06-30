"""
📌 脚本功能说明：
该脚本使用训练好的投影模型（projection_behavior_rawbyte.pt），
对行为流量的原始字节嵌入向量（128维）进行 64 维对比学习表示生成。

输入路径：
    /home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_rawbyte_embeddings/uk

输出路径：
    /home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings/uk

保持与输入路径结构一致，文件名形如 xxx_raw_embed.npy → 64维向量。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# =======================
# 参数配置
# =======================
INPUT_ROOT = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_rawbyte_embeddings/unknown/cicIoT2022"
OUTPUT_ROOT = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings/unknown/cicIoT2022"
MODEL_PATH = "/home/hyj/unknownDeviceIdentification/selfSupervisedProcess/3rd/cicIoT2022_projection_behavior_rawbyte.pt"
EMBEDDING_DIM = 128
PROJECTED_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 投影网络结构
# =======================
class ProjectionMLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# =======================
# 嵌入生成函数
# =======================

def generate_embeddings():
    model = ProjectionMLP(EMBEDDING_DIM, PROJECTED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("🚀 开始生成行为流量的原始字节 64维嵌入向量...")

    # 收集所有输入文件路径
    all_input_paths = []
    for device in os.listdir(INPUT_ROOT):
        device_dir = os.path.join(INPUT_ROOT, device, "activity")
        if not os.path.isdir(device_dir):
            continue
        for behavior in os.listdir(device_dir):
            behavior_dir = os.path.join(device_dir, behavior)
            for file in os.listdir(behavior_dir):
                if file.endswith("_raw_embed.npy"):
                    input_path = os.path.join(behavior_dir, file)
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


# =======================
# 主函数入口
# =======================
if __name__ == "__main__":
    generate_embeddings()
