# xiaorong_optimized_train_multitask_classifier.py
# 📆 多任务 IoT 设备识别 — 消融训练脚本（F0–F12） + 融合策略对比（3.2）

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

from xiaorong_optimized_multimodal_dataset import MultiModalIoTDataset

def safe_collate(batch):
    """丢弃 None 批次项，避免 DataLoader 在读取失败时崩溃。"""
    batch = [b for b in batch if b is not None and all(x is not None for x in b)]
    if not batch:
        return None
    return default_collate(batch)

# ===================== 参数区 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # 你可以改成实际可用的 GPU ID

# —— 数据路径：按需切换不同数据集（uk/us/cicIoT2022 等）——
DATA_ROOT = "/home/hyj/unknownDeviceIdentification/dataset"         # 数据集路径
TRAIN_CSV = f"{DATA_ROOT}/11_multitask_training/uk/3_uk_train.csv"   # train CSV
TEST_CSV  = f"{DATA_ROOT}/11_multitask_training/uk/3_uk_test.csv"    # test  CSV
LABEL_DIR = f"{DATA_ROOT}/11_multitask_training/uk"                  # type/brand/device 的 json

# —— 输出目录（已按你的要求修改）——
OUT_DIR   = f"{DATA_ROOT}/12_expB_outputs/uk"
MODEL_DIR = f"{OUT_DIR}/3_2_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# —— 训练设置 —— 
BATCH_SIZE     = 64
NUM_EPOCHS     = 40
LEARNING_RATE  = 1e-3
MAX_GRAD_NORM  = 2.0
NUM_WORKERS    = 4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 多任务损失权重（可按需调节）——
ALPHA_TYPE   = 1.0
ALPHA_BRAND  = 1.0
ALPHA_DEVICE = 1.0

# —— 消融组合（完整定义）——
ABLATION_MAP = {
    "F0":  "Stat",
    "F1":  "Seq-Embed",
    "F2":  "Raw-Embed",
    "F3":  "Seq-Embed+Raw-Embed",
    "F4":  "Stat+Seq-Embed",
    "F5":  "Stat+Raw-Embed",
    "F6":  "Stat+Seq-Embed+Raw-Embed",
    "F7":  "Seq-CL",
    "F8":  "Raw-CL",
    "F9":  "Seq-CL+Raw-CL",
    "F10": "Stat+Seq-CL",
    "F11": "Stat+Raw-CL",
    "F12": "Stat+Seq-CL+Raw-CL",
}

# —— 跑哪些组 —— 
# 3.1 特征消融：放开全部；3.2 融合对比：推荐只跑 F12
# RUN_GROUPS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"]
RUN_GROUPS = ["F12"]   # ← 做 3.2 时建议这样设
SEEDS      = [0]       # 如需多次重复：例如 [0,1,2]

# —— 融合策略（3.2 关键开关）——
# gate:       σ(Linear(is_behavior)) 条件融合（Baseline）
# concat:     [idle, beh] 拼接(256) → Linear(256→128)
# idleonly:   只用 idle（等价 gate=0）
# alpha:      全局可学习 α ∈ (0,1)：α*beh + (1-α)*idle
FUSION_LIST = ["gate", "concat", "idleonly", "alpha"]  # 做 3.2 用
# FUSION_LIST = ["gate"]  # 只做 3.1 时可以只保留 gate

# ===================== 模型定义 =====================
class MultiTaskClassifier(nn.Module):
    """
    多任务分类器：
    输入：x ∈ R^{B,287} = [31(stat), 128(idle_embed), 128(behavior_embed)]
    融合策略（fusion）：
      - gate:    g = σ(Linear(is_behavior))，embed = g*beh + (1-g)*idle
      - concat:  cat= → Linear(256→128)
      - idleonly:embed = idle
      - alpha:   α可学习（全局标量），embed = α*beh + (1-α)*idle
    编码：FC(159→256) → TransformerEncoder(2层, 4头)
    输出：三头线性层，分别预测 type/brand/device
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_type: int, num_brand: int, num_device: int,
                 fusion: str = "gate"):
        super().__init__()
        self.fusion = fusion.lower()
        # 各策略所需层的注册
        if self.fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(1,1), nn.Sigmoid())
        elif self.fusion == "concat":
            self.concat_reduce = nn.Linear(256, 128)
        elif self.fusion == "alpha":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self._sigm = nn.Sigmoid()
        elif self.fusion == "idleonly":
            pass
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # 特征编码
        self.fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 三个任务头
        self.cls_type   = nn.Linear(hidden_dim, num_type)
        self.cls_brand  = nn.Linear(hidden_dim, num_brand)
        self.cls_device = nn.Linear(hidden_dim, num_device)

    def fuse(self, stats, idle_embed, beh_embed):
        """根据策略融合 128 维 idle/behavior 嵌入，返回 128 维"""
        if self.fusion == "gate":
            is_behavior = stats[:, 30:31]              # (B,1)
            gate = self.gate(is_behavior)              # (B,1)
            return gate*beh_embed + (1-gate)*idle_embed
        elif self.fusion == "concat":
            cat = torch.cat([idle_embed, beh_embed], dim=1)  # (B,256)
            return self.concat_reduce(cat)                   # (B,128)
        elif self.fusion == "idleonly":
            return idle_embed
        elif self.fusion == "alpha":
            a = self._sigm(self.alpha)                 # 标量 (0,1)
            return a*beh_embed + (1-a)*idle_embed
        else:
            raise RuntimeError("invalid fusion")

    def forward(self, x: torch.Tensor):
        """
        x: (B, 287)
        - 前 31 维是统计特征，其中 x[:,30] = is_behavior ∈ {0,1}
        - 中间 128 维是 idle 嵌入
        - 最后 128 维是 behavior 嵌入
        """
        stats = x[:, :31]          # (B,31)
        both  = x[:, 31:]          # (B,256)
        idle_embed = both[:, :128] # (B,128)
        beh_embed  = both[:, 128:] # (B,128)

        # 融合
        weighted_embed = self.fuse(stats, idle_embed, beh_embed)  # (B,128)

        # 拼接统计特征与加权嵌入 → (B,159)，扩一维供 Transformer 使用
        fused = torch.cat([stats, weighted_embed], dim=1).unsqueeze(1)  # (B,1,159)
        h = self.fc(fused)                        # (B,1,256)
        h = self.encoder(h).squeeze(1)           # (B,256)

        # 三个并行头
        out_type   = self.cls_type(h)            # (B,num_type)
        out_brand  = self.cls_brand(h)           # (B,num_brand)
        out_device = self.cls_device(h)          # (B,num_device)
        return out_type, out_brand, out_device

# ===================== 评估函数 =====================
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             loss_fn: nn.Module,
             device: torch.device):
    """
    在验证集上评估模型：
    - 返回：平均损失、type/brand/device 的 top-1 准确率
    - 说明：
      * 标签是 one-hot，使用 argmax 转为类别索引；
      * 精度 = (预测类别 == 真实类别) / 样本数；
      * 损失为三头交叉熵之和的批次加权平均。
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    correct_type = 0
    correct_brand = 0
    correct_device = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inputs, y_type_oh, y_brand_oh, y_device_oh = batch
            inputs     = inputs.to(device)
            y_type_oh  = y_type_oh.to(device)
            y_brand_oh = y_brand_oh.to(device)
            y_device_oh= y_device_oh.to(device)

            # 前向计算
            pred_type, pred_brand, pred_device = model(inputs)

            # 将 one-hot 标签转换成类别索引（与交叉熵接口一致）
            y_type   = y_type_oh.argmax(dim=1)
            y_brand  = y_brand_oh.argmax(dim=1)
            y_device = y_device_oh.argmax(dim=1)

            # 计算多任务损失：三头交叉熵之和
            loss = (ALPHA_TYPE   * loss_fn(pred_type,   y_type) +
                    ALPHA_BRAND  * loss_fn(pred_brand,  y_brand) +
                    ALPHA_DEVICE * loss_fn(pred_device, y_device))

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            # 统计 top-1 准确率
            correct_type   += (pred_type.argmax(1)   == y_type).sum().item()
            correct_brand  += (pred_brand.argmax(1)  == y_brand).sum().item()
            correct_device += (pred_device.argmax(1) == y_device).sum().item()

    if num_samples == 0:
        # 极端情况下 DataLoader 为空
        return float("inf"), 0.0, 0.0, 0.0

    avg_loss = total_loss / num_samples
    acc_type   = correct_type   / num_samples
    acc_brand  = correct_brand  / num_samples
    acc_device = correct_device / num_samples
    return avg_loss, acc_type, acc_brand, acc_device

# ===================== 单组训练（支持多融合） =====================
def train_one_group(feat_key: str, seed: int = 0, fusion: str = "gate"):
    """
    训练消融组合中的一组（例如 F0 / F12），并在指定融合策略下训练：
    - feat_key：'F0'...'F12'
    - seed：随机数种子，便于复现实验
    - fusion：'gate' | 'concat' | 'idleonly' | 'alpha'
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    feat_combo = ABLATION_MAP[feat_key]
    print("\n" + "="*100)
    print(f"[TRAIN] 组合 = {feat_key}  ->  {feat_combo}   |   fusion = {fusion}   |   seed = {seed}")
    print("="*100)

    # 1) 构建 Dataset / DataLoader
    train_dataset = MultiModalIoTDataset(csv_path=TRAIN_CSV, label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
    valid_dataset = MultiModalIoTDataset(csv_path=TEST_CSV,  label_dict_dir=LABEL_DIR, feat_combo=feat_combo)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=safe_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=safe_collate)

    # 2) 初始化模型（类别数从 Dataset 动态读取）
    model = MultiTaskClassifier(
        input_dim=159,              # 31(stat) + 128(weighted embed)
        hidden_dim=256,
        num_type=train_dataset.num_type,
        num_brand=train_dataset.num_brand,
        num_device=train_dataset.num_device,
        fusion=fusion
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"[INFO] 设备: {DEVICE} | CUDA 可用: {torch.cuda.is_available()}")
    print(f"[INFO] 训练样本数: {len(train_dataset)} | 验证样本数: {len(valid_dataset)}")
    print(f"[INFO] 类别数: type={train_dataset.num_type}, brand={train_dataset.num_brand}, device={train_dataset.num_device}")

    # 3) 训练循环
    history = []
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        num_train_samples = 0

        # —— 逐批训练 —— #
        for batch in tqdm(train_loader, desc=f"[{feat_key}|{fusion}|seed{seed}] Epoch {epoch}/{NUM_EPOCHS}"):
            if batch is None:
                continue
            inputs, y_type_oh, y_brand_oh, y_device_oh = batch
            inputs     = inputs.to(DEVICE)
            y_type_oh  = y_type_oh.to(DEVICE)
            y_brand_oh = y_brand_oh.to(DEVICE)
            y_device_oh= y_device_oh.to(DEVICE)

            # one-hot → 类别索引
            y_type   = y_type_oh.argmax(dim=1)
            y_brand  = y_brand_oh.argmax(dim=1)
            y_device = y_device_oh.argmax(dim=1)

            # 前向与损失
            pred_type, pred_brand, pred_device = model(inputs)
            loss = (ALPHA_TYPE   * loss_fn(pred_type,   y_type) +
                    ALPHA_BRAND  * loss_fn(pred_brand,  y_brand) +
                    ALPHA_DEVICE * loss_fn(pred_device, y_device))

            # 反向与更新
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            batch_size = inputs.size(0)
            total_train_loss += loss.item() * batch_size
            num_train_samples += batch_size

        # —— 训练集平均损失 —— #
        avg_train_loss = total_train_loss / num_train_samples if num_train_samples > 0 else float("inf")

        # —— 验证 —— #
        valid_loss, acc_type, acc_brand, acc_device = evaluate(model, valid_loader, loss_fn, DEVICE)

        print(f"[{feat_key}|{fusion}|seed{seed}] "
              f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"TrainLoss: {avg_train_loss:.4f} | "
              f"ValLoss: {valid_loss:.4f} | "
              f"TypeAcc: {acc_type:.4f} | BrandAcc: {acc_brand:.4f} | DeviceAcc: {acc_device:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "valid_loss": float(valid_loss),
            "type_acc": float(acc_type),
            "brand_acc": float(acc_brand),
            "device_acc": float(acc_device),
        })

    # 4) 保存模型与配置
    elapsed = time.time() - start_time
    model_path = Path(MODEL_DIR) / f"optimized_multitask_model__{feat_key}__{fusion}__seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] 模型已保存：{model_path}")

    config = {
        "feat_key": feat_key,
        "feat_combo": feat_combo,
        "fusion": fusion,
        "seed": seed,
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "label_dir": LABEL_DIR,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "alpha_type": ALPHA_TYPE,
        "alpha_brand": ALPHA_BRAND,
        "alpha_device": ALPHA_DEVICE,
        "device": str(DEVICE),
        "num_type": train_dataset.num_type,
        "num_brand": train_dataset.num_brand,
        "num_device": train_dataset.num_device,
        "history": history,
        "time_sec": elapsed,
    }
    cfg_path = Path(MODEL_DIR) / f"optimized_multitask_model__{feat_key}__{fusion}__seed{seed}.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] 组配置与训练曲线已保存：{cfg_path}")

# ===================== 主入口 =====================
def main():
    print(f"[MAIN] 将训练以下组：{RUN_GROUPS} | seeds={SEEDS} | fusions={FUSION_LIST}")
    for feat_key in RUN_GROUPS:
        for fusion in FUSION_LIST:
            for seed in SEEDS:
                train_one_group(feat_key, seed, fusion)

if __name__ == "__main__":
    main()
