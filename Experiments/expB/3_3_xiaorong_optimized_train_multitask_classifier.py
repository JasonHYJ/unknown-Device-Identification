# xiaorong_optimized_train_multitask_classifier.py
# 📆 多任务 IoT 设备识别 — 消融训练脚本（F0–F12）
#
# 功能与设计要点：
# 1) 读取多模态样本：统计特征（31维，末位为 is_behavior），嵌入（序列+原始字节，各64维）；
# 2) 根据 is_behavior，将嵌入放入 idle 或 behavior 一侧（另一侧置零），得到 287 维输入；
# 3) 门控融合（σ(Linear(is_behavior))）：加权 idle vs behavior → 128 维；
# 4) 拼接统计特征（31）+加权嵌入（128）→ 159 维，经 FC+Transformer(2层,4头) 编码；
# 5) 三个分类头并行输出：type / brand / device；
# 6) 批量跑消融组合（F0/F10/F11/F12；其余组已在字典中就绪）。
#
# 使用说明：
# - 修改“参数区”的 TRAIN_CSV / TEST_CSV / LABEL_DIR 指向你当前数据集；
# - 输出模型与配置保存在 OUT_DIR/models/ 下；
# - 评估用你现有的 evaluate 脚本，对每个 .pt 逐个评估即可。
#
# 依赖：
# - xiaorong_optimized_multimodal_dataset.py（确保与本脚本同目录或在 PYTHONPATH）
# - CSV 中包含绝对路径: stat_feature_path / seq_embed_feature_path / raw_embed_feature_path
# - 标签字典：type2idx.json / brand2idx.json / device2idx.json

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
MODEL_DIR = f"{OUT_DIR}/3_3_models"
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

# —— 任务开关（3.3 单/双/多任务实验）——
# 单任务
    # 只 type：USE_TASK_TYPE=True, USE_TASK_BRAND=False, USE_TASK_DEVICE=False
    # 只 brand：False, True, False
    # 只 device：False, False, True
# 双任务
    # type+brand：True, True, False
    # type+device：True, False, True
    # brand+device：False, True, True
# 多任务（基线）：True, True, True（你已做）

USE_TASK_TYPE   = True
USE_TASK_BRAND  = True
USE_TASK_DEVICE = True

def _task_tag():
    t = 'T' if USE_TASK_TYPE else '0'
    b = 'B' if USE_TASK_BRAND else '0'
    d = 'D' if USE_TASK_DEVICE else '0'
    return f"{t}{b}{d}"


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

# —— 默认先跑主线四组（其余随时放开）——
# RUN_GROUPS = ["F0", "F10", "F11", "F12"]
# RUN_GROUPS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"]
RUN_GROUPS = ["F12"]    # 测试单/多任务时候，只在F12上面变化
SEEDS      = [0]  # 如需多次重复：例如 [0,1,2]

# ===================== 模型定义 =====================
class MultiTaskClassifier(nn.Module):
    """
    多任务分类器：
    输入：x ∈ R^{B,287} = [31(stat), 128(idle_embed), 128(behavior_embed)]
    门控：g = σ(Linear(is_behavior))，按 is_behavior 融合 idle/behavior 嵌入
    编码：FC(159→256) → TransformerEncoder(2层, 4头)
    输出：三头线性层，分别预测 type/brand/device
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_type: int, num_brand: int, num_device: int):
        super().__init__()
        # 门控：使用 is_behavior（来自 stat 的最后一维），输出 [0,1] 权重
        self.gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
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
        is_behavior = stats[:, 30:31]  # (B,1)

        # 门控融合（根据 is_behavior 选择倾向）
        gate = self.gate(is_behavior)                 # (B,1) in [0,1]
        weighted_embed = gate*beh_embed + (1-gate)*idle_embed  # (B,128)

        # 拼接统计特征与加权嵌入 → (B,159)，扩一维供 Transformer 使用
        fused = torch.cat([stats, weighted_embed], dim=1).unsqueeze(1)  # (B,1,159)
        h = self.fc(fused)                         # (B,1,256)
        h = self.encoder(h).squeeze(1)            # (B,256)

        # 三个并行头
        out_type   = self.cls_type(h)             # (B,num_type)
        out_brand  = self.cls_brand(h)            # (B,num_brand)
        out_device = self.cls_device(h)           # (B,num_device)
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

            # —— 按开关计算批次损失 —— #
            batch_loss = 0.0
            if USE_TASK_TYPE:
                batch_loss += loss_fn(pred_type,   y_type)
            if USE_TASK_BRAND:
                batch_loss += loss_fn(pred_brand,  y_brand)
            if USE_TASK_DEVICE:
                batch_loss += loss_fn(pred_device, y_device)

            batch_size = inputs.size(0)
            total_loss += float(batch_loss) * batch_size
            num_samples += batch_size

            # —— 按开关统计准确率 —— #
            if USE_TASK_TYPE:
                correct_type   += (pred_type.argmax(1)   == y_type).sum().item()
            if USE_TASK_BRAND:
                correct_brand  += (pred_brand.argmax(1)  == y_brand).sum().item()
            if USE_TASK_DEVICE:
                correct_device += (pred_device.argmax(1) == y_device).sum().item()

    if num_samples == 0:
        return float("inf"), None, None, None

    avg_loss = total_loss / num_samples
    acc_type   = (correct_type   / num_samples) if USE_TASK_TYPE   else None
    acc_brand  = (correct_brand  / num_samples) if USE_TASK_BRAND  else None
    acc_device = (correct_device / num_samples) if USE_TASK_DEVICE else None
    return avg_loss, acc_type, acc_brand, acc_device


def fmt4(x):
    """把可能为 None/inf 的数安全地格式化为 4 位小数；None 用 '—'。"""
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

def to_float_or_none(x):
    """把可能为 None 的数用于 JSON 存储；None 原样返回，其它转 float。"""
    return None if x is None else float(x)


# ===================== 单组训练 =====================
def train_one_group(feat_key: str, seed: int = 0):
    """
    训练消融组合中的一组（例如 F0 / F10 / F11 / F12）：
    - feat_key：'F0'...'F12'
    - seed：随机数种子，便于复现实验
    - 会将每个 epoch 的训练/验证指标打印，并在最终保存模型 + 该组配置 JSON
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    feat_combo = ABLATION_MAP[feat_key]
    print("\n" + "="*80)
    print(f"[TRAIN] 组合 = {feat_key}  ->  {feat_combo}   |   seed = {seed}")
    print("="*80)

    # 1) 构建 Dataset / DataLoader
    #   - MultiModalIoTDataset 内部会根据 feat_combo 选择对比/自监督路径
    #   - 自监督 128 维会自动对齐到 64 维
    train_dataset = MultiModalIoTDataset(csv_path=TRAIN_CSV, label_dict_dir=LABEL_DIR, feat_combo=feat_combo)
    valid_dataset = MultiModalIoTDataset(csv_path=TEST_CSV,  label_dict_dir=LABEL_DIR, feat_combo=feat_combo)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=safe_collate)
    valid_loader   = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=safe_collate)

    # 2) 初始化模型（类别数从 Dataset 动态读取）
    model = MultiTaskClassifier(
        input_dim=159,              # 31(stat) + 128(weighted embed)
        hidden_dim=256,
        num_type=train_dataset.num_type,
        num_brand=train_dataset.num_brand,
        num_device=train_dataset.num_device
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"[INFO] 设备: {DEVICE} | CUDA 可用: {torch.cuda.is_available()}")
    print(f"[INFO] 训练样本数: {len(train_dataset)} | 验证样本数: {len(valid_dataset)}")
    print(f"[INFO] 类别数: type={train_dataset.num_type}, brand={train_dataset.num_brand}, device={train_dataset.num_device}")

    # 3) 训练循环
    history = []
    task_suffix = _task_tag()
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        num_train_samples = 0

        # —— 逐批训练 —— #
        for batch in tqdm(train_loader, desc=f"[{feat_key}|seed{seed}] Epoch {epoch}/{NUM_EPOCHS}"):
            if batch is None:
                continue
            inputs, y_type_oh, y_brand_oh, y_device_oh = batch
            inputs     = inputs.to(DEVICE)
            y_type_oh  = y_type_oh.to(DEVICE)
            y_brand_oh = y_brand_oh.to(DEVICE)
            y_device_oh= y_device_oh.to(DEVICE)

            # 前向
            pred_type, pred_brand, pred_device = model(inputs)

            # one-hot → 索引
            y_type   = y_type_oh.argmax(dim=1)
            y_brand  = y_brand_oh.argmax(dim=1)
            y_device = y_device_oh.argmax(dim=1)

            # —— 按任务开关累加损失 —— #
            loss = 0.0
            if USE_TASK_TYPE:
                loss += ALPHA_TYPE   * loss_fn(pred_type,   y_type)
            if USE_TASK_BRAND:
                loss += ALPHA_BRAND  * loss_fn(pred_brand,  y_brand)
            if USE_TASK_DEVICE:
                loss += ALPHA_DEVICE * loss_fn(pred_device, y_device)

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

        print(f"[{feat_key}|seed{seed}|tasks={task_suffix}] "
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"TrainLoss: {fmt4(avg_train_loss)} | "
            f"ValLoss: {fmt4(valid_loss)} | "
            f"TypeAcc: {fmt4(acc_type)} | BrandAcc: {fmt4(acc_brand)} | DeviceAcc: {fmt4(acc_device)}")
              

        history.append({
            "epoch": int(epoch),
            "train_loss": to_float_or_none(avg_train_loss),
            "valid_loss": to_float_or_none(valid_loss),
            "type_acc": to_float_or_none(acc_type),
            "brand_acc": to_float_or_none(acc_brand),
            "device_acc": to_float_or_none(acc_device),
        })



    # 4) 保存模型与配置
    elapsed = time.time() - start_time
    
    model_path = Path(MODEL_DIR) / f"optimized_multitask_model__{feat_key}__seed{seed}__tasks[{task_suffix}].pt"
    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] 模型已保存：{model_path}")

    config = {
        "feat_key": feat_key,
        "feat_combo": feat_combo,
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
        "time_sec": to_float_or_none(elapsed),
    }
    cfg_path = Path(MODEL_DIR) / f"optimized_multitask_model__{feat_key}__seed{seed}__tasks[{task_suffix}].json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] 组配置与训练曲线已保存：{cfg_path}")

# ===================== 主入口 =====================
def main():
    print(f"[MAIN] 将训练以下组：{RUN_GROUPS} | seeds={SEEDS}")
    for feat_key in RUN_GROUPS:
        for seed in SEEDS:
            train_one_group(feat_key, seed)

if __name__ == "__main__":
    main()
