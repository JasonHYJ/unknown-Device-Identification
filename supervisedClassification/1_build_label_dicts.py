import pandas as pd
import json
from pathlib import Path

# ---------------------- 配置路径 ----------------------
csv_path = Path("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us/3_us_train.csv")
save_dir = Path("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/us")
save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------- 加载数据 ----------------------
df = pd.read_csv(csv_path)

# ---------------------- 构建映射字典 ----------------------
type_labels = sorted(df["type_label"].unique())
brand_labels = sorted(df["brand_label"].unique())
device_labels = sorted(df["device_label"].unique())

type2idx = {label: idx for idx, label in enumerate(type_labels)}
brand2idx = {label: idx for idx, label in enumerate(brand_labels)}
device2idx = {label: idx for idx, label in enumerate(device_labels)}

# ---------------------- 保存为 JSON ----------------------
with open(save_dir / "type2idx.json", "w") as f:
    json.dump(type2idx, f, indent=2)
with open(save_dir / "brand2idx.json", "w") as f:
    json.dump(brand2idx, f, indent=2)
with open(save_dir / "device2idx.json", "w") as f:
    json.dump(device2idx, f, indent=2)

# ---------------------- 打印信息 ----------------------
print("✅ 标签映射字典构建完成并保存至:")
print(f"  - 设备类型 (type2idx): {len(type2idx)} 类别")
print(f"  - 厂商类型 (brand2idx): {len(brand2idx)} 类别")
print(f"  - 具体设备 (device2idx): {len(device2idx)} 类别")
print(f"📁 保存目录: {save_dir}")
