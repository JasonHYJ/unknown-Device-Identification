import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = r'/home/hyj/unknownDeviceIdentification/testCharts/uk&us.xlsx' 
df = pd.read_excel(file_path, header=None)

device_names = []
y_true = []
y_pred = []

# 解析Excel中的数据
for index, row in df.iterrows():
    device = row[0]
    accuracy_total = row[1]
    accuracy, total = map(int, accuracy_total.split('|'))
    device_names.append(device)

    y_true.extend([device] * total)

    predicted_count = 0
    for col in row[2:]:
        if pd.notnull(col):
            predicted_device, count = col.split('(')
            count = int(count[:-1])
            y_pred.extend([predicted_device] * count)
            predicted_count += count

    if predicted_count < total:
        y_pred.extend(["None"] * (total - predicted_count))

print(f"Number of true labels: {len(y_true)}")
print(f"Number of predicted labels: {len(y_pred)}")

# 确保 "None" 在设备名称中
if "None" not in device_names:
    device_names.append("None")

conf_matrix = confusion_matrix(y_true, y_pred, labels=device_names)

# 移除 "None" 的行和列
if "None" in device_names:
    none_index = device_names.index("None")
    device_names.remove("None")
    conf_matrix = conf_matrix[:none_index, :none_index]

# 绘制 SVG 混淆矩阵
def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, format='svg', dpi=300)  # 保存为SVG矢量图格式
    plt.show()

# 保存路径为 SVG 格式
save_path = "confusion_matrix.svg"
plot_confusion_matrix(conf_matrix, device_names, save_path)

print(f"Confusion matrix saved to {save_path}")
