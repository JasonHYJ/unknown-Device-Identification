import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取Excel文件
file_path = r'/home/hyj/unknownDeviceIdentification/testCharts/uk&us.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path, header=None)

# 初始化设备名称列表
device_names = []

# 初始化真实标签和预测标签列表
y_true = []
y_pred = []

# 解析Excel中的数据
for index, row in df.iterrows():
    device = row[0]
    accuracy_total = row[1]
    accuracy, total = map(int, accuracy_total.split('|'))
    device_names.append(device)

    # 添加真实标签
    y_true.extend([device] * total)

    # 添加预测标签
    for col in row[2:]:
        if pd.notnull(col):
            predicted_device, count = col.split('(')
            count = int(count[:-1])
            y_pred.extend([predicted_device] * count)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred, labels=device_names)

# 计算 Precision 和 Recall
precision = {}
recall = {}

for i, device in enumerate(device_names):
    TP = conf_matrix[i, i]
    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP

    precision[device] = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall[device] = TP / (TP + FN) if (TP + FN) > 0 else 0

# 筛选 Precision 或 Recall 小于 1 的设备
devices_to_plot = [device for device in device_names if precision[device] < 1 or recall[device] < 1]

# 筛选对应的混淆矩阵部分
if devices_to_plot:
    index = [device_names.index(device) for device in devices_to_plot]
    filtered_conf_matrix = conf_matrix[np.ix_(index, index)]

    # 计算百分比矩阵
    row_sums = filtered_conf_matrix.sum(axis=1, keepdims=True)
    percentage_matrix = np.divide(filtered_conf_matrix, row_sums, where=row_sums != 0) * 100

    # 绘制并保存混淆矩阵
    def plot_confusion_matrix(cm, labels, save_path):
        plt.figure(figsize=(16, 12))
        
        # 移除颜色条: cbar=False
        ax = sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", 
                         xticklabels=labels, yticklabels=labels,
                         linewidths=0, linecolor='black', 
                         square=False, annot_kws={"size": 15},
                         cbar=False)  # 移除颜色条
        
        # 设置坐标轴位置：横坐标在底部
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        
        # 旋转x轴标签45度，放置在底部
        plt.xticks(rotation=30, ha='right', fontsize=16)  # 倾斜45度，右对齐
        
        # 调整y轴标签
        plt.yticks(rotation=0, ha='right', fontsize=16)
        
        # 设置刻度和标签位置
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        
        # 设置字体加粗
        plt.xlabel('Predicted', fontsize=18, weight='bold')
        plt.ylabel('Actual', fontsize=18, weight='bold')
        
        # 调整布局避免标签重叠
        plt.tight_layout()
        
        # 保存为PDF文件
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()

    # 保存混淆矩阵为PDF文件
    save_path = "filtered_confusion_matrix_percentage.pdf"
    plot_confusion_matrix(percentage_matrix, devices_to_plot, save_path)

    print(f"Filtered confusion matrix (percentage) saved to {save_path}")
else:
    print("No devices with Precision or Recall < 1.")