# @Author: Ming
# @Date: 2025/09/26
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm

# ===================== 字体设置 =====================
# 强制加载 Times New Roman 字体（确保 PDF 嵌入正确）
font_path = r"C:\Windows\Fonts\times.ttf"  # 根据实际情况修改路径
times_new_roman = fm.FontProperties(fname=font_path)

# 设置 seaborn 风格
sns.set(style="whitegrid")

# 设置 matplotlib 全局字体
mpl.rcParams['font.family'] = times_new_roman.get_name()
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['lines.linewidth'] = 1.2

# ===================== 数据读取 =====================
file_path = r'C:\Users\Lenovo\Desktop\论文借鉴\实验结果\actual\uk&us.xlsx'
df = pd.read_excel(file_path, header=None)

# 初始化设备名称列表
device_names = []

# 初始化真实标签和预测标签列表
y_true = []
y_pred = []

# 解析 Excel 数据
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

# ===================== 混淆矩阵计算 =====================
conf_matrix = confusion_matrix(y_true, y_pred, labels=device_names)

# Precision 和 Recall 计算
precision = {}
recall = {}
for i, device in enumerate(device_names):
    TP = conf_matrix[i, i]
    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP

    precision[device] = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall[device] = TP / (TP + FN) if (TP + FN) > 0 else 0

# 筛选 Precision 或 Recall < 1 的设备
devices_to_plot = [device for device in device_names if precision[device] < 1 or recall[device] < 1]

# ===================== 绘图函数 =====================
def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.7,
        linecolor='black',
        cbar_kws={'format': '%d%%'},
        square=False,
        annot_kws={"size": 15}
    )

    # 颜色条字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    # 横坐标放在下方，标签右倾斜 45°
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=45, ha='right', fontsize=15)

    # 调整 y 轴标签
    plt.yticks(rotation=0, ha='right', fontsize=15)

    # 手动设置刻度位置
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)

    # 坐标轴加粗
    plt.xlabel('Predicted', fontsize=18, weight='bold')
    plt.ylabel('Actual', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')  # 保存为 PDF
    plt.show()

# ===================== 保存 PDF =====================
if devices_to_plot:
    index = [device_names.index(device) for device in devices_to_plot]
    filtered_conf_matrix = conf_matrix[np.ix_(index, index)]

    # 计算百分比矩阵
    row_sums = filtered_conf_matrix.sum(axis=1, keepdims=True)
    percentage_matrix = np.divide(filtered_conf_matrix, row_sums, where=row_sums != 0) * 100

    save_path = "filtered_confusion_matrix_percentage.svg"
    plot_confusion_matrix(percentage_matrix, devices_to_plot, save_path)
    print(f"Filtered confusion matrix (percentage) saved to {save_path}")
else:
    print("No devices with Precision or Recall < 1.")
