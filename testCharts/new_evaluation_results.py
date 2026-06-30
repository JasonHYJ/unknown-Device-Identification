# @Author: Ming
# @Date: 2025/05/07
# 加载Excel
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

file_path = r'/home/hyj/unknownDeviceIdentification/testCharts/uk&us.xlsx'
df = pd.read_excel(file_path, header=None)

device_names = []
y_true = []
y_pred = []

for index, row in df.iterrows():
    device = row[0]
    accuracy_total = row[1]
    accuracy, total = map(int, str(accuracy_total).split('|'))

    # 如果是伪标签 unknown device，则跳过计入 device_names
    if device != 'unknown device':
        device_names.append(device)

    y_true.extend([device] * total)

    for col in row[2:]:
        if pd.notnull(col):
            predicted_device, count = col.split('(')
            count = int(count[:-1])
            y_pred.extend([predicted_device] * count)

# 构建用于评估的真实设备标签
true_filtered = []
pred_filtered = []

for yt, yp in zip(y_true, y_pred):
    # 只保留真实设备的情况
    if yt != 'unknown device':
        true_filtered.append(yt)
        if yp in device_names:
            pred_filtered.append(yp)
        else:
            pred_filtered.append('unknown')  # 不影响性能计算的占位

# 计算混淆矩阵（只对真实设备）
conf_matrix = confusion_matrix(true_filtered, pred_filtered, labels=device_names)

# 指标计算
results = []
for i, device in enumerate(device_names):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    auc = 0.5 * (recall + specificity)

    results.append([device, precision, recall, f1, accuracy, specificity, auc])

# 总体平均
avg_results = np.mean([r[1:] for r in results], axis=0)
results.append(['Overall Average'] + avg_results.tolist())

# 保存结果
results_df = pd.DataFrame(results,
                          columns=["Device", "Precision", "Recall", "F1-Score", "Accuracy", "Specificity", "AUC"])
results_df.to_excel('evaluation_results_filtered.xlsx', index=False)
print("Filtered evaluation results saved to evaluation_results_filtered.xlsx")
