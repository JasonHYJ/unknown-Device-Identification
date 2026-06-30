# -*- coding: utf-8 -*-

"""
根据输入的混淆矩阵csv文件，给出该csv文件的混淆矩阵图，其中列多了一列unknown。
然后，再计算总的准确率，精确率，召回率以及F1；此外还有各设备单独的识别性能指标。
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 设置全局字体为 Times New Roman
# mpl.rcParams['font.family'] = 'Times New Roman'


def evaluate_confusion_matrix_from_csv(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, index_col=0)

    # 提取设备名称
    device_names = df.index.tolist()

    # 获取混淆矩阵
    confusion_matrix = df.to_numpy()

    # 计算总的准确率、召回率、精确率、F1分数
    n = confusion_matrix.shape[0]  # 真实设备数量
    total_samples = confusion_matrix.sum()
    total_correct = sum(confusion_matrix[i][i] for i in range(n))

    # 计算准确率
    accuracy = total_correct / total_samples

    # 计算召回率
    recalls = []
    for i in range(n):
        true_total = confusion_matrix[i].sum()
        recalls.append(confusion_matrix[i][i] / true_total if true_total != 0 else 0.0)
    macro_recall = np.mean(recalls)

    # 计算精确率
    precisions = []
    for i in range(n):
        predicted_total = confusion_matrix[:, i].sum()
        precisions.append(confusion_matrix[i][i] / predicted_total if predicted_total != 0 else 0.0)
    macro_precision = np.mean(precisions)

    # 计算F1分数
    f1_scores = []
    for p, r in zip(precisions, recalls):
        f1_scores.append(2 * (p * r) / (p + r) if (p + r) != 0 else 0.0)
    macro_f1 = np.mean(f1_scores)

    # 打印结果
    print(f"总准确率: {accuracy:.4f}")
    print(f"宏平均召回率: {macro_recall:.4f}")
    print(f"宏平均精确率: {macro_precision:.4f}")
    print(f"宏平均F1分数: {macro_f1:.4f}\n")

    for i in range(n):
        print(f"{device_names[i]} - Recall: {recalls[i]:.4f}, Precision: {precisions[i]:.4f}, F1: {f1_scores[i]:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))  # 调整画布大小，确保标题和坐标轴不会被截断
    class_names = device_names
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=device_names,
                linecolor='gray',  # 使用灰色边界线提升视觉边界
                # cbar_kws={'label': 'Sample Count'},
                annot_kws={"size": 12})  # 调整注释字体大小

    # 设置坐标轴字体颜色和大小
    plt.xticks(fontsize=12, color='black', rotation=45)
    plt.yticks(fontsize=12, color='black')

    plt.title('Type Confusion Matrix', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    plt.rcParams['font.family'] = 'Times New Roman'

    # 显示图表
    plt.tight_layout()  # 自动调整布局，防止标题和坐标被截断
    plt.savefig('/home/hyj/unknownDeviceIdentification/testProcessCode/uk_type_confusion_matrix.pdf', format='pdf')
    plt.show()


def main():
    # 用户输入CSV文件路径
    csv_file_path = "/home/hyj/unknownDeviceIdentification/testProcessCode/uk_type_confusion_matrix.csv"

    # 执行评估
    evaluate_confusion_matrix_from_csv(csv_file_path)


if __name__ == "__main__":
    main()
