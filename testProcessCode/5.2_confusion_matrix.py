# -*- coding：utf-8 -*-

"""
根据输入的混淆矩阵csv文件，画出混淆矩阵图，并且生成pdf文件
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 步骤 1：读取混淆矩阵数据
# 假设CSV文件的第一列是行索引（真实设备），第一行是列索引（预测设备）
conf_matrix = pd.read_csv("/home/hyj/unknownDeviceIdentification/testProcessCode/uk_type_confusion_matrix.csv", index_col=0)

# 步骤 2：填充缺失值为0
conf_matrix = conf_matrix.fillna(0)

# 确保所有数据都是整数类型
conf_matrix = conf_matrix.astype(int)

# 可选：检查数据
print(conf_matrix.head())
print(f"混淆矩阵形状: {conf_matrix.shape}")

# 创建一个注释数组，将非零值保留为其本身，零值替换为空字符串
annot_array = np.where(conf_matrix.values != 0, conf_matrix.values.astype(str), '')

# 步骤 3：设置绘图尺寸
# 根据混淆矩阵的复杂性，建议使用较大的尺寸
plt.figure(figsize=(12, 10))  # 宽度12英寸，高度10英寸

# 步骤 4：使用Seaborn绘制热图
# conf_matrix：用于颜色映射的数据。
# annot=annot_data：自定义的注释数据，仅包含非零值。如果不调用annot_array,使用TRUE，则空白单元格会显示0
# fmt=''：格式化字符串为空，因为注释已经预先格式化。
# cmap='Blues'：使用蓝色渐变色。
# linewidths=0：设置单元格之间的线宽。
# cbar=True：显示颜色条。
sns.heatmap(conf_matrix, annot=annot_array, fmt='', cmap='Blues', cbar=True)

# 步骤 5：添加标题和标签
plt.title('US Confusion Matrix', fontsize=16)
plt.xlabel('Matching Result', fontsize=14)
plt.ylabel('Devices labels', fontsize=14)

# 步骤 6：调整x轴和y轴标签的字体大小及旋转角度
plt.xticks(fontsize=8, rotation=90)  # 根据需要调整字体大小
plt.yticks(fontsize=8, rotation=0)

# 步骤 7：优化布局以防止标签被截断
plt.tight_layout()

# 步骤 8：保存图像为PDF格式
plt.savefig('/home/hyj/unknownDeviceIdentification/testProcessCode/uk_type_confusion_matrix.pdf', format='pdf')

# 如果需要，也可以保存为高分辨率的PNG格式
# plt.savefig('confusion_matrix.png', format='png', dpi=300)

# 步骤 9：显示图形（可选）
plt.show()
