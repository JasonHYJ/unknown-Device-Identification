import matplotlib.pyplot as plt
import numpy as np

# 设置字体与风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2
plt.style.use('seaborn-v0_8-whitegrid')

labels = ['E\n(0.3,0.4,0.3)', 'C\n(0.4,0.3,0.3)',
          'B\n(0.5,0.3,0.2)', 'A\n(0.6,0.2,0.2)', 'D\n(0.7,0.15,0.15)',
          'G\n(0.8,0.1,0.1)']
x = np.arange(len(labels))

precision = [84.5, 89.5, 92.0, 97.02, 95.8, 94.5]
recall    = [82.0, 87.0, 90.2, 94.76, 93.8, 91.5]
f1_scores = [81.5, 86.0, 89.5, 94.66, 92.7, 90.0]
accuracy  = [93.2, 96.0, 97.5, 99.82, 98.8, 97.8]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制四条曲线
ax.plot(x, precision, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Precision')
ax.plot(x, recall, marker='^', linestyle='--', color='#ff7f0e', linewidth=2, label='Recall')
ax.plot(x, f1_scores, marker='s', linestyle='-.', color='#2ca02c', linewidth=2, label='F1-score')
ax.plot(x, accuracy, marker='d', linestyle=':', color='#d62728', linewidth=2, label='Accuracy')

# 坐标轴与标题
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_xlabel(r'Weight Combinations $(\alpha, \beta, \gamma)$', fontsize=13)
ax.set_ylim(80, 100)
ax.set_title('Effect of Weight Settings on Evaluation Metrics', fontsize=14, pad=15)

# 设置黑色边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 图例
ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='black')

# 数值标注：四条线全部标注
# 数值标注：精细调整每类线段标注的相对位置
for i in range(len(x)):
    ax.text(x[i], precision[i] + 0.5, f'{precision[i]:.2f}', ha='center', fontsize=9, color='#1f77b4')  # 蓝色
    ax.text(x[i], recall[i] + 0.5, f'{recall[i]:.2f}', ha='center', fontsize=9, color='#ff7f0e')  # 橙色
    ax.text(x[i], f1_scores[i] - 0.7, f'{f1_scores[i]:.2f}', ha='center', fontsize=9, color='#2ca02c')  # 绿色
    ax.text(x[i], accuracy[i] - 0.7, f'{accuracy[i]:.2f}', ha='center', fontsize=9, color='#d62728')  # 红色

plt.tight_layout()
plt.savefig("weight_metrics_plot.svg", format='svg', bbox_inches='tight')
plt.show()
