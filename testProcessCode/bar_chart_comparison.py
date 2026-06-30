import matplotlib.pyplot as plt
import numpy as np

# ---------- 字体设置 ----------
plt.rcParams['font.family'] = 'Times New Roman'      # 全局字体（英文、数字）
plt.rcParams['axes.unicode_minus'] = False           # 解决负号显示
# -----------------------------

# ========== 可调整参数区域 ==========
# 数据设置
methods = ['SVM', 'RF', 'XGB', '1D-CNN', 'LSTM', 'ByteCNN', 'EarlyConcat', 'AttentionFusion', 'LHDI', 'HSGAN-IoT', 'Ours']

# 三个任务的准确率数据
type_acc = [0.9819, 0.9918, 0.9940, 0.9727, 0.9773, 0.8429, 0.9958, 0.9938, 0.9520, 0.9650, 0.9967]
brand_acc = [0.9826, 0.9909, 0.9922, 0.9582, 0.9630, 0.8983, 0.9943, 0.9934, 0.9450, 0.9600, 0.9955]
device_acc = [0.9577, 0.9700, 0.9704, 0.9499, 0.9167, 0.8537, 0.9681, 0.9704, 0.9100, 0.9350, 0.9807]

# 图形尺寸设置
fig_width = 18  # 增加图形宽度以容纳更多方法
fig_height = 9  # 图形高度

# 柱状图设置
bar_width = 0.25  # 稍微减小柱子宽度
group_spacing = 0.02  # 组内间距

# 颜色方案 - 使用更温和的学术颜色
colors = ['#485098', '#80CDBB', '#F3DE67']  # 类型, 厂商, 型号 - 蓝色系

# 字体大小设置（增大后的值）
title_fontsize = 20      # 标题字体大小（已注释但保留）
label_fontsize = 22      # 坐标轴标签字体大小（从18增大）
tick_fontsize = 16       # 刻度标签字体大小（从12增大）
legend_fontsize = 18     # 图例字体大小（从14增大）
value_fontsize = 14      # 柱子上数值标签字体大小（从12增大）

# 其他设置
y_min = 0.80  # 降低Y轴最小值以显示低值方法
y_max = 1.02   # 提高Y轴最大值，避免标签数值超过边界
grid_alpha = 0.3  # 网格透明度
dpi = 300  # 输出图片分辨率
rotation_angle = 45  # X轴标签旋转角度
value_offset = 0.005  # 增加数值标签相对于柱子的垂直偏移量
x_margin = 0.5  # 控制X轴左右边距，值越小边距越小
# ========== 参数区域结束 ==========

# 创建图形
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# 设置位置
x = np.arange(len(methods))
positions = [x - bar_width, x, x + bar_width]

# 绘制条形图
bars1 = ax.bar(positions[0], type_acc, bar_width, label='类型准确率', color=colors[0], alpha=0.85)
bars2 = ax.bar(positions[1], brand_acc, bar_width, label='厂商准确率', color=colors[1], alpha=0.85)
bars3 = ax.bar(positions[2], device_acc, bar_width, label='型号准确率', color=colors[2], alpha=0.85)


# 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + value_offset,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=value_fontsize, rotation=0)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# 美化图表
ax.set_xlabel('方法', fontsize=label_fontsize, fontfamily='SimSun')
ax.set_ylabel('准确率', fontsize=label_fontsize, fontfamily='SimSun')
# ax.set_title('Closed-Set Classification Performance Comparison on Three Tasks', fontsize=title_fontsize, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=rotation_angle, ha='right', fontsize=tick_fontsize)

# 设置y轴刻度标签字体大小
ax.tick_params(axis='y', labelsize=tick_fontsize)

# ---------- 修改：图例用宋体 ----------
ax.legend(loc='upper left', prop={'family': 'SimSun', 'size': legend_fontsize})
# ------------------------------------
ax.grid(axis='y', alpha=grid_alpha)
ax.set_ylim(y_min, y_max)

# 调整X轴范围，减少左右边距
ax.set_xlim(-x_margin, len(methods) - 1 + x_margin)

# 添加性能优势标注
ax.text(len(methods)-1, 0.97, '本文方法: 最佳效果',
        fontfamily='SimSun',  # 指定宋体
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.9, edgecolor='#CCCCCC'),
        ha='center', va='bottom', fontsize=12)  # 此标注的字体大小保持较小，可根据需要调整

# 设置背景色为更柔和的颜色
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# 添加水平参考线，使低值更容易读取
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('C:/Users/jiyiy/Desktop/学位论文/论文插图/第五章/中文图3类型厂商识别准确率.pdf', bbox_inches='tight', dpi=dpi, format='pdf')
plt.show()