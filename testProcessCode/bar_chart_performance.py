import matplotlib.pyplot as plt
import numpy as np

# ---------- 字体设置 ----------
plt.rcParams['font.family'] = 'Times New Roman'      # 全局字体（英文、数字）
plt.rcParams['axes.unicode_minus'] = False           # 解决负号显示
# -----------------------------

# ========== 可调整参数区域 ==========
# 数据设置
datasets = ['Mon(IoT)r-uk', 'Mon(IoT)r-us', 'CIC IoT 2022']

# 类型识别准确率
idle_type_acc = [0.9868, 0.9776, 0.9821]
behavior_type_acc = [0.9633, 0.9677, 0.9633]

# 厂商识别准确率
idle_brand_acc = [0.9754, 0.9715, 0.9811]
behavior_brand_acc = [0.9646, 0.9632, 0.9746]

# 图形尺寸设置（稍微缩小宽度，避免柱子太细后显得太空）
fig_width = 12   # 原14 → 12
fig_height = 8

# 柱状图设置
bar_width = 0.12                    # 单根柱子宽度（保持较窄）
n_bars_per_group = 4                # 每组4根柱子
group_width = bar_width * n_bars_per_group * 1.1   # 每组总宽度，乘1.3留出组内间隙
group_spacing_factor = 0.65          # 组间距缩放因子，越小越紧凑（可调 0.8~1.0）
colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']

# 字体大小设置（全部显著增大）
title_fontsize = 20    # 原16 → 20
label_fontsize = 18    # 原14 → 18（X/Y轴标签）
tick_fontsize = 18     # 原12 → 15（刻度标签）
legend_fontsize = 15   # 原12 → 15
value_fontsize = 13    # 原12 → 13（柱子上数值，更清晰）

# 其他设置
y_min = 0.9
y_max = 1.0
grid_alpha = 0.3
dpi = 300
# ========== 参数区域结束 ==========

# 创建图形
fig, ax = plt.subplots(figsize=(fig_width, fig_height))


# 计算每个组的中心位置（组间距缩小）
x_centers = np.arange(len(datasets)) * group_spacing_factor

# 计算每组内4根柱子的偏移（均匀分布 + 留间隙）
offsets = np.linspace(-group_width/2 + bar_width/2,
                      group_width/2 - bar_width/2,
                      n_bars_per_group)

# 实际柱子位置 = 组中心 + 偏移
positions = [x_centers + offset for offset in offsets]

# 数据组合
data_groups = [
    (idle_type_acc, '闲时-类型'),
    (behavior_type_acc, '行为-类型'),
    (idle_brand_acc, '闲时-厂商'),
    (behavior_brand_acc, '行为-厂商')
]

# 绘制柱状图
bars = []
for i, (data, label) in enumerate(data_groups):
    bar = ax.bar(positions[i], data, bar_width, label=label, color=colors[i], alpha=0.8)
    bars.append(bar)

# 添加数值标签（数值字体更大，向上偏移稍多一点避免重叠）
for i, (data, _) in enumerate(data_groups):
    for j, value in enumerate(data):
        ax.text(positions[i][j], value + 0.003, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=value_fontsize)  # 加粗更醒目

# 美化图表（所有文字都用更大的字体）
ax.set_xlabel('数据集', fontsize=label_fontsize, fontfamily='SimSun')
ax.set_ylabel('识别准确率', fontsize=label_fontsize, fontfamily='SimSun')
# ax.set_title('Performance Comparison on Unknown Set: Type and Brand Identification', fontsize=title_fontsize, pad=20)
ax.set_xticks(x_centers)
ax.set_xticklabels(datasets, fontsize=tick_fontsize)
ax.legend(fontsize=legend_fontsize, loc='lower right', prop={'family': 'SimSun', 'size': legend_fontsize})  # 改为lower right，避免遮挡高准确率区域
ax.grid(axis='y', alpha=grid_alpha)
ax.set_ylim(y_min, y_max)

# Y轴网格间距
ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))

# 调整布局并保存
plt.tight_layout()
plt.savefig('C:/Users/jiyiy/Desktop/学位论文/论文插图/第五章/图3类型厂商识别准确率中文.png', dpi=dpi, bbox_inches='tight', format='png')
plt.show()