import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = r'C:\Users\Lenovo\Desktop\论文借鉴\实验结果\evaluation_actual.xlsx'
df = pd.read_excel(file_path)

# 筛选 Precision < 1 或 Recall < 1 的设备记录
filtered_df = df[(df['Precision'] < 1) | (df['Recall'] < 1)].copy()

# 将 Precision 和 Recall 转换为百分比
filtered_df.loc[:, 'Precision'] = filtered_df['Precision'] * 100
filtered_df.loc[:, 'Recall'] = filtered_df['Recall'] * 100

# 按设备名首字母降序排序（使其在图中从上到下按字母顺序排列）
filtered_df = filtered_df.sort_values(by='Device', ascending=False)

# 设置自定义颜色，Precision 为柔和橙色，Recall 为深海蓝色
colors = {'Precision': '#FFA07A', 'Recall': '#1B3A5E'}

# 设置图表风格和背景
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(15, 10))

# 仅绘制 Precision 和 Recall，设备名称作为纵坐标
bar_width = 0.6  # 调整柱状图宽度
filtered_df.set_index('Device')[['Precision', 'Recall']].plot(
    kind='barh',
    ax=ax,
    alpha=0.85,
    color=[colors['Precision'], colors['Recall']],
    zorder=3,
    edgecolor='black',
    width=bar_width
)

# 设置横坐标刻度为 10%, 20%, ..., 100%
plt.xticks(
    ticks=range(0, 110, 10),
    labels=[f'{i}%' for i in range(0, 110, 10)],
    rotation=0,
    fontsize=18
)

# 将横坐标限制在 0% 到 100% 之间
ax.set_xlim(0, 100)

# 设置图例更接近横坐标，并调整字体大小
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=18)

# 去除左侧的 "Device" 字符
ax.set_ylabel('')

# 添加横坐标的浅灰色参考线，并减少网格线密度
ax.grid(axis='x', color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)

# 为每个条形图添加数值标签
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=18, color='black')

# 调整字体和图表间距
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
plt.tight_layout()

# 保存图表为 SVG 格式
save_path = "precision_recall_comparison.svg"
plt.savefig(save_path, format='svg', dpi=300)
plt.show()

print(f"Bar chart saved as SVG at {save_path}")
