# @Author: Ming
# @Date: 2025/09/26
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm

# ===================== 字体设置 =====================
# 方法 1: 直接用名字（如果系统能识别 Times New Roman）
# mpl.rcParams['font.family'] = 'Times New Roman'

# 方法 2: 强制加载字体文件（推荐，确保论文输出严格使用 Times New Roman）
# Windows 默认路径：C:\Windows\Fonts\times.ttf 或 timesnewroman.ttf
font_path = r"C:\Windows\Fonts\times.ttf"   # 根据实际情况修改路径
times_new_roman = fm.FontProperties(fname=font_path)

# 设置 seaborn 风格
sns.set(style="whitegrid")

# 设置 matplotlib 全局字体和样式
mpl.rcParams['font.family'] = times_new_roman.get_name()
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['lines.linewidth'] = 1.2

# ===================== 数据读取 =====================
file_path = r'C:\Users\Lenovo\Desktop\论文借鉴\实验结果\evaluation_actual.xlsx'
df = pd.read_excel(file_path)

# 筛选 Precision < 1 或 Recall < 1 的设备
filtered_df = df[(df['Precision'] < 1) | (df['Recall'] < 1)].copy()

# 转换为百分比
filtered_df.loc[:, 'Precision'] = filtered_df['Precision'] * 100
filtered_df.loc[:, 'Recall'] = filtered_df['Recall'] * 100

# 按设备名首字母降序
filtered_df = filtered_df.sort_values(by='Device', ascending=False)

# ===================== 绘图 =====================
colors = {'Precision': '#FFA07A', 'Recall': '#1B3A5E'}

fig, ax = plt.subplots(figsize=(15, 10))

bar_width = 0.6
filtered_df.set_index('Device')[['Precision', 'Recall']].plot(
    kind='barh',
    ax=ax,
    alpha=0.85,
    color=[colors['Precision'], colors['Recall']],
    zorder=3,
    edgecolor='black',
    width=bar_width
)

# 横坐标刻度
plt.xticks(
    ticks=range(0, 110, 10),
    labels=[f'{i}%' for i in range(0, 110, 10)],
    rotation=0,
    fontsize=13
)

ax.set_xlim(0, 100)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=12)
ax.set_ylabel('')
ax.grid(axis='x', color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)

# 数值标签
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=13, color='black')

ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)

plt.tight_layout()

# ===================== 保存 PDF =====================
save_path = "precision_recall_comparison.svg"
plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
plt.show()

print(f"Bar chart saved as PDF at {save_path}")
