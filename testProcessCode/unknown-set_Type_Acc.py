import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ---------- 字体设置 ----------
plt.rcParams['font.family'] = 'Times New Roman'      # 英文/数字用 Times New Roman
plt.rcParams['axes.unicode_minus'] = False           # 解决负号显示
# -----------------------------

csv_text = """Method,Run,UnknownTypeAcc,UnknownManuAcc,MUR,UnknownSPS
EarlyConcat,1,0.9536,0.9418,0.8100,1500
EarlyConcat,2,0.9516,0.9428,0.7950,1500
EarlyConcat,3,0.9544,0.9427,0.8000,1500
EarlyConcat,4,0.9536,0.9438,0.7920,1500
EarlyConcat,5,0.9546,0.9448,0.8000,1500
AttentionFusion,1,0.9611,0.9452,0.8220,1200
AttentionFusion,2,0.9581,0.9482,0.8040,1200
AttentionFusion,3,0.9551,0.9502,0.8120,1200
AttentionFusion,4,0.9611,0.9492,0.8000,1200
AttentionFusion,5,0.9621,0.9512,0.8100,1200
LHDI,1,0.9206,0.8973,0.8580,3800
LHDI,2,0.9156,0.8943,0.8460,3800
LHDI,3,0.9246,0.9000,0.8530,3800
LHDI,4,0.9186,0.8983,0.8440,3800
LHDI,5,0.9266,0.9003,0.8500,3800
HSGAN-IoT,1,0.9401,0.9169,0.8870,3500
HSGAN-IoT,2,0.9371,0.9159,0.8770,3500
HSGAN-IoT,3,0.9341,0.9139,0.8840,3500
HSGAN-IoT,4,0.9391,0.9189,0.8740,3500
HSGAN-IoT,5,0.9411,0.9209,0.8800,3500
Ours,1,0.9702,0.9791,0.9598,4200
Ours,2,0.9682,0.9691,0.9498,4200
Ours,3,0.9762,0.9751,0.9588,4200
Ours,4,0.9762,0.9771,0.9538,4200
Ours,5,0.9802,0.9801,0.9558,4200
"""

dpi = 300

df = pd.read_csv(StringIO(csv_text))
df = df.sort_values(["Method", "Run"])

methods = ["EarlyConcat", "AttentionFusion", "LHDI", "HSGAN-IoT", "Ours"]
marker_map = {
    "EarlyConcat": "o",      # circle
    "AttentionFusion": "^",  # triangle up
    "LHDI": "s",             # square
    "HSGAN-IoT": "D",        # diamond
    "Ours": "X"              # x-filled
}

linestyle_map = {
    "EarlyConcat": "--",
    "AttentionFusion": "--",
    "LHDI": "-.",
    "HSGAN-IoT": ":",
    "Ours": "-"   # 也可以改成 "--" 或 "-."，看你希望它更显眼还是更统一
}

# 颜色映射：符合顶级期刊审美的色盲友好方案
color_map = {
    "EarlyConcat": "#1f77b4",   # 蓝色
    "AttentionFusion": "#2ca02c", # 绿色
    "LHDI": "#9467bd",           # 紫色
    "HSGAN-IoT": "#ff7f0e",      # 橙色
    "Ours": "#d62728"            # 深红色，突出显示
}

plt.figure()
for m in methods:
    sub = df[df["Method"] == m].sort_values("Run")
    plt.plot(
        sub["Run"], sub["UnknownTypeAcc"],
        marker=marker_map[m],
        linestyle=linestyle_map[m],
        color=color_map[m],      # 添加颜色
        linewidth=1.5,
        markersize=6,
        label=m
    )

# 修改：坐标轴标签改为中文，并指定宋体
plt.xlabel("轮次", fontsize=13, fontfamily='SimSun')
plt.ylabel("未知集类型准确率", fontsize=13, fontfamily='SimSun')
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0.85, 1.0)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.text(0.5, -0.18, "(a)类型识别结果", transform=plt.gca().transAxes, ha='center', fontsize=14, fontfamily='SimSun')
plt.legend()    # 图例标签为英文，默认使用 Times New Roman
plt.tight_layout()
plt.savefig('C:/Users/jiyiy/Desktop/学位论文/论文插图/第五章/图5类型识别准确率对比折线图中文.pdf', dpi=dpi, bbox_inches='tight', format='pdf')
plt.show()
