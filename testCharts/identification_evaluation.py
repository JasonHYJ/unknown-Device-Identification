import re
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 Times New Roman
# rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 15
rcParams['axes.unicode_minus'] = False

# 时间点标签
time_points = ["300s", "600s", "1200s", "1500s", "1800s", "3600s", "7200s", ">7200s"]
time_count = {tp: 0 for tp in time_points}
total_samples = 0

# 文件路径
file_path = r"/home/hyj/unknownDeviceIdentification/testCharts/时间效率.txt"

# 正则匹配
normal_pattern = re.compile(r"在(\d+s)推测出结果\*(\d+)")
unknown_pattern = re.compile(r"持续识别从300s->7200s未推测出设备类型\*(\d+)")

# 数据处理
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if "：" not in line:
            continue
        per_device_counts = {tp: 0 for tp in time_points}
        sample_sum = 0

        for match in normal_pattern.findall(line):
            t, c = match
            if t in per_device_counts:
                per_device_counts[t] += int(c)
                sample_sum += int(c)

        umatch = unknown_pattern.search(line)
        if umatch:
            per_device_counts[">7200s"] += int(umatch.group(1))
            sample_sum += int(umatch.group(1))

        total_samples += sample_sum
        for tp in time_points:
            time_count[tp] += per_device_counts[tp]

# 百分比
percentages = [time_count[tp] / total_samples * 100 for tp in time_points]

# 开始绘图
fig, ax = plt.subplots(figsize=(10, 5.5))

# 使用更加“学术”的蓝色调
ax.plot(time_points, percentages, marker='o', linestyle='-', color='#1f4e79',
        linewidth=2.2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)

# 百分比标注
for x, y in zip(time_points, percentages):
    ax.text(x, y + 0.9, f"{y:.1f}%", ha='center', va='bottom', fontsize=12)

# 标签与标题
ax.set_xlabel("Time Point $T_i$ (seconds)", fontsize=15)
ax.set_ylabel("Sample Ratio (%)", fontsize=15)
ax.set_title("Device Identification Rate at Each Time Point", fontsize=17, pad=12)

# 坐标轴设置
ax.set_ylim(0, max(percentages) + 6)
ax.tick_params(axis='both', which='major', labelsize=12)

# 边框设置（去除顶部和右侧）
ax.spines['top'].set_visible(1.3)
ax.spines['right'].set_visible(1.3)
ax.spines['left'].set_linewidth(1.3)
ax.spines['bottom'].set_linewidth(1.3)

# 添加网格
ax.grid(True, linestyle='-', alpha=0.4)

# 布局与保存
plt.tight_layout(pad=1.5)
output_path = 'actual\device_identification_efficiency_final2.svg'
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
plt.show()
