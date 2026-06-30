import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, truncnorm
from scipy.optimize import brentq
import matplotlib as mpl

# 设置更专业的字体与风格
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['lines.linewidth'] = 1.2

np.random.seed(42)

# 参数设置
mu_pos, sigma_pos = 87, 6.7
mu_neg, sigma_neg = 57, 11
x_min, x_max = 35, 95
x_right_limit = 82

# 正负样本生成
positive = np.clip(np.random.normal(mu_pos, sigma_pos, 500), x_min, x_max)
a, b = (x_min - mu_neg) / sigma_neg, (x_right_limit - mu_neg) / sigma_neg
negative = truncnorm.rvs(a, b, loc=mu_neg, scale=sigma_neg, size=500)

# KDE 估计
kde_pos = gaussian_kde(positive)
kde_neg = gaussian_kde(negative)
x = np.linspace(x_min, x_max, 500)

def diff(x): return kde_pos(x) - kde_neg(x)

# 查找交点
diff_values = diff(x)
sign_changes = np.where(np.diff(np.sign(diff_values)))[0]

if len(sign_changes) == 0:
    print("No intersection found.")
    x0 = None
else:
    idx = sign_changes[0]
    x0 = brentq(diff, x[idx], x[idx + 1])
    y0 = kde_pos(x0)[0]

    # 专业配色
    color_pos = "#1f77b4"  # 深蓝
    color_neg = "#d62728"  # 深红
    threshold_color = "black"
    dot_color = "#4B0082"  # 靛紫点

    # 绘图
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(x, kde_pos(x), label='Matched Positive Sample', color=color_pos)
    ax.fill_between(x, kde_pos(x), color=color_pos, alpha=0.15)

    ax.plot(x, kde_neg(x), label='Matched Negative Sample', color=color_neg)
    ax.fill_between(x, kde_neg(x), color=color_neg, alpha=0.15)

    ax.axvline(x0, color=threshold_color, linestyle='--', linewidth=1.2,
               label=r'Threshold $\theta$ = {:.2f}'.format(x0))
    ax.scatter([x0], [y0], color=dot_color, s=30, zorder=5)

    # 标注点值
    ax.text(x0 + 0.8, y0, f"$x$ = {x0:.2f}\n$f(x)$ = {y0:.3f}",
            color=dot_color, verticalalignment='bottom', fontsize=11)

    ax.set_xlabel('Matching Score')
    ax.set_ylabel('Probability Density')
    # ax.set_title('Distribution of Matching Scores')  # 可移除，避免冗余

    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig("Distribution_of_Matching_Scores.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.show()
