import matplotlib.pyplot as plt
import numpy as np

# Data for each condition
conditions_data = {
    'Type-Idle': {
        'features': ['max_pkt_len', 'payload_bytes_ratio', 'seq_len_mean', 'down_bytes', 'avg_pkt_len'],
        'mi_scores': [1.556, 1.510, 1.475, 1.471, 1.445]
    },
    'Type-Activity': {
        'features': ['max_pkt_len', 'min_pkt_len', 'seq_len_std', 'down_bytes', 'payload_bytes_total'],
        'mi_scores': [1.271, 1.070, 1.000, 0.982, 0.934]
    },
    'Brand-Idle': {
        'features': ['max_pkt_len', 'raw_nonzero_ratio', 'down_bytes', 'seq_len_mean', 'avg_pkt_len'],
        'mi_scores': [2.583, 2.427, 2.418, 2.415, 2.325]
    },
    'Brand-Activity': {
        'features': ['max_pkt_len', 'raw_byte_entropy', 'up_bytes', 'down_bytes', 'raw_row_sparsity_mean'],
        'mi_scores': [2.343, 2.082, 1.948, 1.902, 1.844]
    }

}

# Create 2x2 subplots with adjusted size
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 可调整：整体图形尺寸。调整整体图形宽度和高度
axes = axes.flatten()

# Colors for different conditions
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 可调整：颜色方案

# Plot each condition
for idx, (condition, data) in enumerate(conditions_data.items()):
    features = data['features']
    scores = data['mi_scores']
    
    # Create horizontal bar chart with thinner bars
    y_pos = np.arange(len(features))
    bar_height = 0.5  # 可调整：条形图的高度，值越小条形越细
    bars = axes[idx].barh(y_pos, scores, height=bar_height, color=colors[idx], alpha=0.8)   # alpha=0.8 - 调整条形图的透明度
    
    # Customize each subplot with larger fonts
    axes[idx].set_title(condition, fontsize=14, pad=10)  # 可调整：子图标题字体大小 pad=10 - 子图标题与图表之间的间距
    axes[idx].set_yticks(y_pos)
    axes[idx].set_yticklabels(features, fontsize=12)  # 可调整：Y轴标签字体大小
    axes[idx].set_xlabel('Mutual Information Score', fontsize=12)  # 可调整：X轴标签字体大小
    axes[idx].tick_params(axis='x', labelsize=11)  # 可调整：X轴刻度字体大小
    axes[idx].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    # for bar, score in zip(bars, scores):
    #     width = bar.get_width()
    #     axes[idx].text(width + 0.05, bar.get_y() + bar.get_height()/2, 
    #                   f'{score:.3f}', ha='left', va='center', fontsize=10)  # 可调整：条形图上数值标签字体大小

# Add overall title with adjusted position
plt.suptitle('Feature Importance Ranking Across Different Conditions', 
             fontsize=16, y=0.95)  # 可调整：总标题位置和字体大小   y=0.98 - 总标题的垂直位置

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 可调整：为总标题留出空间

# Save as PDF
plt.savefig('/home/hyj/unknownDeviceIdentification/testProcessCode/feature_importance_subplots.png', bbox_inches='tight') # 文件可以改为其他格式，如PNG、SVG等
plt.show()