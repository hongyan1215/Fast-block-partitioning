import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 讀取 Pareto 數據
df = pd.read_csv('pareto_data.csv')

# 設定風格
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ============================================
# 圖 1: Pareto Frontier (Speedup vs Accuracy)
# ============================================
ax1 = axes[0]

# 繪製曲線
ax1.plot(df['speedup'], df['accuracy'], 'o-', color='#2E86AB', 
         linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)

# 標註每個點的 threshold
for i, row in df.iterrows():
    offset = (0.08, 2) if row['threshold'] != 0.6 else (-0.15, -5)
    ax1.annotate(f"θ={row['threshold']}", 
                 (row['speedup'], row['accuracy']),
                 textcoords="offset points", xytext=offset,
                 fontsize=9, color='#333')

# 標記最佳平衡點
best_idx = 4  # threshold = 0.9
ax1.scatter([df.iloc[best_idx]['speedup']], [df.iloc[best_idx]['accuracy']], 
            s=200, c='#E94F37', zorder=5, marker='*', label='Best Balance (θ=0.9)')

ax1.set_xlabel('Speedup (×)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Pareto Frontier: Speedup vs Accuracy', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.set_xlim(0.8, 5)
ax1.set_ylim(0, 100)

# 添加區域標註
ax1.axhspan(80, 100, alpha=0.1, color='green', label='High Accuracy Zone')
ax1.axvspan(2, 5, alpha=0.1, color='blue', label='High Speed Zone')
ax1.text(3.5, 25, 'Fast but\nInaccurate', fontsize=10, ha='center', color='#666')
ax1.text(1.2, 90, 'Accurate but\nSlow', fontsize=10, ha='center', color='#666')

# ============================================
# 圖 2: Threshold vs Metrics (雙軸圖)
# ============================================
ax2 = axes[1]
ax2_twin = ax2.twinx()

# Speedup 曲線
line1 = ax2.plot(df['threshold'], df['speedup'], 's-', color='#2E86AB', 
                  linewidth=2, markersize=8, label='Speedup')
ax2.set_xlabel('Confidence Threshold (θ)', fontsize=12)
ax2.set_ylabel('Speedup (×)', fontsize=12, color='#2E86AB')
ax2.tick_params(axis='y', labelcolor='#2E86AB')

# Accuracy 曲線
line2 = ax2_twin.plot(df['threshold'], df['accuracy'], 'o-', color='#E94F37', 
                       linewidth=2, markersize=8, label='Accuracy')
ax2_twin.set_ylabel('Accuracy (%)', fontsize=12, color='#E94F37')
ax2_twin.tick_params(axis='y', labelcolor='#E94F37')

# 合併圖例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='center right')

ax2.set_title('Effect of Confidence Threshold', fontsize=14, fontweight='bold')
ax2.set_xlim(0.45, 1.05)

# 標記建議區間
ax2.axvspan(0.8, 0.95, alpha=0.2, color='yellow', label='Recommended')
ax2.text(0.875, 1.5, 'Recommended\nZone', fontsize=9, ha='center', color='#666')

plt.tight_layout()
plt.savefig('pareto_frontier.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('pareto_frontier.pdf', bbox_inches='tight')

print("✓ Charts saved:")
print("  - pareto_frontier.png")
print("  - pareto_frontier.pdf")

# 顯示圖片
plt.show()
