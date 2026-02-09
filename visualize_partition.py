import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# 視覺化不同 Threshold 下的區塊分割結果
# ============================================================

def generate_dummy_image(size=64):
    """生成與 C++ 相同的測試圖片"""
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if x < size // 2 and y < size // 2:
                img[y, x] = 10
            elif x > size // 2 and y > size // 2:
                img[y, x] = 240
            else:
                img[y, x] = (x * y) % 255
    return img

def compute_features(img, x, y, size):
    """計算區塊特徵"""
    block = img[y:y+size, x:x+size].astype(float)
    variance = np.var(block)
    
    grad_h = 0
    grad_v = 0
    for i in range(size):
        for j in range(size):
            if j + 1 < size:
                grad_h += abs(int(block[i, j]) - int(block[i, j+1]))
            if i + 1 < size:
                grad_v += abs(int(block[i, j]) - int(block[i+1, j]))
    grad_h /= (size * size)
    grad_v /= (size * size)
    
    return variance, grad_h, grad_v

def predict_split_ml(variance, grad_h, grad_v):
    """ML 模型預測（與 C++ 相同的邏輯）"""
    if variance <= 5141.600098:
        if grad_h <= 62.218750:
            if grad_h <= 18.218750:
                if variance <= 0.000000:
                    return False, 1.0000
                else:
                    return False, 0.8000
            else:
                return False, 1.0000
        else:
            return True, 1.0000
    else:
        if variance <= 5754.145020:
            if variance <= 5577.659912:
                return False, 0.6111
            else:
                return True, 0.8462
        else:
            if variance <= 8911.485352:
                return False, 0.9286
            else:
                return True, 1.0000

def compute_rdo(img, x, y, size, lambda_val=0.5):
    """計算 RDO 決策"""
    block = img[y:y+size, x:x+size].astype(float)
    pred_value = np.mean(block)
    
    # 不切分的 Cost
    distortion_no_split = np.sum((block - pred_value) ** 2)
    variance = np.var(block)
    rate_no_split = np.log2(variance + 1) / 8.0 * size * size
    cost_no_split = distortion_no_split + lambda_val * rate_no_split
    
    # 切分的 Cost
    if size > 4:
        half = size // 2
        total_distortion = 0
        total_rate = 0
        
        for dy in range(2):
            for dx in range(2):
                sx, sy = x + dx * half, y + dy * half
                sub_block = img[sy:sy+half, sx:sx+half].astype(float)
                sub_pred = np.mean(sub_block)
                total_distortion += np.sum((sub_block - sub_pred) ** 2)
                sub_var = np.var(sub_block)
                total_rate += np.log2(sub_var + 1) / 8.0 * half * half
        
        cost_split = total_distortion + lambda_val * (total_rate + 2.0)
    else:
        cost_split = float('inf')
    
    return cost_split < cost_no_split

def partition_hybrid(img, x, y, size, threshold, blocks, use_fallback=True):
    """Hybrid 分割（與 C++ 邏輯相同）"""
    var, grad_h, grad_v = compute_features(img, x, y, size)
    should_split_ml, confidence = predict_split_ml(var, grad_h, grad_v)
    
    if confidence >= threshold:
        should_split = should_split_ml
    elif use_fallback:
        should_split = compute_rdo(img, x, y, size)
    else:
        should_split = should_split_ml
    
    if size <= 4:
        should_split = False
    
    if should_split:
        half = size // 2
        partition_hybrid(img, x, y, half, threshold, blocks, use_fallback)
        partition_hybrid(img, x + half, y, half, threshold, blocks, use_fallback)
        partition_hybrid(img, x, y + half, half, threshold, blocks, use_fallback)
        partition_hybrid(img, x + half, y + half, half, threshold, blocks, use_fallback)
    else:
        blocks.append((x, y, size))

def partition_c_model(img, x, y, size, blocks):
    """C Model 完整 RDO 分割"""
    should_split = compute_rdo(img, x, y, size)
    
    if size <= 4:
        should_split = False
    
    if should_split:
        half = size // 2
        partition_c_model(img, x, y, half, blocks)
        partition_c_model(img, x + half, y, half, blocks)
        partition_c_model(img, x, y + half, half, blocks)
        partition_c_model(img, x + half, y + half, half, blocks)
    else:
        blocks.append((x, y, size))

def draw_partition(ax, img, blocks, title):
    """繪製分割結果"""
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    
    # 依照區塊大小著色
    colors = {
        64: '#FF6B6B',  # 紅色
        32: '#4ECDC4',  # 青色
        16: '#45B7D1',  # 藍色
        8:  '#96CEB4',  # 綠色
        4:  '#FFEAA7',  # 黃色
    }
    
    for (x, y, size) in blocks:
        color = colors.get(size, '#FFFFFF')
        rect = patches.Rectangle((x-0.5, y-0.5), size, size, 
                                  linewidth=1.5, edgecolor=color, 
                                  facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

# ============================================================
# 主程式
# ============================================================

img = generate_dummy_image(64)

# 測試不同 threshold
thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 1.01]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 1. 原始圖片
ax = axes[0, 0]
ax.imshow(img, cmap='gray', vmin=0, vmax=255)
ax.set_title('Original Image', fontsize=10, fontweight='bold')
ax.axis('off')

# 2. C Model (Ground Truth)
blocks_c = []
partition_c_model(img, 0, 0, 64, blocks_c)
draw_partition(axes[0, 1], img, blocks_c, f'C Model (RDO)\n{len(blocks_c)} blocks')

# 3-8. 不同 Threshold
for idx, thresh in enumerate(thresholds):
    row = (idx + 2) // 4
    col = (idx + 2) % 4
    
    blocks = []
    partition_hybrid(img, 0, 0, 64, thresh, blocks, use_fallback=True)
    
    # 計算準確度
    match = len(set(blocks) & set(blocks_c))
    accuracy = 100 * match / len(blocks_c) if blocks_c else 0
    
    title = f'Threshold={thresh}\n{len(blocks)} blocks, {accuracy:.1f}% acc'
    draw_partition(axes[row, col], img, blocks, title)

# 圖例
legend_elements = [
    patches.Patch(facecolor='none', edgecolor='#FF6B6B', linewidth=2, label='64×64'),
    patches.Patch(facecolor='none', edgecolor='#4ECDC4', linewidth=2, label='32×32'),
    patches.Patch(facecolor='none', edgecolor='#45B7D1', linewidth=2, label='16×16'),
    patches.Patch(facecolor='none', edgecolor='#96CEB4', linewidth=2, label='8×8'),
    patches.Patch(facecolor='none', edgecolor='#FFEAA7', linewidth=2, label='4×4'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)

plt.suptitle('Block Partitioning Results at Different Confidence Thresholds', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('partition_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved: partition_comparison.png')

# ============================================================
# 額外：ML-Only vs C Model 比較
# ============================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

# ML-Only (無 fallback)
blocks_ml = []
partition_hybrid(img, 0, 0, 64, 0.0, blocks_ml, use_fallback=False)
draw_partition(axes2[0], img, blocks_ml, f'ML-Only\n{len(blocks_ml)} blocks')

# C Model
draw_partition(axes2[1], img, blocks_c, f'C Model (Ground Truth)\n{len(blocks_c)} blocks')

# Hybrid (最佳平衡)
blocks_best = []
partition_hybrid(img, 0, 0, 64, 0.9, blocks_best, use_fallback=True)
match = len(set(blocks_best) & set(blocks_c))
accuracy = 100 * match / len(blocks_c)
draw_partition(axes2[2], img, blocks_best, f'Hybrid (θ=0.9)\n{len(blocks_best)} blocks, {accuracy:.1f}% acc')

plt.suptitle('ML-Only vs C Model vs Hybrid Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('partition_ml_vs_c.png', dpi=150, bbox_inches='tight')
print('✓ Saved: partition_ml_vs_c.png')

print('\nDone!')
