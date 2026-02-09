import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_image(size=64):
    """
    生成與 C++ 相同的測試圖片
    - 左上角：平滑區 (低變異，值=10)
    - 右下角：平滑區 (低變異，值=240)
    - 其他區域：紋理區 (高變異，(x*y) % 255)
    """
    img = np.zeros((size, size), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            if x < size // 2 and y < size // 2:
                img[y, x] = 10  # 平滑區 (深色)
            elif x > size // 2 and y > size // 2:
                img[y, x] = 240  # 平滑區 (淺色)
            else:
                img[y, x] = (x * y) % 255  # 紋理區
    
    return img

# 生成圖片
img = generate_dummy_image(64)

# === 圖 1: 原始圖片 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
im1 = ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Dummy Test Image (64×64)', fontsize=12, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1, label='Pixel Value')

# 標註區域
ax1.axhline(y=32, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.axvline(x=32, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(8, 8, 'Smooth\n(Low Var)', fontsize=9, color='white', ha='center')
ax1.text(48, 48, 'Smooth\n(Low Var)', fontsize=9, color='black', ha='center')
ax1.text(48, 16, 'Texture\n(High Var)', fontsize=9, color='white', ha='center')
ax1.text(16, 48, 'Texture\n(High Var)', fontsize=9, color='white', ha='center')

# === 圖 2: 變異數分布 ===
ax2 = axes[1]

# 計算每個 8x8 區塊的變異數
block_size = 8
var_map = np.zeros((64 // block_size, 64 // block_size))

for by in range(64 // block_size):
    for bx in range(64 // block_size):
        block = img[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
        var_map[by, bx] = np.var(block)

im2 = ax2.imshow(var_map, cmap='hot', interpolation='nearest')
ax2.set_title('Variance Map (8×8 blocks)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Block X')
ax2.set_ylabel('Block Y')
plt.colorbar(im2, ax=ax2, label='Variance')

# === 圖 3: 梯度強度 ===
ax3 = axes[2]

# 計算梯度
grad_h = np.abs(np.diff(img.astype(float), axis=1))
grad_v = np.abs(np.diff(img.astype(float), axis=0))
grad_combined = np.zeros_like(img, dtype=float)
grad_combined[:-1, :-1] = (grad_h[:-1, :] + grad_v[:, :-1]) / 2

im3 = ax3.imshow(grad_combined, cmap='viridis')
ax3.set_title('Gradient Magnitude', fontsize=12, fontweight='bold')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
plt.colorbar(im3, ax=ax3, label='Gradient')

plt.tight_layout()
plt.savefig('dummy_image_analysis.png', dpi=150, bbox_inches='tight')
print('✓ Saved: dummy_image_analysis.png')

# === 單獨保存原始圖片 ===
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Test Image (64×64)', fontsize=14)
plt.axis('off')
plt.savefig('dummy_image.png', dpi=150, bbox_inches='tight')
print('✓ Saved: dummy_image.png')

# === 顯示統計資訊 ===
print('\n=== Image Statistics ===')
print(f'Size: {img.shape[0]}×{img.shape[1]}')
print(f'Min pixel: {img.min()}')
print(f'Max pixel: {img.max()}')
print(f'Mean: {img.mean():.2f}')
print(f'Std Dev: {img.std():.2f}')
print(f'\nVariance by region:')
print(f'  Top-Left (smooth):  {np.var(img[:32, :32]):.2f}')
print(f'  Top-Right (texture): {np.var(img[:32, 32:]):.2f}')
print(f'  Bottom-Left (texture): {np.var(img[32:, :32]):.2f}')
print(f'  Bottom-Right (smooth): {np.var(img[32:, 32:]):.2f}')
