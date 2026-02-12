import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. Load the image (color only)
path = 'sample_data/original/1_7000_0.png'
img_bgr = cv.imread(path, cv.IMREAD_COLOR)

if img_bgr is None:
    print("Error: Could not read image.")
    exit()

def balanced_metal_distribution(img, target_min=30, target_max=180):
    # 1. Identify the metal range
    mask = (img > 5) & (img < 250)
    metal_pixels = img[mask]

    if len(metal_pixels) == 0:
        return img

    in_min = np.percentile(metal_pixels, 1)
    in_max = np.percentile(metal_pixels, 99)

    dark_threshold = in_min + (in_max - in_min) * 0.3
    mid_threshold = in_min + (in_max - in_min) * 0.7

    xp = [0, in_min, dark_threshold, mid_threshold, in_max, 255]
    fp = [0,
          target_min,
          target_min + (target_max - target_min) * 0.5,
          target_min + (target_max - target_min) * 0.8,
          target_max,
          255]
    img_interp = np.interp(img, xp, fp).astype(np.uint8)

    final = img.copy()
    final[mask] = img_interp[mask]
    return final

# --- Convert to LAB color space ---
# LAB separates lightness (L) from color (a, b)
# Processing only L preserves the original color information
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
L, a, b = cv.split(img_lab)

# --- Apply piecewise contrast stretching to L-channel ---
L_enhanced = balanced_metal_distribution(L, target_min=30, target_max=180)

# --- Reconstruct BGR image ---
img_lab_enhanced = cv.merge([L_enhanced, a, b])
img_bgr_enhanced = cv.cvtColor(img_lab_enhanced, cv.COLOR_LAB2BGR)

# --- Visualization with Histograms ---
plt.figure(figsize=(14, 10))

# Original Image (Color)
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
plt.title('Original (Dark)')

# Enhanced Image (Color)
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(img_bgr_enhanced, cv.COLOR_BGR2RGB))
plt.title('Enhanced RGB (Piecewise Stretching on L)')

# Histogram of Original L-Channel
plt.subplot(2, 2, 3)
plt.hist(L.flatten(), 256, [1, 256], color='red', alpha=0.7)
plt.axvline(90, color='black', linestyle='--')
plt.title('Original L-Channel Histogram')
plt.grid(axis='y', alpha=0.3)

# Histogram of Enhanced L-Channel
plt.subplot(2, 2, 4)
plt.hist(L_enhanced.flatten(), 256, [1, 256], color='blue', alpha=0.7)
plt.axvline(180, color='black', linestyle='--')
plt.title('Enhanced L-Channel Histogram (30-180)')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# --- Save result and create side-by-side comparison ---
output_path = 'Histrogram_Normalization_output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created folder: {output_path}")

comparison = np.hstack((img_bgr, img_bgr_enhanced))
cv.imwrite(output_path + 'piecewise_rgb_enhanced.png', img_bgr_enhanced)
cv.imwrite(output_path + 'piecewise_rgb_comparison.png', comparison)
print("RGB piecewise contrast stretching complete.")
print("Side-by-side comparison saved as 'piecewise_rgb_comparison.png'")
