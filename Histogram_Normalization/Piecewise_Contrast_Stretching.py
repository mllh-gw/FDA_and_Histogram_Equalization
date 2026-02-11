import os 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. Load the image
path = 'sample_data/original/1_7000_0.png'
img_original = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)

if img_gray is None:
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
          target_min + (target_max - target_min) * 0.5, # Smooth mid-point 1
          target_min + (target_max - target_min) * 0.8, # Smooth mid-point 2
          target_max, 
          255]
    img_interp = np.interp(img, xp, fp).astype(np.uint8)
    
    final = img.copy()
    final[mask] = img_interp[mask]
    return final

# --- Execute and Visualization ---
img_balanced = balanced_metal_distribution(img_gray, target_min=30, target_max=180)
# img_balanced = balanced_metal_distribution(img_gray, target_min=10, target_max=140)

# --- Visualization with Histograms ---
plt.figure(figsize=(14, 10))

# Result Image
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
plt.title('Original (Dark)')

plt.subplot(2, 2, 2)
plt.imshow(img_balanced, cmap='gray', vmin=0, vmax=255)
plt.title('Balanced Distribution (Target Max 180)')

# Histogram of Original (Note the narrow 0-90 spike)
plt.subplot(2, 2, 3)
plt.hist(img_gray.flatten(), 256, [1, 256], color='red', alpha=0.7)
plt.axvline(90, color='black', linestyle='--')
plt.title('Original Histogram (Narrow 0-90)')
plt.grid(axis='y', alpha=0.3)

# Histogram of Balanced (Note the wider distribution)
plt.subplot(2, 2, 4)
plt.hist(img_balanced.flatten(), 256, [1, 256], color='blue', alpha=0.7)
plt.axvline(180, color='black', linestyle='--')
plt.title('New Histogram (Distributed 30-180)')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Save result and create side-by-side comparison
img_balanced_3ch = cv.cvtColor(img_balanced, cv.COLOR_GRAY2BGR)

# Stack original RGB image with enhanced result
comparison = np.hstack((img_original, img_balanced_3ch))

output_path = 'Histrogram_Normalization_output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created folder: {output_path}")

cv.imwrite(output_path + 'balanced_metal_distribution.png', img_balanced)
cv.imwrite(output_path + 'balanced_comparison.png', comparison)
print("Saved. Check the histogram to see the 0-90 range expanded to 30-180.")
print("Side-by-side comparison saved as 'balanced_comparison.png'")