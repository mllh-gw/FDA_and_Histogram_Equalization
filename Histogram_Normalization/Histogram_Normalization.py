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
    """
    Spreads the 0-90 metal range into a balanced 30-180 range.
    30 keeps the background/shadows visible.
    180 makes the silver 'pop' without being 'too white'.
    """
    # Identify the metal range (ignore 0 and 255 noise)
    mask = (img > 5) & (img < 250)
    metal_pixels = img[mask]

    if len(metal_pixels) == 0:
        return img

    # Use percentiles to find the 'true' metal bounds
    in_min = np.percentile(metal_pixels, 1)  
    in_max = np.percentile(metal_pixels, 99) 

    # Multiplicative Enhancement (preserves color characteristics)
    # Instead of linear stretch, use multiplicative factors
    img_float = img.astype(np.float32)
    
    # Calculate enhancement factor for each pixel
    # This preserves the original color characteristics better than linear stretch
    enhancement_factor = np.ones_like(img_float)
    
    # Apply stronger enhancement to darker areas, less to brighter areas
    dark_threshold = in_min + (in_max - in_min) * 0.3
    mid_threshold = in_min + (in_max - in_min) * 0.7
    
    dark_areas = img_float < dark_threshold
    mid_areas = (img_float >= dark_threshold) & (img_float < mid_threshold)
    bright_areas = img_float >= mid_threshold
    
    enhancement_factor[dark_areas] = 1.8  # Boost dark areas
    enhancement_factor[mid_areas] = 1.3   # Moderate boost for mid areas
    enhancement_factor[bright_areas] = 1.0  # Keep bright areas as-is
    
    # Apply enhancement with clipping to avoid over-brightening
    res = img_float * enhancement_factor
    res = np.clip(res, 0, 255).astype(np.uint8)
    
    # We apply the transformation only to the workpiece to keep background natural
    final = img.copy()
    final[mask] = res[mask]
    return final

# --- Execute ---
# target_max=180 is the "Silver" sweet spot. 
img_balanced = balanced_metal_distribution(img_gray, target_min=30, target_max=180)

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
# Convert enhanced grayscale to 3-channel for stacking
img_balanced_3ch = cv.cvtColor(img_balanced, cv.COLOR_GRAY2BGR)

# Stack original RGB image with enhanced result
os.makedirs('Normalized_Output', exist_ok=True)
comparison = np.hstack((img_original, img_balanced_3ch))
cv.imwrite('Normalized_Output/balanced_metal_distribution.png', img_balanced)
cv.imwrite('Normalized_Output/balanced_comparison.png', comparison)
print("Saved. Check the histogram to see the 0-90 range expanded to 30-180.")
print("Side-by-side comparison saved as 'balanced_comparison.png'")