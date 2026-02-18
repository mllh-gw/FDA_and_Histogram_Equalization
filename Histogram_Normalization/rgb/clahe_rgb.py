import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = 'sample_data/original/1_7000_0.png'
img_bgr = cv.imread(path, cv.IMREAD_COLOR)

if img_bgr is None:
    print("Error: Could not read image. Please check the file path.")
    exit()

# --- Convert to LAB color space ---
# LAB separates lightness (L) from color (a, b)
# Processing only L preserves the original color information
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
L, a, b = cv.split(img_lab)

# --- ANALYSIS: Check the original L-channel dynamic range ---
print(f"Original L-channel range: [{L.min()}, {L.max()}]")
print(f"Original L-channel mean: {L.mean():.2f}")
print(f"Original L-channel std: {L.std():.2f}")

# --- CLAHE on L-channel ---
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
L_clahe = clahe.apply(L)

print(f"CLAHE L-channel range: [{L_clahe.min()}, {L_clahe.max()}]")
print(f"CLAHE L-channel mean: {L_clahe.mean():.2f}")
print(f"CLAHE L-channel std: {L_clahe.std():.2f}")

# --- Reconstruct BGR image ---
img_lab_clahe = cv.merge([L_clahe, a, b])
img_bgr_clahe = cv.cvtColor(img_lab_clahe, cv.COLOR_LAB2BGR)

# --- VISUALIZATION: Original vs CLAHE (RGB) ---
plt.figure(figsize=(12, 8))

# Original Image (Color)
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
plt.title('Original (Color)')

# CLAHE Result (Color)
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(img_bgr_clahe, cv.COLOR_BGR2RGB))
plt.title('CLAHE Enhanced (Color)')

# Original L-Channel Histogram
plt.subplot(2, 2, 3)
plt.hist(L.flatten(), bins=256, range=(0, 256), color='r', alpha=0.7)
plt.title('Original L-Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# CLAHE L-Channel Histogram
plt.subplot(2, 2, 4)
plt.hist(L_clahe.flatten(), bins=256, range=(0, 256), color='b', alpha=0.7)
plt.title('CLAHE L-Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# --- SAVE COMPARISON RESULT ---
output_path = 'Histrogram_Normalization_output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created folder: {output_path}")

comparison = np.hstack((img_bgr, img_bgr_clahe))
cv.imwrite(output_path + 'clahe_rgb_comparison.png', comparison)
print("\n=== CLAHE RGB Result Saved ===")
