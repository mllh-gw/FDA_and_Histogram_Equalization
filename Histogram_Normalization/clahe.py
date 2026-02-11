import os 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = 'sample_data/original/1_7000_0.png'
img_original = cv.imread(path, cv.IMREAD_COLOR)  # Load original color image
img = cv.imread(path, cv.IMREAD_GRAYSCALE)     # Load grayscale for processing

if img is None or img_original is None:
    print("Error: Could not read image. Please check the file path.")
    exit()

# --- ANALYSIS: Check the original dynamic range ---
print(f"Original image range: [{img.min()}, {img.max()}]")
print(f"Original image mean: {img.mean():.2f}")
print(f"Original image std: {img.std():.2f}")

# --- CLAHE METHOD ---
# CLAHE = Contrast Limited Adaptive Histogram Equalization
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)

print(f"CLAHE result range: [{img_clahe.min()}, {img_clahe.max()}]")
print(f"CLAHE result mean: {img_clahe.mean():.2f}")
print(f"CLAHE result std: {img_clahe.std():.2f}")

# --- VISUALIZATION: Original vs CLAHE ---
plt.figure(figsize=(12, 8))

# Original Image (Color)
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
plt.title('Original (Color)')

# CLAHE Result
plt.subplot(2, 2, 2)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE Enhanced')

# Original Histogram
plt.subplot(2, 2, 3)
plt.hist(img.flatten(), bins=256, range=(0, 256), color='r', alpha=0.7)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# CLAHE Histogram
plt.subplot(2, 2, 4)
plt.hist(img_clahe.flatten(), bins=256, range=(0, 256), color='b', alpha=0.7)
plt.title('CLAHE Histogram')
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

img_clahe_3ch = cv.cvtColor(img_clahe, cv.COLOR_GRAY2BGR) 
comparison = np.hstack((img_original, img_clahe_3ch)) 
cv.imwrite(output_path + 'clahe_comparison.png', comparison)
print("\n=== CLAHE Result Saved ===")

