import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Replace '.png' with your actual image path
img_original = cv.imread('sample_data/original/1_7000_0.png', cv.IMREAD_COLOR)  # Load original color image
img = cv.imread('sample_data/original/1_7000_0.png', cv.IMREAD_GRAYSCALE)     # Load grayscale for processing

if img is None or img_original is None:
    print("Error: Could not read image. Please check the file path.")
    exit()

# This is the industry-standard way to do it
img_opencv = cv.equalizeHist(img)

# --- Visualization ---
plt.figure(figsize=(14, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

# Equalized Image
plt.subplot(2, 2, 2)
plt.imshow(img_opencv, cmap='gray')
plt.title('Equalized')
plt.axis('off')

# Original Histogram + CDF
plt.subplot(2, 2, 3)
hist_orig, bins_orig = np.histogram(img.flatten(), 256, [0, 256])
cdf_orig = hist_orig.cumsum()
cdf_normalized_orig = cdf_orig * hist_orig.max() / cdf_orig.max() # Scale for visualization

plt.plot(cdf_normalized_orig, color='blue', label='CDF')
plt.hist(img.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Histogram')
plt.title('Original: Hist & CDF')
plt.legend(loc='upper left')

# Equalized Histogram + CDF
plt.subplot(2, 2, 4)
hist_eq, bins_eq = np.histogram(img_opencv.flatten(), 256, [0, 256])
cdf_eq = hist_eq.cumsum()
cdf_normalized_eq = cdf_eq * hist_eq.max() / cdf_eq.max() # Scale for visualization

plt.plot(cdf_normalized_eq, color='blue', label='CDF')
plt.hist(img_opencv.flatten(), 256, [0, 256], color='b', alpha=0.5, label='Histogram')
plt.title('Equalized: Hist & CDF')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Save the side-by-side comparison: [Original Color | Equalized Grayscale]
os.makedirs('Equalize_Output', exist_ok=True)
img_opencv_3ch = cv.cvtColor(img_opencv, cv.COLOR_GRAY2BGR)
res = np.hstack((img_original, img_opencv_3ch)) 
cv.imwrite('Equalize_Output/comparism_equalize.png', res)
print("Result saved as 'comparism_equalize.png'")