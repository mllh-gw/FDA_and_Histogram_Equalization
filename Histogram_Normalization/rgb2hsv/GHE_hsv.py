import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
from eval_divergence import calculate_evaluate_divergence,plot_analysis

path = 'sample_data/original/1_7000_0.png'
img_bgr = cv.imread(path, cv.IMREAD_COLOR)

if img_bgr is None:
    print("Error: Could not read image. Please check the file path.")
    exit()


img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)

v_ghe = cv.equalizeHist(v)

print(f"Original V-channel range: [{v.min()}, {v.max()}] | Mean: {v.mean():.2f}")
print(f"GHE V-channel range:      [{v_ghe.min()}, {v_ghe.max()}] | Mean: {v_ghe.mean():.2f}")

img_hsv_ghe = cv.merge([h, s, v_ghe])
img_bgr_ghe = cv.cvtColor(img_hsv_ghe, cv.COLOR_HSV2BGR)

results=calculate_evaluate_divergence(v, v_ghe)
plot_analysis(img_bgr, img_bgr_ghe, results)
# plt.figure(figsize=(14, 10))

# plt.subplot(2, 2, 1)
# plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
# plt.title('Original (Color)')
# plt.axis('off')


# plt.subplot(2, 2, 2)
# plt.imshow(cv.cvtColor(img_bgr_ghe, cv.COLOR_BGR2RGB))
# plt.title('GHE Enhanced (Color - HSV V-channel)')
# plt.axis('off')

# plt.subplot(2, 2, 3)
# hist_orig, bins_orig = np.histogram(v.flatten(), 256, [0, 256])
# cdf_orig = hist_orig.cumsum()
# cdf_norm_orig = cdf_orig * hist_orig.max() / cdf_orig.max()
# plt.plot(cdf_norm_orig, color='blue', label='CDF')
# plt.hist(v.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Histogram')
# plt.title('Original V-Channel Hist & CDF')
# plt.legend(loc='upper left')


# plt.subplot(2, 2, 4)
# hist_ghe, bins_ghe = np.histogram(v_ghe.flatten(), 256, [0, 256])
# cdf_ghe = hist_ghe.cumsum()
# cdf_norm_ghe = cdf_ghe * hist_ghe.max() / cdf_ghe.max()
# plt.plot(cdf_norm_ghe, color='blue', label='CDF')
# plt.hist(v_ghe.flatten(), 256, [0, 256], color='b', alpha=0.5, label='Histogram')
# plt.title('GHE V-Channel Hist & CDF')
# plt.legend(loc='upper left')

# plt.tight_layout()
# plt.show()


output_path = 'Histrogram_Normalization_output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created folder: {output_path}")

comparison = np.hstack((img_bgr, img_bgr_ghe))
cv.imwrite(os.path.join(output_path, 'ghe_rgb2hsv_comparison.png'), comparison)
print(f"\n=== GHE RGB (HSV) Result Saved ===")