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
    print("Error: Could not read image.")
    exit()

# --- Convert to HSV color space ---
# HSV separates hue (H) from color (S, V)
# Processing only V preserves the original color information
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
H, S, V = cv.split(img_hsv)

# --- ANALYSIS: Check the original V-channel dynamic range ---
print(f"Original V-channel range: [{V.min()}, {V.max()}]")
print(f"Original V-channel mean: {V.mean():.2f}")
print(f"Original V-channel std: {V.std():.2f}")

# --- CLAHE on V-channel ---
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
V_clahe = clahe.apply(V)

print(f"CLAHE V-channel range: [{V_clahe.min()}, {V_clahe.max()}]")
print(f"CLAHE V-channel mean: {V_clahe.mean():.2f}")
print(f"CLAHE V-channel std: {V_clahe.std():.2f}")

# --- Reconstruct BGR image ---
img_hsv_clahe= cv.merge([H, S, V_clahe])
img_bgr_hsv_clahe = cv.cvtColor(img_hsv_clahe, cv.COLOR_HSV2BGR)

# Evaluate Divergence
results=calculate_evaluate_divergence(V, V_clahe)
plot_analysis(img_bgr, img_bgr_hsv_clahe, results)


output_path = 'Histrogram_Normalization_output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

comparison = np.hstack((img_bgr, img_bgr_hsv_clahe))
cv.imwrite(output_path + 'clahe_rgb2hsv_comparison.png', comparison)
print("\n=== CLAHE RGB Result Saved ===")