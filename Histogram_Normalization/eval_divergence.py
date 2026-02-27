import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def calculate_evaluate_divergence(channel_orig, channel_enh, bins=256):
    """
    Calculates statistical metrics comparing two image channels.
    Returns: kl_score, js_divergence, ent_p, ent_q, kl_contributions, P, Q
    """
    # Calculate histograms
    hist_p, _ = np.histogram(channel_orig, bins=bins, range=(0, 256), density=True)
    hist_q, _ = np.histogram(channel_enh, bins=bins, range=(0, 256), density=True)
    
    # Add epsilon to prevent log(0) and division by zero
    P = hist_p + 1e-10
    Q = hist_q + 1e-10
    
    # Metric calculations (Base 2 for bits)
    kl_score = entropy(P, Q, base=2)
    # Jensen-Shannon distance to divergence: distance^2
    js_divergence = jensenshannon(P, Q, base=2) ** 2
    
    ent_p = entropy(P, base=2)
    ent_q = entropy(Q, base=2)
    
    # Component-wise KL contribution: P(x) * log2(P(x)/Q(x))
    kl_contributions = P * np.log2(P / Q)
    
    return {
        "kl_score": kl_score,
        "js_divergence": js_divergence,
        "entropy_orig": ent_p,
        "entropy_enh": ent_q,
        "kl_contributions": kl_contributions,
        "p_dist": P,
        "q_dist": Q
    }

def plot_analysis(img_bgr_orig, img_bgr_enh, metrics):
    plt.figure(figsize=(16, 10))
    bins = np.arange(256)
    
    #original 
    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(img_bgr_orig, cv.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=12)
    plt.axis('off')

    # enhanced
    plt.subplot(2, 3, 2)
    plt.imshow(cv.cvtColor(img_bgr_enh, cv.COLOR_BGR2RGB))
    plt.title('Enhanced Image', fontsize=12)
    plt.axis('off')

    # text
    plt.subplot(2, 3, 3)
    plt.axis('off')
    text = (
        f"Metrics (bits)\n"
        f"{'-'*30}\n"
        f"KL Divergence:  {metrics['kl_score']:.4f}\n"
        f"JS Divergence:  {metrics['js_divergence']:.4f}\n\n"
        f"Original Entropy:   {metrics['entropy_orig']:.4f}\n"
        f"Enhanced Entropy:   {metrics['entropy_enh']:.4f}\n"
        # f"Entropy Gain:   {metrics['entropy_enh'] - metrics['entropy_orig']:.4f}"
    )
    plt.text(0.05, 0.5, text, fontsize=11, family='monospace', va='center')

    # histogram 
    # original channel
    plt.subplot(2, 2, 3)
    plt.fill_between(bins, metrics['p_dist'], color='red', alpha=0.3, label='Original (P)')
    plt.plot(bins, metrics['p_dist'], color='red', lw=1)
    plt.fill_between(bins, metrics['q_dist'], color='blue', alpha=0.3, label='Enhanced (Q)')
    plt.plot(bins, metrics['q_dist'], color='blue', lw=1)
    plt.title('L-channel Probability Distributions', fontsize=12)   # LAB
    # plt.title('V-channel Probability Distributions', fontsize=12) # HSV
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # kl contribution
    plt.subplot(2, 2, 4)
    plt.bar(bins, metrics['kl_contributions'], color='coral', width=1.0, edgecolor='none')
    plt.axhline(0, color='black', lw=0.8)
    plt.title('KL Contribution: $P(x) \log_2(P(x)/Q(x))$', fontsize=12)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Contribution (bits)')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    plt.show()