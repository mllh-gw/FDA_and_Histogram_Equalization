#!/usr/bin/env python3
"""
Run 9 experiments with different FDA and histogram combinations.
Source: CG images, Target: Real images
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path

def load_fda_demo():
    """Load FDA_demo.py functions."""
# Import FDA functions
    sys.path.append('.')
    from utils import FDA_source_to_target_np
    import numpy as np
    from PIL import Image
    
    def apply_fda_to_image(source_path, target_path, beta, output_path):
        """Apply FDA from target to source image."""
        try:
            # Load images
            src_img = Image.open(source_path).convert('RGB')
            trg_img = Image.open(target_path).convert('RGB')
            
            # Convert to numpy arrays
            src_np = np.asarray(src_img, np.float32) / 255.0
            trg_np = np.asarray(trg_img, np.float32) / 255.0
            
            # Transpose to CxHxW format for FDA
            src_np = src_np.transpose((2, 0, 1))
            trg_np = trg_np.transpose((2, 0, 1))
            
            # Apply FDA
            result_np = FDA_source_to_target_np(src_np, trg_np, L=beta)
            
            # Convert back to HxWxC and save
            result_np = result_np.transpose((1, 2, 0))
            result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)
            
            result_img = Image.fromarray(result_np)
            result_img.save(output_path)
            return True
        except Exception as e:
            print(f"FDA failed: {e}")
            return False
    
    return apply_fda_to_image

def load_histogram_clahe_lab():
    """Load CLAHE RGB functions."""
    import cv2 as cv
    import numpy as np
    
    def apply_clahe_rgb(input_path, output_path):
        """Apply CLAHE in RGB color space."""
        try:
            img_bgr = cv.imread(input_path, cv.IMREAD_COLOR)
            if img_bgr is None:
                return False
            
            # Convert to LAB
            img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
            L, a, b = cv.split(img_lab)
            
            # Apply CLAHE to L-channel
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            L_clahe = clahe.apply(L)
            
            # Reconstruct and save
            img_lab_clahe = cv.merge([L_clahe, a, b])
            img_bgr_clahe = cv.cvtColor(img_lab_clahe, cv.COLOR_LAB2BGR)
            cv.imwrite(output_path, img_bgr_clahe)
            return True
        except Exception as e:
            print(f"CLAHE RGB failed: {e}")
            return False
    
    return apply_clahe_rgb

def load_histogram_clahe():
    """Load CLAHE functions from clahe.py."""
    import cv2 as cv
    import numpy as np
    
    def apply_clahe_grayscale(input_path, output_path):
        """Apply CLAHE to color image using clahe.py method."""
        try:
            # Load color image (like clahe.py does)
            img_original = cv.imread(input_path, cv.IMREAD_COLOR)
            if img_original is None:
                return False
            
            # Convert to grayscale for processing (like clahe.py)
            img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                return False
            
            # Apply CLAHE with same parameters as clahe.py
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(img)
            
            # Convert back to 3-channel for consistency
            img_clahe_3ch = cv.cvtColor(img_clahe, cv.COLOR_GRAY2BGR)
            
            # Save result
            cv.imwrite(output_path, img_clahe_3ch)
            return True
        except Exception as e:
            print(f"CLAHE grayscale failed: {e}")
            return False
    
    return apply_clahe_grayscale

def load_pairs():
    """Load matching pairs from mixed_paired_images."""
    pairs = []
    cg_dir = 'mixed_paired_images/cg'
    real_dir = 'mixed_paired_images/real'
    
    if not os.path.exists(cg_dir) or not os.path.exists(real_dir):
        print("Error: mixed_paired_images directory not found!")
        print("Please run create_mixed_pair_folders.py first.")
        return []
    
    cg_files = sorted([f for f in os.listdir(cg_dir) if f.endswith('.png')])
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.png')])
    
    for cg_file, real_file in zip(cg_files, real_files):
        pairs.append({
            'cg_path': os.path.join(cg_dir, cg_file),
            'real_path': os.path.join(real_dir, real_file),
            'pair_id': cg_file.replace('.png', '')
        })
    
    return pairs

def run_experiment(exp_name, beta, histogram_method=None):
    """Run a single experiment."""
    print(f"\n=== Running Experiment: {exp_name} ===")
    
    # Create experiment directory
    exp_dir = f'experiments/{exp_name}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load processing functions
    apply_fda = load_fda_demo()
    
    if histogram_method == 'A':
        apply_hist = load_histogram_clahe_lab()
    elif histogram_method == 'B':
        apply_hist = load_histogram_clahe()
    else:
        apply_hist = None
    
    # Load pairs
    pairs = load_pairs()
    if not pairs:
        return False
    
    print(f"Processing {len(pairs)} pairs...")
    
    # Process each pair
    results = []
    for pair in pairs:
        pair_id = pair['pair_id']
        cg_path = pair['cg_path']
        real_path = pair['real_path']
        
        # Step 1: Apply FDA (CG source, Real target)
        fda_output = os.path.join(exp_dir, f'{pair_id}_fda.png')
        if not apply_fda(cg_path, real_path, beta, fda_output):
            print(f"  Failed FDA for {pair_id}")
            continue
        
        # Step 2: Apply histogram if specified
        if histogram_method:
            hist_output = os.path.join(exp_dir, f'{pair_id}_fda_hist.png')
            if not apply_hist(fda_output, hist_output):
                print(f"  Failed histogram for {pair_id}")
                continue
            final_output = hist_output
        else:
            final_output = fda_output
        
        results.append({
            'pair_id': pair_id,
            'cg_source': cg_path,
            'real_target': real_path,
            'fda_output': fda_output,
            'final_output': final_output,
            'beta': beta,
            'histogram_method': histogram_method or 'None'
        })
    
    # Save experiment metadata
    metadata = {
        'experiment_name': exp_name,
        'beta': beta,
        'histogram_method': histogram_method,
        'total_pairs': len(pairs),
        'successful_pairs': len(results),
        'results': results
    }
    
    with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Completed {exp_name}: {len(results)}/{len(pairs)} pairs successful")
    return True

def main():
    """Run all 9 experiments."""
    print("Starting 9 experiments...")
    
    # Define experiments
    experiments = [
        # FDA only
        ('01_fda_only_003', 0.003, None),
        ('02_fda_only_005', 0.005, None),
        ('03_fda_only_010', 0.01, None),
        
        # FDA + Histogram Candidate A (CLAHE RGB)
        ('04_fda_claheA_003', 0.003, 'A'),
        ('05_fda_claheA_005', 0.005, 'A'),
        ('06_fda_claheA_010', 0.01, 'A'),
        
        # FDA + Histogram Candidate B (CLAHE grayscale)
        ('07_fda_claheB_003', 0.003, 'B'),
        ('08_fda_claheB_005', 0.005, 'B'),
        ('09_fda_claheB_010', 0.01, 'B'),
    ]
    
    # Create experiments directory
    os.makedirs('experiments', exist_ok=True)
    
    # Run all experiments
    successful = 0
    for exp_name, beta, hist_method in experiments:
        if run_experiment(exp_name, beta, hist_method):
            successful += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Experiments completed: {successful}/9")
    print(f"Results saved in: experiments/")
    
    # Create summary file
    summary = {
        'total_experiments': 9,
        'successful_experiments': successful,
        'experiments': [
            {
                'name': exp[0],
                'beta': exp[1],
                'histogram_method': exp[2] or 'None',
                'folder': f'experiments/{exp[0]}'
            }
            for exp in experiments
        ]
    }
    
    with open('experiments/experiments_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Summary saved to: experiments/experiments_summary.json")

if __name__ == "__main__":
    main()
