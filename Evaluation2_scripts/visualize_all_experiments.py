#!/usr/bin/env python3
"""
Visualization script to stack original CG, real, and all 9 experiment outputs.
"""

import os
import json
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_image(path):
    """Load image in RGB format."""
    if not os.path.exists(path):
        return None
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        return None
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def load_experiment_metadata():
    """Load all experiment metadata."""
    experiments = []
    exp_dirs = sorted([d for d in os.listdir('experiments') if os.path.isdir(f'experiments/{d}')])
    
    for exp_dir in exp_dirs:
        metadata_path = f'experiments/{exp_dir}/metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                experiments.append({
                    'name': exp_dir,
                    'beta': metadata['beta'],
                    'histogram': metadata['histogram_method'],
                    'results': metadata['results']
                })
    
    return experiments

def add_text_label(img, text, position=(10, 10), font_size=20, color=(255, 255, 255)):
    """Add text label to image."""
    try:
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Add text background for better visibility
        if font:
            bbox = draw.textbbox(position, text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle
            draw.rectangle([position, (position[0] + text_width + 4, position[1] + text_height + 4)], 
                        fill=(0, 0, 0))
            # Draw text
            draw.text(position, text, fill=color, font=font)
        else:
            # Fallback: simple text without font
            draw.text(position, text, fill=color)
        
        # Convert back to numpy array
        return np.array(pil_img)
    except Exception as e:
        print(f"Warning: Could not add text label: {e}")
        return img

def create_comparison_grid(pair_id, cg_path, real_path, experiments):
    """Create comparison grid for one pair (2 rows × 6 columns)."""
    # Load original images
    cg_img = load_image(cg_path)
    real_img = load_image(real_path)
    
    if cg_img is None or real_img is None:
        return None
    
    # Resize to same dimensions (use smaller size)
    h1, w1 = cg_img.shape[:2]
    h2, w2 = real_img.shape[:2]
    target_h, target_w = min(h1, h2), min(w1, w2)
    
    cg_img = cv.resize(cg_img, (target_w, target_h))
    real_img = cv.resize(real_img, (target_w, target_h))
    
    # Add labels to original images
    cg_img = add_text_label(cg_img, "CG Original", position=(5, 5), font_size=16)
    real_img = add_text_label(real_img, "Real Original", position=(5, 5), font_size=16)
    
    # Collect experiment results
    exp_results = []
    for exp in experiments:
        for result in exp['results']:
            if result['pair_id'] == pair_id:
                final_img = load_image(result['final_output'])
                if final_img is not None:
                    # Resize to match target dimensions
                    final_img = cv.resize(final_img, (target_w, target_h))
                    
                    # Create label based on experiment type
                    if exp['histogram'] is None:
                        label = f"FDA β={exp['beta']}"
                    elif exp['histogram'] == 'A':
                        label = f"FDA+CLAHE-LAB β={exp['beta']}"
                    elif exp['histogram'] == 'B':
                        label = f"FDA+CLAHE-RGB β={exp['beta']}"
                    else:
                        label = exp['name']
                    
                    final_img = add_text_label(final_img, label, position=(5, 5), font_size=12)
                    exp_results.append((exp['name'], final_img))
                break
    
    # Sort experiments by name
    exp_results.sort(key=lambda x: x[0])
    
    # Create 2 rows × 6 columns layout
    # Row 1: CG Original + 5 experiments
    row1_imgs = [cg_img] + [img for name, img in exp_results[:5]]
    while len(row1_imgs) < 6:
        row1_imgs.append(np.zeros_like(cg_img))
    
    # Row 2: Real Original + remaining experiments
    row2_imgs = [real_img] + [img for name, img in exp_results[5:9]]
    while len(row2_imgs) < 6:
        row2_imgs.append(np.zeros_like(real_img))
    
    # Create rows with consistent width
    def create_row(imgs):
        # Ensure all images have same height
        max_h = max(img.shape[0] for img in imgs)
        resized_imgs = []
        for img in imgs:
            if img.shape[0] < max_h:
                # Pad height
                padding = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
                img = np.vstack([img, padding])
            resized_imgs.append(img)
        
        # Horizontal stack
        row = np.hstack(resized_imgs)
        return row
    
    row1 = create_row(row1_imgs)
    row2 = create_row(row2_imgs)
    
    # Ensure both rows have same width
    max_width = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] < max_width:
        padding = np.zeros((row1.shape[0], max_width - row1.shape[1], 3), dtype=row1.dtype)
        row1 = np.hstack([row1, padding])
    if row2.shape[1] < max_width:
        padding = np.zeros((row2.shape[0], max_width - row2.shape[1], 3), dtype=row2.dtype)
        row2 = np.hstack([row2, padding])
    
    # Combine rows
    full_grid = np.vstack([row1, row2])
    
    return full_grid

def create_visualization():
    """Create visualization for all pairs."""
    print("Loading experiment data...")
    experiments = load_experiment_metadata()
    
    # Load pairs from mixed_paired_images
    pairs = []
    cg_dir = 'mixed_paired_images/cg'
    real_dir = 'mixed_paired_images/real'
    
    cg_files = sorted([f for f in os.listdir(cg_dir) if f.endswith('.png')])
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.png')])
    
    for cg_file, real_file in zip(cg_files, real_files):
        pair_id = cg_file.replace('.png', '')
        pairs.append({
            'id': pair_id,
            'cg_path': os.path.join(cg_dir, cg_file),
            'real_path': os.path.join(real_dir, real_file)
        })
    
    print(f"Found {len(pairs)} pairs and {len(experiments)} experiments")
    
    # Create output directory
    output_dir = 'experiment_comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each pair
    successful = 0
    for i, pair in enumerate(pairs):
        print(f"Processing pair {i+1}/{len(pairs)}: {pair['id']}")
        
        grid = create_comparison_grid(pair['id'], pair['cg_path'], pair['real_path'], experiments)
        
        if grid is not None:
            # Save comparison
            output_path = os.path.join(output_dir, f"{pair['id']}_comparison.png")
            
            # Resize if too large
            h, w = grid.shape[:2]
            max_size = 2000
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                grid = cv.resize(grid, (new_w, new_h))
            
            cv.imwrite(output_path, cv.cvtColor(grid, cv.COLOR_RGB2BGR))
            successful += 1
        else:
            print(f"  Failed to create grid for {pair['id']}")
    
    print(f"\nCompleted {successful}/{len(pairs)} comparisons")
    print(f"Results saved to: {output_dir}")
    
    # Create summary image with experiment labels
    create_experiment_summary(experiments, output_dir)

def create_experiment_summary(experiments, output_dir):
    """Create summary showing experiment configurations."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    exp_names = [exp['name'] for exp in experiments]
    betas = [exp['beta'] for exp in experiments]
    histograms = [exp['histogram'] or 'None' for exp in experiments]
    
    # Create table data
    table_data = []
    for i, (name, beta, hist) in enumerate(zip(exp_names, betas, histograms)):
        table_data.append([i+1, name, f"β={beta}", hist])
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                  colLabels=['#', 'Experiment', 'Beta', 'Histogram'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[i, j]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f3f4')
    
    plt.title('Experiment Configurations', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'experiment_summary.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Experiment summary saved")

def main():
    """Main function."""
    print("Creating experiment visualizations...")
    
    # Check required directories
    if not os.path.exists('mixed_paired_images'):
        print("Error: mixed_paired_images directory not found!")
        print("Please run create_mixed_pair_folders.py first.")
        return
    
    if not os.path.exists('experiments'):
        print("Error: experiments directory not found!")
        print("Please run run_experiments.py first.")
        return
    
    create_visualization()
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
