#!/usr/bin/env python3
"""
Simple script to create paired image folders regardless of OK/NG category.
Pairs CG and real images with matching row/column positions.
"""

import os
import random
import shutil

def parse_filename(filename):
    """Extract row and column from filename."""
    parts = filename.replace('.png', '').split('_')
    row, col = None, None
    
    for i, part in enumerate(parts):
        if part.startswith('r') and len(part) > 1 and part[1:].isdigit():
            row = int(part[1:])
        elif part.startswith('c') and len(part) > 1 and part[1:].isdigit():
            col = int(part[1:])
    
    return row, col

def create_mixed_pair_folders(base_path='sample_dataset', target_ng=25, target_ok=20, seed=42):
    """Create paired image folders with target NG/OK counts from real images."""
    random.seed(seed)
    
    # Create output directories
    output_dir = 'mixed_paired_images'
    for img_type in ['cg', 'real']:
        os.makedirs(os.path.join(output_dir, img_type), exist_ok=True)
    
    # Collect all images by type and category
    all_images = {'cg': {'NG': [], 'OK': []}, 'real': {'NG': [], 'OK': []}}
    
    # Scan directories
    for img_type in ['cg', 'real']:
        for category in ['NG', 'OK']:
            dir_path = os.path.join(base_path, img_type, category)
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.endswith('.png'):
                        row, col = parse_filename(filename)
                        if row is not None and col is not None:
                            all_images[img_type][category].append({
                                'path': os.path.join(dir_path, filename),
                                'filename': filename,
                                'row': row,
                                'col': col,
                                'category': category
                            })
    
    print(f"Found images:")
    print(f"  CG NG: {len(all_images['cg']['NG'])}")
    print(f"  CG OK: {len(all_images['cg']['OK'])}")
    print(f"  Real NG: {len(all_images['real']['NG'])}")
    print(f"  Real OK: {len(all_images['real']['OK'])}")
    
    # Find matches by category to maintain target counts
    selected_pairs = []
    
    for category, target_count in [('NG', target_ng), ('OK', target_ok)]:
        # Create lookup for real images of this category
        real_lookup = {(img['row'], img['col']): img for img in all_images['real'][category]}
        
        # Find matches for this category
        matches = []
        for cg_img in all_images['cg']['NG'] + all_images['cg']['OK']:  # CG can be any category
            key = (cg_img['row'], cg_img['col'])
            if key in real_lookup:
                real_img = real_lookup[key]
                matches.append((cg_img, real_img, category))
        
        print(f"\n{category} category: {len(matches)} matching pairs found")
        
        # Select target number of pairs
        if len(matches) >= target_count:
            category_selected = random.sample(matches, target_count)
        else:
            category_selected = matches
            print(f"  Warning: Only {len(matches)} pairs available (target: {target_count})")
        
        selected_pairs.extend(category_selected)
        print(f"  Selected {len(category_selected)} pairs")
    
    print(f"\nTotal pairs selected: {len(selected_pairs)}")
    
    # Copy pairs to output folders
    for i, (cg_img, real_img, category) in enumerate(selected_pairs):
        # Copy CG image
        cg_dst = os.path.join(output_dir, 'cg', f"pair_{i:02d}_{cg_img['filename']}")
        shutil.copy2(cg_img['path'], cg_dst)
        
        # Copy Real image
        real_dst = os.path.join(output_dir, 'real', f"pair_{i:02d}_{real_img['filename']}")
        shutil.copy2(real_img['path'], real_dst)
    
    print(f"Copied {len(selected_pairs)} pairs to {output_dir}")
    
    # Show category breakdown
    ng_pairs = sum(1 for cg, real, cat in selected_pairs if cat == 'NG')
    ok_pairs = sum(1 for cg, real, cat in selected_pairs if cat == 'OK')
    
    print(f"\nCategory breakdown:")
    print(f"  NG pairs: {ng_pairs}")
    print(f"  OK pairs: {ok_pairs}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total pairs copied: {len(selected_pairs)}")
    print(f"Output directory: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"    ├── cg/ (pair_XX_*.png)")
    print(f"    └── real/ (pair_XX_*.png)")
    
    return output_dir

if __name__ == "__main__":
    print("Creating mixed paired image folders...")
    create_mixed_pair_folders()
