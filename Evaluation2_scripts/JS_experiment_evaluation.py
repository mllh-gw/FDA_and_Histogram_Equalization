import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
from tqdm import tqdm
import json
import pandas as pd
import cv2
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ImageFolderDataset(Dataset):
    """Wrap image list as a PyTorch Dataset"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img


class JSEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # For loading images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_images_from_paths(self, image_paths):
        """Load images from paths and return as BGR numpy arrays"""
        images = []
        for path in tqdm(image_paths, desc="Loading images"):
            img = cv2.imread(path)
            if img is not None:
                # Resize to standard size for consistency
                img = cv2.resize(img, (224, 224))
                images.append(img)
        return images

    def compute_histogram(self, image_bgr, bins=255):
        """Return per-channel histogram as probability distribution (excluding zero=background)"""
        histograms = {}
        for ch, name in enumerate(['B', 'G', 'R']):
            hist = cv2.calcHist([image_bgr], [ch], None, [bins], [1, 256]).flatten()
            hist_prob = hist / (hist.sum() + 1e-10)
            histograms[name] = hist_prob
        return histograms

    def compute_histogram_distances(self, hist_a, hist_b):
        """Compute JS distance and EMD between two histograms"""
        distances = {}
        bins = np.arange(len(hist_a))
        distances['JS'] = float(jensenshannon(hist_a, hist_b))
        distances['EMD'] = float(wasserstein_distance(bins, bins, hist_a, hist_b))
        return distances

    def aggregate_histograms(self, images, bins=255):
        """Compute average histogram across multiple images"""
        all_hists = {'R': [], 'G': [], 'B': []}
        for img in images:
            h = self.compute_histogram(img, bins)
            for ch in ['R', 'G', 'B']:
                all_hists[ch].append(h[ch])

        avg_hists = {}
        for ch in ['R', 'G', 'B']:
            avg = np.mean(all_hists[ch], axis=0)
            avg_hists[ch] = avg / (avg.sum() + 1e-10)
        return avg_hists

    def load_experiment_data(self, experiments_dir='experiments'):
        """Load experiment data from existing outputs."""
        print("Loading experiment data...")
        
        # Load original CG and real images
        cg_paths = sorted(glob.glob(os.path.join('mixed_paired_images', 'cg', '*.png')))
        real_paths = sorted(glob.glob(os.path.join('mixed_paired_images', 'real', '*.png')))
        
        print(f"Found {len(cg_paths)} CG images and {len(real_paths)} Real images")
        
        # Load experiment results
        experiments = {}
        exp_dirs = sorted([d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))])
        
        for exp_dir in exp_dirs:
            exp_path = os.path.join(experiments_dir, exp_dir)
            metadata_path = os.path.join(exp_path, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Get final output paths (processed images)
                final_paths = []
                for result in metadata['results']:
                    if 'final_output' in result:
                        final_paths.append(result['final_output'])
                
                experiments[exp_dir] = {
                    'beta': metadata['beta'],
                    'histogram': metadata['histogram_method'],
                    'paths': sorted(final_paths),
                    'metadata': metadata
                }
                
                print(f"Loaded {exp_dir}: {len(final_paths)} images (β={metadata['beta']}, hist={metadata['histogram_method']})")
        
        return cg_paths, real_paths, experiments

    def evaluate_experiments(self, experiments_dir='experiments', output_root="js_results"):
        """Evaluate all experiments using Jensen-Shannon divergence."""
        os.makedirs(output_root, exist_ok=True)
        
        # Load data
        cg_paths, real_paths, experiments = self.load_experiment_data(experiments_dir)
        
        if not cg_paths or not real_paths:
            print("Error: No original images found!")
            return
        
        # Load images
        print("\nLoading real images...")
        real_imgs = self.load_images_from_paths(real_paths)
        
        print("Loading original CG images...")
        cg_imgs = self.load_images_from_paths(cg_paths)
        
        # Compute baseline histograms
        print("Computing baseline histograms...")
        hist_real = self.aggregate_histograms(real_imgs)
        hist_cg = self.aggregate_histograms(cg_imgs)
        
        # Compute baseline JS distances
        baseline_results = {}
        for ch in ['R', 'G', 'B']:
            d_baseline = self.compute_histogram_distances(hist_cg[ch], hist_real[ch])
            baseline_results[ch] = d_baseline
        
        print(f"\n[Baseline] JS Distances - R: {baseline_results['R']['JS']:.4f}, G: {baseline_results['G']['JS']:.4f}, B: {baseline_results['B']['JS']:.4f}")
        
        results = []
        
        # Evaluate each experiment
        for exp_name, exp_data in experiments.items():
            print(f"\n--- Evaluating: {exp_name} ---")
            print(f"Beta: {exp_data['beta']}, Histogram: {exp_data['histogram']}")
            
            # Load processed images
            processed_imgs = self.load_images_from_paths(exp_data['paths'])
            
            # Compute histograms
            hist_processed = self.aggregate_histograms(processed_imgs)
            
            # Compute JS distances
            exp_results = {}
            for ch in ['R', 'G', 'B']:
                d_after = self.compute_histogram_distances(hist_processed[ch], hist_real[ch])
                
                # Compute reduction percentage
                js_before = baseline_results[ch]['JS']
                js_after = d_after['JS']
                reduction = (1 - js_after / (js_before + 1e-10)) * 100
                
                exp_results[ch] = {
                    'JS_before': js_before,
                    'JS_after': js_after,
                    'JS_reduction_pct': reduction,
                    'EMD_before': baseline_results[ch]['EMD'],
                    'EMD_after': d_after['EMD'],
                    'EMD_reduction_pct': (1 - d_after['EMD'] / (baseline_results[ch]['EMD'] + 1e-10)) * 100,
                }
            
            # Compute average reductions
            avg_js_reduction = np.mean([exp_results[ch]['JS_reduction_pct'] for ch in ['R', 'G', 'B']])
            avg_emd_reduction = np.mean([exp_results[ch]['EMD_reduction_pct'] for ch in ['R', 'G', 'B']])
            
            print(f">>> {exp_name} | Avg JS Reduction: {avg_js_reduction:+.2f}% | Avg EMD Reduction: {avg_emd_reduction:+.2f}%")
            
            results.append({
                'experiment': exp_name,
                'beta': exp_data['beta'],
                'histogram': exp_data['histogram'],
                'avg_js_reduction': avg_js_reduction,
                'avg_emd_reduction': avg_emd_reduction,
                'channel_results': exp_results,
                'num_images': len(exp_data['paths'])
            })
        
        # Save results and create visualizations
        self.save_results(results, output_root)
        self.create_visualizations(results, hist_real, hist_cg, experiments, output_root)
        
        return results

    def save_results(self, results, output_root):
        """Save results to CSV and print summary."""
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by average JS reduction
        df_sorted = df.sort_values('avg_js_reduction', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_root, "js_experiment_results.csv")
        df_sorted.to_csv(csv_path, index=False)
        
        # Print summary
        print(f"\n{'='*100}")
        print("JENSEN-SHANNON DIVERGENCE EXPERIMENT EVALUATION SUMMARY")
        print(f"{'='*100}")
        print(f"{'Experiment':<20} {'Beta':<8} {'Histogram':<12} {'Avg JS Reduction':<15} {'Avg EMD Reduction':<15}")
        print("-" * 100)
        
        for _, row in df_sorted.iterrows():
            print(f"{row['experiment']:<20} {row['beta']:<8.3f} {str(row['histogram']):<12} "
                  f"{row['avg_js_reduction']:+15.2f}% {row['avg_emd_reduction']:+15.2f}%")
        
        print(f"\nBest performer: {df_sorted.iloc[0]['experiment']} with {df_sorted.iloc[0]['avg_js_reduction']:+.2f}% average JS reduction")
        print(f"Results saved to: {csv_path}")
        
        # Channel-wise summary
        print(f"\n{'='*100}")
        print("CHANNEL-WISE JENSEN-SHANNON DIVERGENCE SUMMARY")
        print(f"{'='*100}")
        
        for ch in ['R', 'G', 'B']:
            print(f"\n{ch} Channel Results:")
            channel_results = []
            for result in results:
                ch_result = result['channel_results'][ch]
                channel_results.append({
                    'experiment': result['experiment'],
                    'beta': result['beta'],
                    'histogram': result['histogram'],
                    'js_reduction': ch_result['JS_reduction_pct'],
                    'emd_reduction': ch_result['EMD_reduction_pct']
                })
            
            channel_df = pd.DataFrame(channel_results)
            channel_df_sorted = channel_df.sort_values('js_reduction', ascending=False)
            
            for _, row in channel_df_sorted.iterrows():
                print(f"  {row['experiment']:<20} | JS: {row['js_reduction']:+6.2f}% | EMD: {row['emd_reduction']:+6.2f}%")

    def create_visualizations(self, results, hist_real, hist_cg, experiments, output_root):
        """Create histogram comparisons for all experiments."""
        print("\nCreating histogram comparisons for all experiments...")
        
        # Sort results by JS reduction for consistent ordering
        results_sorted = sorted(results, key=lambda x: x['avg_js_reduction'], reverse=True)
        
        # Create histogram comparison for each experiment
        for i, result in enumerate(results_sorted):
            exp_name = result['experiment']
            print(f"Creating histogram comparison for {exp_name} ({i+1}/{len(results_sorted)})")
            
            # Load experiment images
            exp_imgs = self.load_images_from_paths(experiments[exp_name]['paths'])
            hist_exp = self.aggregate_histograms(exp_imgs)
            
            # Create histogram comparison following verification pipeline style
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            colors_map = {'R': ('red', '#ff6666', '#ff0000'),
                          'G': ('green', '#66cc66', '#00aa00'),
                          'B': ('blue', '#6666ff', '#0000ff')}
            
            for idx, ch in enumerate(['R', 'G', 'B']):
                ax = axes[idx]
                x = np.arange(1, 256)
                
                ax.fill_between(x, hist_real[ch], alpha=0.3, color='gray', label='Real')
                ax.plot(x, hist_real[ch], color='gray', linewidth=1.5)
                ax.plot(x, hist_cg[ch], color=colors_map[ch][1], linewidth=1.2,
                        linestyle='--', alpha=0.7, label='CG (Original)')
                ax.plot(x, hist_exp[ch], color=colors_map[ch][2], linewidth=1.5,
                        label=f'CG ({exp_name})')
                
                ch_result = result['channel_results'][ch]
                ax.set_title(f'{ch} Channel\n'
                            f'JS: {ch_result["JS_before"]:.3f} -> {ch_result["JS_after"]:.3f} '
                            f'({ch_result["JS_reduction_pct"]:+.1f}%)\n'
                            f'EMD: {ch_result["EMD_before"]:.1f} -> {ch_result["EMD_after"]:.1f} '
                            f'({ch_result["EMD_reduction_pct"]:+.1f}%)',
                            fontsize=11)
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Probability')
                ax.legend(fontsize=9)
                ax.set_xlim(1, 255)
            
            # Format experiment name for title
            exp_title = exp_name.replace('_fda_', ' + FDA ').replace('_', ' ').title()
            beta_str = f"β={result['beta']:.3f}"
            hist_str = f"Hist={result['histogram']}" if result['histogram'] else "Hist=None"
            
            fig.suptitle(f'Phase 1: Histogram Comparison -- {exp_title}\n'
                        f'{beta_str}, {hist_str} | Avg JS Reduction: {result["avg_js_reduction"]:+.2f}%',
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save with experiment name
            filename = f'histogram_comparison_{exp_name}.png'
            plt.savefig(os.path.join(output_root, filename), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create a summary grid showing all experiments
        self.create_experiment_grid(results_sorted, hist_real, hist_cg, experiments, output_root)
        
        # Create comprehensive metrics report
        self.create_comprehensive_report(results, hist_real, hist_cg, experiments, output_root)
        
        print(f"Histogram comparisons saved to: {output_root}/")

    def create_experiment_grid(self, results, hist_real, hist_cg, experiments, output_root):
        """Create a grid showing all experiments' histogram comparisons."""
        print("Creating experiment overview grid...")
        
        # Create a 3x3 grid for all 9 experiments
        fig, axes = plt.subplots(3, 3, figsize=(27, 27))
        fig.suptitle('All Experiments: Histogram Overview (G Channel Only)\n'
                    'Showing G channel JS divergence for all experiments',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Sort results by JS reduction
        results_sorted = sorted(results, key=lambda x: x['avg_js_reduction'], reverse=True)
        
        for i, result in enumerate(results_sorted):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            exp_name = result['experiment']
            
            # Load experiment images
            exp_imgs = self.load_images_from_paths(experiments[exp_name]['paths'])
            hist_exp = self.aggregate_histograms(exp_imgs)
            
            # Plot G channel histogram (middle channel for visibility)
            ch = 'G'
            x = np.arange(1, 256)
            
            ax.fill_between(x, hist_real[ch], alpha=0.3, color='gray', label='Real')
            ax.plot(x, hist_real[ch], color='gray', linewidth=1.5)
            ax.plot(x, hist_cg[ch], color='#66cc66', linewidth=1.2,
                    linestyle='--', alpha=0.7, label='CG (Orig)')
            ax.plot(x, hist_exp[ch], color='#00aa00', linewidth=1.5,
                    label='CG (FDA)')
            
            ch_result = result['channel_results'][ch]
            
            # Format experiment name
            exp_short = exp_name.replace('_fda_', '\n').replace('_', ' ')
            
            ax.set_title(f'{exp_short}\n'
                        f'JS: {ch_result["JS_before"]:.3f} -> {ch_result["JS_after"]:.3f} '
                        f'({ch_result["JS_reduction_pct"]:+.1f}%)',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Pixel Intensity', fontsize=9)
            ax.set_ylabel('Probability', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_xlim(1, 255)
            ax.grid(True, alpha=0.3)
            
            # Add ranking
            ax.text(0.02, 0.98, f'#{i+1}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color='red',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_root, 'all_experiments_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def create_comprehensive_report(self, results, hist_real, hist_cg, experiments, output_root):
        """Create comprehensive report following verification pipeline structure."""
        print("Creating comprehensive metrics report...")
        
        # Generate detailed metrics
        all_metrics = {
            'js_divergence': {},
            'best_experiment': {},
            'experiment_groups': {},
            'summary': {}
        }
        
        # Best experiment details
        best_exp = max(results, key=lambda x: x['avg_js_reduction'])
        all_metrics['best_experiment'] = {
            'name': best_exp['experiment'],
            'beta': best_exp['beta'],
            'histogram_method': best_exp['histogram'],
            'avg_js_reduction': best_exp['avg_js_reduction'],
            'avg_emd_reduction': best_exp['avg_emd_reduction'],
            'channel_results': best_exp['channel_results']
        }
        
        # Group statistics
        fda_only = [r for r in results if r['histogram'] is None]
        clahe_rgb = [r for r in results if r['histogram'] == 'A']
        clahe_gray = [r for r in results if r['histogram'] == 'B']
        
        all_metrics['experiment_groups'] = {
            'fda_only': {
                'count': len(fda_only),
                'avg_js_reduction': np.mean([r['avg_js_reduction'] for r in fda_only]) if fda_only else 0,
                'avg_emd_reduction': np.mean([r['avg_emd_reduction'] for r in fda_only]) if fda_only else 0
            },
            'clahe_rgb': {
                'count': len(clahe_rgb),
                'avg_js_reduction': np.mean([r['avg_js_reduction'] for r in clahe_rgb]) if clahe_rgb else 0,
                'avg_emd_reduction': np.mean([r['avg_emd_reduction'] for r in clahe_rgb]) if clahe_rgb else 0
            },
            'clahe_gray': {
                'count': len(clahe_gray),
                'avg_js_reduction': np.mean([r['avg_js_reduction'] for r in clahe_gray]) if clahe_gray else 0,
                'avg_emd_reduction': np.mean([r['avg_emd_reduction'] for r in clahe_gray]) if clahe_gray else 0
            }
        }
        
        # Overall summary
        all_metrics['summary'] = {
            'total_experiments': len(results),
            'avg_js_reduction': np.mean([r['avg_js_reduction'] for r in results]),
            'avg_emd_reduction': np.mean([r['avg_emd_reduction'] for r in results]),
            'positive_improvements': sum(1 for r in results if r['avg_js_reduction'] > 0),
            'best_js_reduction': max(r['avg_js_reduction'] for r in results),
            'worst_js_reduction': min(r['avg_js_reduction'] for r in results)
        }
        
        # Save JSON report
        with open(os.path.join(output_root, 'js_metrics_report.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        # Save CSV report
        flat_data = []
        for r in results:
            row = {
                'experiment': r['experiment'],
                'beta': r['beta'],
                'histogram_method': r['histogram'],
                'avg_js_reduction': r['avg_js_reduction'],
                'avg_emd_reduction': r['avg_emd_reduction'],
                'num_images': r['num_images']
            }
            # Add channel-wise results
            for ch in ['R', 'G', 'B']:
                ch_result = r['channel_results'][ch]
                row[f'js_reduction_{ch}'] = ch_result['JS_reduction_pct']
                row[f'emd_reduction_{ch}'] = ch_result['EMD_reduction_pct']
                row[f'js_before_{ch}'] = ch_result['JS_before']
                row[f'js_after_{ch}'] = ch_result['JS_after']
            
            flat_data.append(row)
        
        df = pd.DataFrame(flat_data)
        df = df.sort_values('avg_js_reduction', ascending=False)
        df.to_csv(os.path.join(output_root, 'js_metrics_report.csv'), index=False)
        
        print(f"Comprehensive report saved to: {output_root}/js_metrics_report.json and .csv")


def main():
    """Main function for JS divergence evaluation"""
    # Check if required directories exist
    if not os.path.exists('experiments'):
        print("Error: 'experiments' directory not found!")
        print("Please run run_experiments.py first to generate experiment results.")
        return
    
    if not os.path.exists('mixed_paired_images'):
        print("Error: 'mixed_paired_images' directory not found!")
        print("Please run create_mixed_pair_folders.py first to generate paired images.")
        return
    
    # Initialize evaluator
    evaluator = JSEvaluator()
    
    # Run evaluation
    print("\n=== Running Jensen-Shannon Divergence Evaluation for All Experiments ===")
    results = evaluator.evaluate_experiments(
        experiments_dir='experiments',
        output_root="js_experiment_results"
    )
    
    if results:
        print(f"\n=== JS Divergence Evaluation Complete ===")
        print(f"Evaluated {len(results)} experiments")
        
        # Find best experiment
        best_exp = max(results, key=lambda x: x['avg_js_reduction'])
        print(f"\nBest experiment: {best_exp['experiment']} with {best_exp['avg_js_reduction']:+.2f}% average JS reduction")
        
        # Count improvements
        improvements = sum(1 for r in results if r['avg_js_reduction'] > 0)
        print(f"Experiments with positive JS reduction: {improvements}/{len(results)}")
    else:
        print("No experiments were evaluated.")


if __name__ == "__main__":
    main()
