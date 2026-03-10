import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import warnings
import json

# Try to import optional dependencies
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. t-SNE will be skipped.")
    TSNE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")
    PLOTTING_AVAILABLE = False

warnings.filterwarnings('ignore')

class ResNet101FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet101FeatureExtractor, self).__init__()
        print("Loading Pre-trained ResNet101 as feature extractor...")
        
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.backbone(x)
        return torch.flatten(features, 1)

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

class ExperimentEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_extractor = ResNet101FeatureExtractor().to(device)
        self.feature_extractor.eval()

        # Standard ImageNet normalization for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_paths, batch_size=32):
        """Extract features from image paths."""
        if not image_paths:
            return np.array([]), []
            
        dataset = ImageDataset(image_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        features = []
        paths = []

        with torch.no_grad():
            for images, image_paths_batch in tqdm(dataloader, desc="Extracting Features"):
                images = images.to(self.device)
                batch_features = self.feature_extractor(images)
                features.append(batch_features.cpu().numpy())
                paths.extend(image_paths_batch)

        return np.vstack(features), paths

    def load_experiment_data(self, experiments_dir):
        """Load all experiment data and original images."""
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

    def compute_tsne_visualization(self, feat_cg, feat_exp, feat_real, exp_name, beta, histogram, out_dir):
        """Compute t-SNE visualization for one experiment."""
        if not TSNE_AVAILABLE or not PLOTTING_AVAILABLE:
            print(f"Skipping t-SNE for {exp_name} (missing dependencies)")
            return 0.0, 0.0, 0.0
            
        print(f"Computing t-SNE for {exp_name}...")
        
        # Combine features
        all_feats = np.vstack([feat_cg, feat_exp, feat_real])
        labels = (['Original CG'] * len(feat_cg) +
                  [f'{exp_name}'] * len(feat_exp) +
                  ['Real'] * len(feat_real))

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats)//3))
        res = tsne.fit_transform(all_feats)

        df = pd.DataFrame({'x': res[:, 0], 'y': res[:, 1], 'Domain': labels})

        # Calculate centroids and distances
        centroids = df.groupby('Domain')[['x', 'y']].mean()
        
        # Calculate domain gap reduction
        dist_orig = np.linalg.norm(centroids.loc['Original CG'] - centroids.loc['Real'])
        dist_exp = np.linalg.norm(centroids.loc[exp_name] - centroids.loc['Real'])
        reduction = ((dist_orig - dist_exp) / dist_orig) * 100 if dist_orig > 0 else 0

        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='Domain',
                        palette={'Original CG': 'red', exp_name: 'green', 'Real': 'blue'},
                        s=100, alpha=0.8, edgecolor='black')

        # Draw lines to show domain gaps
        plt.plot([centroids.loc['Original CG', 'x'], centroids.loc['Real', 'x']],
                 [centroids.loc['Original CG', 'y'], centroids.loc['Real', 'y']], 
                 'r--', alpha=0.5, linewidth=2, label='Original Gap')
        plt.plot([centroids.loc[exp_name, 'x'], centroids.loc['Real', 'x']],
                 [centroids.loc[exp_name, 'y'], centroids.loc['Real', 'y']], 
                 'g--', alpha=0.5, linewidth=2, label='Experiment Gap')

        # Add experiment details to title
        hist_str = f"+ {histogram}" if histogram else "Only"
        title = f't-SNE: {exp_name}\nOriginal CG vs Real vs {exp_name}\n'
        title += f'Beta={beta}, Histogram={hist_str}\n'
        title += f'Domain Gap Reduction: {reduction:.2f}%'
        
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        save_path = os.path.join(out_dir, f"tsne_{exp_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_path}")

        return reduction, dist_orig, dist_exp

    def evaluate_all_experiments(self, experiments_dir='experiments'):
        """Evaluate all 9 experiments with t-SNE."""
        out_dir = "tsne_evaluation_results"
        os.makedirs(out_dir, exist_ok=True)
        
        # Load data
        cg_paths, real_paths, experiments = self.load_experiment_data(experiments_dir)
        
        if not cg_paths or not real_paths:
            print("Error: No original images found!")
            return
        
        # Extract baseline features
        print("\nExtracting baseline features...")
        feat_cg, _ = self.extract_features(cg_paths)
        feat_real, _ = self.extract_features(real_paths)
        
        # Store results
        results = []
        
        # Evaluate each experiment
        for exp_name, exp_data in experiments.items():
            print(f"\n{'='*50}")
            print(f"Evaluating: {exp_name}")
            print(f"Beta: {exp_data['beta']}, Histogram: {exp_data['histogram']}")
            
            # Extract features for experiment results
            feat_exp, _ = self.extract_features(exp_data['paths'])
            
            if len(feat_exp) == 0:
                print(f"No features extracted for {exp_name}, skipping...")
                continue
            
            # Compute t-SNE visualization
            reduction, dist_orig, dist_exp = self.compute_tsne_visualization(
                feat_cg, feat_exp, feat_real, 
                exp_name, exp_data['beta'], exp_data['histogram'], 
                out_dir
            )
            
            # Store results
            results.append({
                'experiment': exp_name,
                'beta': exp_data['beta'],
                'histogram': exp_data['histogram'],
                'original_gap': dist_orig,
                'experiment_gap': dist_exp,
                'reduction_percent': reduction,
                'num_images': len(exp_data['paths'])
            })
            
            print(f"Domain gap reduction: {reduction:.2f}%")
        
        # Create summary
        self.create_summary(results, out_dir)
        
        print(f"\n{'='*50}")
        print("Evaluation complete!")
        print(f"Results saved to: {out_dir}")
        
        return results

    def create_summary(self, results, out_dir):
        """Create summary plot and table."""
        if not results:
            print("No results to summarize!")
            return
        
        # Create summary table
        df = pd.DataFrame(results)
        df = df.sort_values('reduction_percent', ascending=False)
        
        # Save summary table
        table_path = os.path.join(out_dir, "experiment_summary.csv")
        df.to_csv(table_path, index=False)
        
        # Create summary plot if plotting is available
        if PLOTTING_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Domain gap reduction
            colors = ['green' if r > 0 else 'red' for r in df['reduction_percent']]
            bars = ax1.bar(range(len(df)), df['reduction_percent'], color=colors, alpha=0.7)
            ax1.set_xlabel('Experiment')
            ax1.set_ylabel('Domain Gap Reduction (%)')
            ax1.set_title('Domain Gap Reduction by Experiment')
            ax1.set_xticks(range(len(df)))
            ax1.set_xticklabels(df['experiment'], rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, df['reduction_percent'])):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}%', ha='center', va='bottom')
            
            # Plot 2: Distance comparison
            x = np.arange(len(df))
            width = 0.35
            ax2.bar(x - width/2, df['original_gap'], width, label='Original Gap', alpha=0.7, color='red')
            ax2.bar(x + width/2, df['experiment_gap'], width, label='Experiment Gap', alpha=0.7, color='green')
            ax2.set_xlabel('Experiment')
            ax2.set_ylabel('Distance')
            ax2.set_title('Domain Gaps: Original vs Experiment')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df['experiment'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            summary_path = os.path.join(out_dir, "experiments_summary.png")
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Summary plot saved to: {summary_path}")
        else:
            print("Skipping summary plots (matplotlib not available)")
        
        # Print summary
        print(f"\n{'='*50}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*50}")
        print(f"{'Experiment':<20} {'Beta':<8} {'Histogram':<12} {'Reduction':<12} {'Gap Orig':<10} {'Gap Exp':<10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['experiment']:<20} {row['beta']:<8.3f} {str(row['histogram']):<12} "
                  f"{row['reduction_percent']:<12.2f} {row['original_gap']:<10.2f} {row['experiment_gap']:<10.2f}")
        
        print(f"\nTable saved to: {table_path}")

def main():
    """Main function to evaluate all experiments."""
    evaluator = ExperimentEvaluator()
    
    # Check if experiments directory exists
    if not os.path.exists('experiments'):
        print("Error: 'experiments' directory not found!")
        print("Please run run_experiments.py first to generate experiment results.")
        return
    
    if not os.path.exists('mixed_paired_images'):
        print("Error: 'mixed_paired_images' directory not found!")
        print("Please run create_mixed_pair_folders.py first to generate paired images.")
        return
    
    # Evaluate all experiments
    results = evaluator.evaluate_all_experiments()
    
    if results:
        print(f"\nEvaluation completed for {len(results)} experiments!")
    else:
        print("No experiments were evaluated.")

if __name__ == "__main__":
    main()
