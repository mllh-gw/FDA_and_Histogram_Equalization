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

def compute_mmd_rbf(X, Y, gamma=None):
    """Compute unbiased MMD^2 estimate with RBF kernel"""
    from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

    if gamma is None:
        XY = np.vstack([X, Y])
        dists = euclidean_distances(XY, XY)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-10)

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    n = X.shape[0]
    m = Y.shape[0]

    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd2 = (K_XX.sum() / (n * (n - 1) + 1e-10) +
            K_YY.sum() / (m * (m - 1) + 1e-10) -
            2 * K_XY.mean())

    return float(max(mmd2, 0))

def compute_mmd_with_permutation_test(X, Y, n_permutations=1000):
    """Compute MMD + permutation test for p-value"""
    observed_mmd = compute_mmd_rbf(X, Y)

    XY = np.vstack([X, Y])
    n = X.shape[0]

    null_mmds = []
    for _ in range(n_permutations):
        perm = np.random.permutation(len(XY))
        X_perm = XY[perm[:n]]
        Y_perm = XY[perm[n:]]
        null_mmds.append(compute_mmd_rbf(X_perm, Y_perm))

    p_value = np.mean(np.array(null_mmds) >= observed_mmd)

    return observed_mmd, p_value

def compute_kernel(x, y, sigmas):
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2x*y
    dist = torch.pow(x, 2).sum(1).view(-1, 1) + \
           torch.pow(y, 2).sum(1).view(1, -1) - \
           2.0 * torch.mm(x, y.t())
    dist = torch.clamp(dist, min=0.0)
    
    kernel_val = 0
    for s in sigmas:
        gamma = 1.0 / (2.0 * s**2)
        kernel_val += torch.exp(-gamma * dist)
    return kernel_val / len(sigmas)

def mmd_square(x, y, sigmas=[1.0, 2.0, 5.0, 10.0, 20.0]):
    n = x.size(0)
    m = y.size(0)
    
    x_kernel = compute_kernel(x, x, sigmas)
    y_kernel = compute_kernel(y, y, sigmas)
    xy_kernel = compute_kernel(x, y, sigmas)

    # Unbiased estimator
    k_xx = (x_kernel.sum() - x_kernel.diag().sum()) / (n * (n - 1))
    k_yy = (y_kernel.sum() - y_kernel.diag().sum()) / (m * (m - 1))
    k_xy = xy_kernel.mean()

    return (k_xx + k_yy - 2 * k_xy).item()

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

class ExperimentEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, image_paths, batch_size=32):
        dataset = ImageDataset(image_paths, self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        all_features = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting Features"):
                feat = self.model(batch.to(self.device))
                feat = torch.flatten(feat, 1)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                all_features.append(feat.cpu())
        
        return torch.cat(all_features, dim=0)

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

    def evaluate_experiments(self, experiments_dir='experiments', batch_size=32, output_root="mmd_results", n_permutations=1000):
        """Evaluate all experiments using MMD with permutation testing."""
        os.makedirs(output_root, exist_ok=True)
        
        # Load data
        cg_paths, real_paths, experiments = self.load_experiment_data(experiments_dir)
        
        if not cg_paths or not real_paths:
            print("Error: No original images found!")
            return
        
        # Get target features (real images)
        print("\nExtracting target (real) features...")
        feat_trg = self.get_features(real_paths, batch_size)
        
        # Get source features (original CG)
        print("Extracting source (original CG) features...")
        feat_src_orig = self.get_features(cg_paths, batch_size)
        
        # Convert to numpy for sklearn functions
        feat_trg_np = feat_trg.cpu().numpy()
        feat_src_orig_np = feat_src_orig.cpu().numpy()
        
        # Compute baseline MMD with permutation test
        print("\nComputing baseline MMD (with permutation test)...")
        mmd_baseline, p_baseline = compute_mmd_with_permutation_test(feat_src_orig_np, feat_trg_np, n_permutations)
        
        print(f"\n[Baseline] MMD: {mmd_baseline:.6f} (p={p_baseline:.4f})")
        
        results = []
        
        # Evaluate each experiment
        for exp_name, exp_data in experiments.items():
            print(f"\n--- Evaluating: {exp_name} ---")
            print(f"Beta: {exp_data['beta']}, Histogram: {exp_data['histogram']}")
            
            # Get features for processed images
            feat_processed = self.get_features(exp_data['paths'], batch_size)
            feat_processed_np = feat_processed.cpu().numpy()
            
            # Compute MMD with permutation test
            print(f"Computing MMD for {exp_name} (with permutation test)...")
            mmd_after, p_after = compute_mmd_with_permutation_test(feat_processed_np, feat_trg_np, n_permutations)
            reduction = (1 - mmd_after / (mmd_baseline + 1e-10)) * 100
            
            print(f">>> {exp_name} | MMD: {mmd_after:.6f} (p={p_after:.4f}) | reduction: {reduction:+.2f}%")
            
            results.append({
                'experiment': exp_name,
                'beta': exp_data['beta'],
                'histogram': exp_data['histogram'],
                'mmd_baseline': mmd_baseline,
                'mmd_after': mmd_after,
                'reduction_percent': reduction,
                'p_baseline': p_baseline,
                'p_after': p_after,
                'num_images': len(exp_data['paths'])
            })
        
        # Save results
        self.save_results(results, output_root)
        
        return results

    def save_results(self, results, output_root):
        """Save results to CSV with simple separate rows format."""
        # Create simple rows format
        simple_results = []
        for result in results:
            simple_results.append({
                'experiment': result['experiment'],
                'beta': result['beta'],
                'histogram': result['histogram'],
                'mmd_baseline': result['mmd_baseline'],
                'mmd_after': result['mmd_after'],
                'reduction_percent': result['reduction_percent'],
                'p_baseline': result['p_baseline'],
                'p_after': result['p_after'],
                'num_images': result['num_images']
            })
        
        # Create DataFrame
        df = pd.DataFrame(simple_results)
        
        # Sort by reduction
        df_sorted = df.sort_values('reduction_percent', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_root, "mmd_experiment_results.csv")
        df_sorted.to_csv(csv_path, index=False)
        
        # Print simple summary
        print("\n" + "="*60)
        print("MMD EXPERIMENT EVALUATION SUMMARY")
        print("="*60)
        print(f"{'Experiment':<20} {'Beta':<8} {'Histogram':<10} {'Reduction':<12} {'MMD_Baseline':<15} {'MMD_After':<12}")
        print("-" * 80)
        
        for _, row in df_sorted.iterrows():
            hist_str = str(row['histogram']) if pd.notna(row['histogram']) else 'None'
            print(f"{row['experiment']:<20} {row['beta']:<8.3f} {hist_str:<10} {row['reduction_percent']:+12.2f}% {row['mmd_baseline']:<15.6f} {row['mmd_after']:<12.6f}")
        
        print("-" * 80)
        best_exp = df_sorted.iloc[0]
        print(f"\nBest performer: {best_exp['experiment']} with {best_exp['reduction_percent']:+.2f}% MMD reduction")
        print(f"Results saved to: {csv_path}")
        print("="*60)
        
    

def main():
    """Main function following the structure of MMD_fda_evaluation_byfolder.py"""
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
    evaluator = ExperimentEvaluator()
    
    # Run evaluation (this will evaluate all 9 experiments automatically)
    print("\n=== Running MMD Evaluation for All Experiments ===")
    results = evaluator.evaluate_experiments(
        experiments_dir='experiments',
        batch_size=32,
        output_root="mmd_experiment_results"
    )
    
    if results:
        print(f"\n=== MMD Evaluation Complete ===")
        print(f"Evaluated {len(results)} experiments")
    else:
        print("No experiments were evaluated.")

if __name__ == "__main__":
    main()
