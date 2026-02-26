import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import torchvision
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import FDA_source_to_target_np
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class ResNet101FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet101FeatureExtractor, self).__init__()
        print("Loading Pre-trained ResNet101 as feature extractor...")
        
        # Fixed: resnet and self.backbone are now correctly inside __init__
        resnet = torchvision.models.resnet101(pretrained=True)
        # This gives us a 2048-dimensional feature vector
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # Shape: [batch, 3, 224, 224] -> [batch, 2048, 1, 1]
        features = self.backbone(x)
        # Flatten to [batch, 2048]
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

class FDAEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Using the corrected ResNet101 extractor
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

    def apply_fda_transformation(self, source_paths, target_paths, L=0.01, output_dir="fda_transformed"):
        os.makedirs(output_dir, exist_ok=True)
        transformed_paths = []

        print(f"Applying FDA (L={L}) to {len(source_paths)} images...")

        # Limit target paths to match source length
        target_paths = target_paths[:len(source_paths)]

        for src_path, trg_path in zip(source_paths, target_paths):
            try:
                src_img = Image.open(src_path).convert('RGB').resize((512, 512))
                trg_img = Image.open(trg_path).convert('RGB').resize((512, 512))

                src_array = np.asarray(src_img, np.float32).transpose((2, 0, 1)) / 255.0
                trg_array = np.asarray(trg_img, np.float32).transpose((2, 0, 1)) / 255.0

                # Core FDA algorithm call
                fda_img = FDA_source_to_target_np(src_array, trg_array, L=L)

                fda_img = fda_img.transpose((1, 2, 0))
                fda_img = np.clip(fda_img * 255.0, 0, 255).astype(np.uint8)

                out_path = os.path.join(output_dir, f"fda_{L}_{os.path.basename(src_path)}")
                Image.fromarray(fda_img).save(out_path)
                transformed_paths.append(out_path)
            except Exception as e:
                print(f"Failed on {src_path}: {e}")

        return transformed_paths

    def compute_tsne_visualization(self, feat_orig, feat_fda, feat_real, paths, L, out_dir):
        print(f"Computing t-SNE for L={L}...")
        all_feats = np.vstack([feat_orig, feat_fda, feat_real])
        labels = (['Original CG'] * len(feat_orig) +
                  ['FDA CG'] * len(feat_fda) +
                  ['Real'] * len(feat_real))

        tsne = TSNE(n_components=2, random_state=42, perplexity=10)
        res = tsne.fit_transform(all_feats)

        df = pd.DataFrame({'x': res[:, 0], 'y': res[:, 1], 'Domain': labels})

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='Domain',
                        palette={'Original CG': 'red', 'FDA CG': 'green', 'Real': 'blue'},
                        s=100, alpha=0.8, edgecolor='black')

        # Calculate centroids
        c = df.groupby('Domain')[['x', 'y']].mean()
        dist_orig = np.linalg.norm(c.loc['Original CG'] - c.loc['Real'])
        dist_fda = np.linalg.norm(c.loc['FDA CG'] - c.loc['Real'])
        reduction = ((dist_orig - dist_fda) / dist_orig) * 100

        # Draw lines from clusters to the Real cluster
        plt.plot([c.loc['Original CG', 'x'], c.loc['Real', 'x']],
                 [c.loc['Original CG', 'y'], c.loc['Real', 'y']], 'r--', alpha=0.5, label='Original Gap')
        plt.plot([c.loc['FDA CG', 'x'], c.loc['Real', 'x']],
                 [c.loc['FDA CG', 'y'], c.loc['Real', 'y']], 'g--', alpha=0.5, label='FDA Gap')

        plt.title(f'ResNet101 t-SNE (L={L})\nGap Reduction: {reduction:.2f}%', fontsize=14)
        plt.legend()

        save_path = os.path.join(out_dir, f"tsne_plot_L_{L}.png")
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.show()

        return reduction

    def evaluate_domain_gap(self, cg_dir, real_dir, L_values=[0.01]):
        out_dir = "fda_evaluation_results"
        os.makedirs(out_dir, exist_ok=True)

        cg_paths = sorted(glob.glob(os.path.join(cg_dir, "*.png")))[:100]
        real_paths = sorted(glob.glob(os.path.join(real_dir, "*.png")))[:100]

        print("Extracting Baseline Features...")
        f_orig, p_orig = self.extract_features(cg_paths)
        f_real, p_real = self.extract_features(real_paths)

        for L in L_values:
            print(f"\n--- Testing L = {L} ---")
            fda_paths = self.apply_fda_transformation(cg_paths, real_paths, L=L, output_dir=os.path.join(out_dir, f"images_L_{L}"))
            f_fda, p_fda = self.extract_features(fda_paths)

            reduction = self.compute_tsne_visualization(f_orig, f_fda, f_real, cg_paths+fda_paths+real_paths, L, out_dir)
            print(f"Results for L={L}: {reduction:.2f}% Improvement")

def main():
    CG_PATH = "sample_data/sample/02_omniverse/_defects.pos4/rgb"
    REAL_PATH = "sample_data/sample/01_original/train"

    evaluator = FDAEvaluator()
    evaluator.evaluate_domain_gap(CG_PATH, REAL_PATH, L_values=[0.001, 0.01, 0.05, 0.1])

if __name__ == "__main__":
    main()