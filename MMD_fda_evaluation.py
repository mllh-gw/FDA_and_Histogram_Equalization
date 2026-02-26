import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils import FDA_source_to_target_np

# ==================== MMD Implementation ====================

def mmd_square(x, y, sigmas=[1.0, 2.0, 5.0, 10.0, 20.0]):
    """
    ComputesMMD for both single pairs and batches.
    Uses multi-scale kernel kernels to avoid the 'distance collapse' problem.
    """
    if x.dim() == 1: x = x.unsqueeze(0)
    if y.dim() == 1: y = y.unsqueeze(0)
    
    n = x.size(0)
    m = y.size(0)
    
    # Compute pairwise distances
    x_kernel = compute_kernel(x, x, sigmas)
    y_kernel = compute_kernel(y, y, sigmas)
    xy_kernel = compute_kernel(x, y, sigmas)

    if n > 1 and m > 1:
        # Unbiased estimator for batches --> not only processed with single img
        k_xx = (x_kernel.sum() - x_kernel.diag().sum()) / (n * (n - 1))
        k_yy = (y_kernel.sum() - y_kernel.diag().sum()) / (m * (m - 1))
        k_xy = xy_kernel.mean()
    else:
        # single img
        k_xx = x_kernel.mean()
        # print(k_xx)
        k_yy = y_kernel.mean()
        # print(k_yy)
        k_xy = xy_kernel.mean()
        # print(k_xy)

    return (k_xx + k_yy - 2 * k_xy).item()

def compute_kernel(x, y, sigmas):
    """Computes multi-scale kernel kernel matrix"""
    # Calculate squared euclidean distance
    dist = torch.pow(x, 2).sum(1).view(-1, 1) + \
           torch.pow(y, 2).sum(1).view(1, -1) - \
           2.0 * torch.mm(x, y.t())
    
    dist = torch.clamp(dist, min=0.0)
    
    kernel_val = 0
    for s in sigmas:
        gamma = 1.0 / (2.0 * s**2)
        kernel_val += torch.exp(-gamma * dist)
    
    return kernel_val / len(sigmas)
    

# ==================== Feature Extractor ====================

class FeatureExtractor:
    def __init__(self, device='cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load ResNet and take up to Layer 4
        full_resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2, 
            full_resnet.layer3,
            full_resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # full_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # self.feature_extractor = nn.Sequential(
        #     full_resnet.conv1,
        #     full_resnet.bn1,
        #     full_resnet.relu,
        #     full_resnet.maxpool,
        #     full_resnet.layer1,
        #     full_resnet.layer2, 
        #     full_resnet.layer3,
        #     full_resnet.layer4,
        #     nn.AdaptiveAvgPool2d((1, 1))
        # ).to(self.device).eval()

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_np):
        # Convert [C, H, W] to [H, W, C] if needed
        if image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
        
        # Ensure 0-255 uint8 for ToPILImage
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        img_tensor = self.preprocess(image_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(img_tensor).view(1, -1)
            # L2 Normalization makes MMD
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu()

# ==================== FDA Evaluator ====================

class FDAEvaluator:
    def __init__(self, device='cuda'):
        
        self.fe = FeatureExtractor(device)
        
    def evaluate(self, src_img, trg_img, beta_list, output_dir):
        src_feat = self.fe.extract_features(src_img)
        trg_feat = self.fe.extract_features(trg_img)

        mmd_before = mmd_square(src_feat, trg_feat)
        
        results = []

        print("\n" + "="*60)
        print(f"FDA ")
        print("="*60)
        print(f"{'Beta (L)':<10} | {'MMD Before':<12} | {'MMD After':<12} ")
        print("-" * 60)

        for beta in beta_list:
            src_fda = FDA_source_to_target_np(src_img, trg_img, L=beta)
            fda_feat = self.fe.extract_features(src_fda)

            mmd_after = mmd_square(fda_feat, trg_feat)

            # reduction = (mmd_before - mmd_after) / mmd_before * 100 if mmd_before > 0 else 0
            
            results.append({
                'beta': beta,
                'mmd_before': mmd_before,
                'mmd_after': mmd_after,
                # 'reduction_percent': reduction
            })
            

            print(f"{beta:<10.4f} | {mmd_before:<12.6f} | {mmd_after:<12.6f} ")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_img = (src_fda.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(save_img).save(os.path.join(output_dir, f"fda_beta_{beta}.png"))
        
        print("="*60)
        return results

# main

def main():
    src_path="sample_data/omniverse/rgb_0000.png"
    trg_path = "sample_data/original/1_7000_0.png"
    output_dir = "res"
    

    im_src = Image.open(src_path).convert('RGB')
    im_trg = Image.open(trg_path).convert('RGB')
    # im_src = Image.open(src_path).convert('RGB').resize((2048, 1024))
    # im_trg = Image.open(trg_path).convert('RGB').resize((2048, 1024))
    if im_src.size != im_trg.size:
        im_trg = im_trg.resize(im_src.size, Image.BICUBIC)
    # To [C, H, W] float32
    np_src = np.asarray(im_src, np.float32) / 255.0
    np_trg = np.asarray(im_trg, np.float32) / 255.0
    np_src = np_src.transpose(2, 0, 1)
    np_trg = np_trg.transpose(2, 0, 1)
    
    evaluator = FDAEvaluator()
    evaluator.evaluate(np_src, np_trg, [0.01, 0.001,0.0003,0.0005,0.0008,0.0001], output_dir)

if __name__ == "__main__":
    main()

