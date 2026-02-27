import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils import FDA_source_to_target_np

# ==================== CMD Implementation ====================

def cmd_loss(x, y, n_moments=5):
    """
    x (torch.Tensor): Features from the source domain [N, D].
    y (torch.Tensor): Features from the target domain [M, D].
    n_moments (int): Number of higher-order central moments to match.
    Returns:
        float: The cumulative discrepancy value.
    """
    # Ensure inputs are 2D tensors [Batch, Features]
    if x.dim() == 1: x = x.unsqueeze(0)
    if y.dim() == 1: y = y.unsqueeze(0)

    # First-order moment (Mean) discrepancy
    # This aligns the "center" of the two distributions
    mu_x = torch.mean(x, dim=0)
    mu_y = torch.mean(y, dim=0)
    mean_diff = torch.norm(mu_x - mu_y, p=2)

    # Higher-order central moments discrepancy
    # Centralize data: subtract the mean to focus on the "shape" of the distribution
    centralized_x = x - mu_x
    centralized_y = y - mu_y
    
    moment_diffs = 0
    for i in range(2, n_moments+1):
        # Calculate the i-th central moment: E[(X - E[X])^i]
        moment_x = torch.mean(torch.pow(centralized_x, i), dim=0)
        moment_y = torch.mean(torch.pow(centralized_y, i), dim=0)
        
        # Accumulate L2 distance between the i-th moments
        moment_diffs += torch.norm(moment_x - moment_y, p=2)

    total_cmd = mean_diff + moment_diffs
    return total_cmd.item()
    

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
        if image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
        
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        img_tensor = self.preprocess(image_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(img_tensor).view(1, -1)
            # sigmoid
            # features = torch.sigmoid(features)
            #L1 
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        
        return features.cpu()

# ==================== FDA Evaluator ====================

class FDAEvaluator:
    def __init__(self, device='cuda'):
        
        self.fe = FeatureExtractor(device)
        
    def evaluate(self, src_img, trg_img, beta_list, output_dir):
        src_feat = self.fe.extract_features(src_img)
        trg_feat = self.fe.extract_features(trg_img)

        cmd_before = cmd_loss(src_feat, trg_feat)
        
        results = []

        print("\n" + "="*60)
        print(f"FDA Evaluation (CMD) ")
        print("="*60)
        print(f"{'Beta (L)':<10} | {'CMD Before':<12} | {'CMD After':<12} ")
        print("-" * 60)

        for beta in beta_list:
            src_fda = FDA_source_to_target_np(src_img, trg_img, L=beta)
            fda_feat = self.fe.extract_features(src_fda)

            cmd_after = cmd_loss(fda_feat, trg_feat)

            # reduction = (cmd_before - cmd_after) / cmd_before * 100 if cmd_before > 0 else 0
            
            results.append({
                'beta': beta,
                'cmd_before': cmd_before,
                'cmd_after': cmd_after,
                # 'reduction_percent': reduction
            })
            

            print(f"{beta:<10.4f} | {cmd_before:<12.6f} | {cmd_after:<12.6f} ")

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

    

    if im_src.size != im_trg.size:
        im_trg = im_trg.resize(im_src.size, Image.BICUBIC)
    # To [C, H, W] float32
    np_src = np.asarray(im_src, np.float32) / 255.0
    np_trg = np.asarray(im_trg, np.float32) / 255.0
    np_src = np_src.transpose(2, 0, 1)
    np_trg = np_trg.transpose(2, 0, 1)

    evaluator = FDAEvaluator()
    evaluator.evaluate(np_src, np_trg, [0.01, 0.001,0.0003,0.0005,0.0008,0.0001], output_dir)
#[0.01, 0.001,0.0003,0.0005,0.0008,0.0001]
if __name__ == "__main__":
    main()