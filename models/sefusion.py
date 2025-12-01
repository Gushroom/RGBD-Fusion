import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from models import UpBlock


class AdaptiveGatedFusion(nn.Module):
    """
    Learns to weight RGB vs Depth based on feature quality.
    Key insight: Use depth features to estimate RGB reliability.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Estimate RGB quality from both modalities
        self.quality_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(channels // reduction, 2),  # 2 gates: rgb_weight, depth_weight
        )
        
        # Channel attention for refined fusion
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention to focus on informative regions
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize to favor RGB (pretrained)
        self._init_rgb_bias()
    
    def _init_rgb_bias(self):
        # Initialize final linear layer to output [1, 0] (pure RGB)
        nn.init.constant_(self.quality_estimator[-1].weight, 0)
        nn.init.constant_(self.quality_estimator[-1].bias[0], 1.0)  # RGB weight
        nn.init.constant_(self.quality_estimator[-1].bias[1], 0.0)  # Depth weight
        
    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.shape
        
        # Step 1: Estimate modality weights
        joint = torch.cat([rgb_feat, depth_feat], dim=1)
        weights = self.quality_estimator(joint)  # [B, 2]
        weights = F.softmax(weights, dim=1)
        rgb_w, depth_w = weights[:, 0:1, None, None], weights[:, 1:2, None, None]
        
        # Step 2: Weighted combination
        fused = rgb_w * rgb_feat + depth_w * depth_feat
        
        # Step 3: Channel attention refinement
        ca = self.channel_attn(fused)
        fused = fused * ca
        
        # Step 4: Spatial attention
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        fused = fused * spatial
        
        return fused, weights.squeeze()


class DepthConfidenceModule(nn.Module):
    """
    Estimates per-pixel depth reliability.
    Depth is reliable where gradients exist (edges, surfaces).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, depth_feat):
        return self.conv(depth_feat)


class ResNetUNetSEFusion(nn.Module):
    """
    Multi-scale RGB-D fusion with adaptive gating.
    
    Key improvements over SE fusion:
    1. Multi-scale fusion (not just bottleneck)
    2. Adaptive gating learns when to trust depth
    3. Depth confidence weighting
    4. Skip connections from BOTH modalities (weighted)
    """
    def __init__(self, num_classes=32, pretrained=True):
        super().__init__()
        
        # ===== RGB ENCODER =====
        rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.rgb_conv1 = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu)
        self.rgb_maxpool = rgb.maxpool
        self.rgb_l1 = rgb.layer1  # 64 ch
        self.rgb_l2 = rgb.layer2  # 128 ch
        self.rgb_l3 = rgb.layer3  # 256 ch
        self.rgb_l4 = rgb.layer4  # 512 ch
        
        # ===== DEPTH ENCODER =====
        depth = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        depth.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            depth.conv1.weight.data = rgb.conv1.weight.data.mean(1, keepdim=True)
        
        self.depth_conv1 = nn.Sequential(depth.conv1, depth.bn1, depth.relu)
        self.depth_maxpool = depth.maxpool
        self.depth_l1 = depth.layer1
        self.depth_l2 = depth.layer2
        self.depth_l3 = depth.layer3
        self.depth_l4 = depth.layer4
        
        # ===== MULTI-SCALE FUSION =====
        # Fusion at multiple scales captures both coarse and fine depth cues
        self.fusion4 = AdaptiveGatedFusion(512)  # Bottleneck
        self.fusion3 = AdaptiveGatedFusion(256)  # 1/16
        self.fusion2 = AdaptiveGatedFusion(128)  # 1/8
        self.fusion1 = AdaptiveGatedFusion(64)   # 1/4
        
        # Depth confidence at each scale
        self.depth_conf4 = DepthConfidenceModule(512)
        self.depth_conf3 = DepthConfidenceModule(256)
        self.depth_conf2 = DepthConfidenceModule(128)
        self.depth_conf1 = DepthConfidenceModule(64)
        
        # ===== DECODER =====
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256 + 256, 128)  # + skip
        self.up2 = UpBlock(128 + 128, 64)
        self.up1 = UpBlock(64 + 64, 64)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # ===== OUTPUT =====
        self.final = nn.Conv2d(32, num_classes, 1)
        
        # Freeze depth encoder initially for stability
        self._freeze_depth_encoder()
    
    def _freeze_depth_encoder(self):
        """Freeze depth encoder for first N epochs"""
        for param in [self.depth_conv1, self.depth_l1, self.depth_l2]:
            for p in param.parameters():
                p.requires_grad = False
    
    def unfreeze_depth_encoder(self):
        """Call after warmup epochs"""
        for param in [self.depth_conv1, self.depth_l1, self.depth_l2]:
            for p in param.parameters():
                p.requires_grad = True
    
    def forward(self, rgb, depth):
        # ===== ENCODE =====
        # RGB pathway
        r0 = self.rgb_conv1(rgb)
        r1 = self.rgb_l1(self.rgb_maxpool(r0))
        r2 = self.rgb_l2(r1)
        r3 = self.rgb_l3(r2)
        r4 = self.rgb_l4(r3)
        
        # Depth pathway
        d0 = self.depth_conv1(depth)
        d1 = self.depth_l1(self.depth_maxpool(d0))
        d2 = self.depth_l2(d1)
        d3 = self.depth_l3(d2)
        d4 = self.depth_l4(d3)
        
        # ===== MULTI-SCALE FUSION =====
        # Apply depth confidence weighting before fusion
        d4_conf = self.depth_conf4(d4)
        d3_conf = self.depth_conf3(d3)
        d2_conf = self.depth_conf2(d2)
        d1_conf = self.depth_conf1(d1)
        
        d4_weighted = d4 * d4_conf
        d3_weighted = d3 * d3_conf
        d2_weighted = d2 * d2_conf
        d1_weighted = d1 * d1_conf
        
        # Fuse at each scale
        f4, w4 = self.fusion4(r4, d4_weighted)
        f3, w3 = self.fusion3(r3, d3_weighted)
        f2, w2 = self.fusion2(r2, d2_weighted)
        f1, w1 = self.fusion1(r1, d1_weighted)
        
        # ===== DECODE =====
        u4 = self.up4(f4)
        u4 = torch.cat([u4, f3], dim=1)  # Fused skip
        
        u3 = self.up3(u4)
        u3 = torch.cat([u3, f2], dim=1)
        
        u2 = self.up2(u3)
        u2 = torch.cat([u2, f1], dim=1)
        
        u1 = self.up1(u2)
        u0 = self.up0(u1)
        
        return self.final(u0)