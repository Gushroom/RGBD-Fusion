import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from models import UpBlock


class DepthGuidedSEFusion(nn.Module):
    """
    SE fusion with Cross-modal interaction (joint attention)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Joint squeeze-and-excitation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb, depth):
        # Global context from both modalities
        rgb_pool = self.gap(rgb)
        depth_pool = self.gap(depth)
        
        # Generate attention from joint context
        joint = torch.cat([rgb_pool, depth_pool], dim=1)
        attention = self.fc(joint)
        
        # Modulate RGB with depth-guided attention
        refined = rgb * attention
        return rgb + refined 


class ResNetUNetSEFusion(nn.Module):
    """
    Fixed SE fusion for small datasets (~250 samples).
    
    Key improvements:
    - Late fusion only (at bottleneck) - minimal new parameters
    - Cross-modal SE attention with residual connection
    - RGB pathway identical to baseline (preserves 70% mIoU)
    - Depth as refinement signal, not replacement
    """
    def __init__(self, num_classes=32, pretrained=True):
        super().__init__()

        # ===== RGB ENCODER (Main pathway - unchanged from baseline) =====
        rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.rgb_conv1 = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu)
        self.rgb_maxpool = rgb.maxpool
        self.rgb_l1 = rgb.layer1  # 64 channels
        self.rgb_l2 = rgb.layer2  # 128 channels
        self.rgb_l3 = rgb.layer3  # 256 channels
        self.rgb_l4 = rgb.layer4  # 512 channels

        depth = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Initialize depth conv1 from RGB pretrained weights
        depth.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            depth.conv1.weight.data = rgb.conv1.weight.data.mean(1, keepdim=True)

        self.depth_conv1 = nn.Sequential(depth.conv1, depth.bn1, depth.relu)
        self.depth_maxpool = depth.maxpool
        self.depth_l1 = depth.layer1
        self.depth_l2 = depth.layer2
        self.depth_l3 = depth.layer3
        self.depth_l4 = depth.layer4

        self.fusion = DepthGuidedSEFusion(512, reduction=16)
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256 + 256, 128)
        self.up2 = UpBlock(128 + 128, 64)
        self.up1 = UpBlock(64 + 64, 64)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, rgb, depth):
        # RGB encoder
        r0 = self.rgb_conv1(rgb)
        r1 = self.rgb_l1(self.rgb_maxpool(r0))
        r2 = self.rgb_l2(r1)
        r3 = self.rgb_l3(r2)
        r4 = self.rgb_l4(r3)

        # Depth encoder
        d0 = self.depth_conv1(depth)
        d1 = self.depth_l1(self.depth_maxpool(d0))
        d2 = self.depth_l2(d1)
        d3 = self.depth_l3(d2)
        d4 = self.depth_l4(d3)

        # FUSION: Depth refines RGB at bottleneck only
        fused = self.fusion(r4, d4)

        # Decoder (uses fused bottleneck + RGB-only skip connections)
        # Skip connections from RGB preserve baseline quality
        u4 = self.up4(fused)
        u4 = torch.cat([u4, r3], dim=1)  # Skip from RGB

        u3 = self.up3(u4)
        u3 = torch.cat([u3, r2], dim=1)  # Skip from RGB

        u2 = self.up2(u3)
        u2 = torch.cat([u2, r1], dim=1)  # Skip from RGB

        u1 = self.up1(u2)
        u0 = self.up0(u1)
        
        return self.final(u0)