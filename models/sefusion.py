import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from models import UpBlock


class SEFusion(nn.Module):
    """Tiny SE fusion for two feature maps with same channels."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, depth):
        w_rgb = self.fc(rgb)
        w_depth = self.fc(depth)
        return w_rgb * rgb + w_depth * depth


class ResNetUNetSEFusion(nn.Module):
    """
    SE-based attention fusion (very lightweight).
    Fusion happens at encoder layer2 + layer3.
    """
    def __init__(self, num_classes=32, pretrained=True):
        super().__init__()

        # RGB encoder
        rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.rgb_conv1 = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu)
        self.rgb_maxpool = rgb.maxpool
        self.rgb_l1 = rgb.layer1
        self.rgb_l2 = rgb.layer2
        self.rgb_l3 = rgb.layer3
        self.rgb_l4 = rgb.layer4

        # Depth encoder
        depth = resnet18(weights=None)
        depth.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)

        if pretrained:
            depth.conv1.weight.data = rgb.conv1.weight.data.mean(1, keepdim=True)

        self.depth_conv1 = nn.Sequential(depth.conv1, depth.bn1, depth.relu)
        self.depth_maxpool = depth.maxpool
        self.depth_l1 = depth.layer1
        self.depth_l2 = depth.layer2
        self.depth_l3 = depth.layer3
        self.depth_l4 = depth.layer4

        # --- Attention Fusion ---
        self.fuse_l2 = SEFusion(128)
        self.fuse_l3 = SEFusion(256)

        # Decoder
        self.up4 = UpBlock(512 * 2, 256)
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

        # Depth
        d0 = self.depth_conv1(depth)
        d1 = self.depth_l1(self.depth_maxpool(d0))
        d2 = self.depth_l2(d1)
        d3 = self.depth_l3(d2)
        d4 = self.depth_l4(d3)

        # Fuse
        f2 = self.fuse_l2(r2, d2)
        f3 = self.fuse_l3(r3, d3)

        # Decode
        f4 = torch.cat([r4, d4], dim=1)
        u4 = self.up4(f4)
        u4 = torch.cat([u4, f3], dim=1)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, f2], dim=1)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, r1], dim=1)

        u1 = self.up1(u2)
        u0 = self.up0(u1)
        return self.final(u0)
