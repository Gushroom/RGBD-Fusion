import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetUNetMidFusion(nn.Module):
    """
    Mid-Fusion U-Net (RGB + D fused after encoder layer2)
    """
    def __init__(self, num_classes=32, pretrained=True):
        super().__init__()

        # ---------- RGB Encoder ----------
        rgb_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.rgb_conv1 = nn.Sequential(rgb_resnet.conv1, rgb_resnet.bn1, rgb_resnet.relu)
        self.rgb_maxpool = rgb_resnet.maxpool
        self.rgb_layer1 = rgb_resnet.layer1
        self.rgb_layer2 = rgb_resnet.layer2
        self.rgb_layer3 = rgb_resnet.layer3
        self.rgb_layer4 = rgb_resnet.layer4

        # ---------- Depth Encoder ----------
        depth_resnet = resnet18(weights=None)
        depth_resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)

        # copy pretrained RGB → depth as grayscale if desired
        if pretrained:
            pretrained_r = rgb_resnet.conv1.weight.data
            depth_resnet.conv1.weight.data = pretrained_r.mean(dim=1, keepdim=True)

        self.depth_conv1 = nn.Sequential(depth_resnet.conv1, depth_resnet.bn1, depth_resnet.relu)
        self.depth_maxpool = depth_resnet.maxpool
        self.depth_layer1 = depth_resnet.layer1
        self.depth_layer2 = depth_resnet.layer2
        self.depth_layer3 = depth_resnet.layer3
        self.depth_layer4 = depth_resnet.layer4

        # ---------- Shared decoder ----------
        self.up4 = UpBlock(1024, 256)
        self.up3 = UpBlock(512, 128)
        self.up2 = UpBlock(384, 64)
        self.up1 = UpBlock(128, 64)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, rgb, depth):

        # ----- RGB -----
        r0 = self.rgb_conv1(rgb)
        r1 = self.rgb_layer1(self.rgb_maxpool(r0))
        r2 = self.rgb_layer2(r1)
        r3 = self.rgb_layer3(r2)
        r4 = self.rgb_layer4(r3)

        # ----- Depth -----
        d0 = self.depth_conv1(depth)
        d1 = self.depth_layer1(self.depth_maxpool(d0))
        d2 = self.depth_layer2(d1)
        d3 = self.depth_layer3(d2)
        d4 = self.depth_layer4(d3)

        # ----- Fusion -----
        f2 = torch.cat([r2, d2], dim=1)  # fuse here

        # ----- Decode -----
        f4 = torch.cat([r4, d4], dim=1)
        d4_up = self.up4(f4)
        d4_up = torch.cat([d4_up, r3], dim=1)

        d3_up = self.up3(d4_up)
        d3_up = torch.cat([d3_up, f2], dim=1)

        d2_up = self.up2(d3_up)
        d2_up = torch.cat([d2_up, r1], dim=1)

        d1_up = self.up1(d2_up)
        out = self.up0(d1_up)

        return self.final(out)
    
class UpBlock(nn.Module):
    """Upsampling block for decoder"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                     kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
