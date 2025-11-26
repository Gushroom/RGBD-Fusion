import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from models import UpBlock

class ResNetUNet(nn.Module):
    """ResNet18-based U-Net for semantic segmentation (RGB only baseline)"""
    
    def __init__(self, num_classes=32, in_channels=3, pretrained=True):
        super().__init__()
        
        # Load ResNet18 as encoder
        if pretrained and in_channels == 3:
            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
        else:
            resnet = resnet18(weights=None)
        
        # Modify first conv if needed (for depth)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                                     padding=3, bias=False)
        
        # Encoder layers (ResNet blocks)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Decoder (upsampling path)
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256 + 256, 128)  # +256 from skip connection
        self.up2 = UpBlock(128 + 128, 64)   # +128 from skip connection
        self.up1 = UpBlock(64 + 64, 64)     # +64 from skip connection
        
        # Final upsampling to original resolution
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x0 = self.conv1(x)           # [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)        # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)         # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)         # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)         # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)         # [B, 512, H/32, W/32]
        
        # Decoder with skip connections
        d4 = self.up4(x4)            # [B, 256, H/16, W/16]
        d4 = torch.cat([d4, x3], dim=1)  # Skip connection
        
        d3 = self.up3(d4)            # [B, 128, H/8, W/8]
        d3 = torch.cat([d3, x2], dim=1)  # Skip connection
        
        d2 = self.up2(d3)            # [B, 64, H/4, W/4]
        d2 = torch.cat([d2, x1], dim=1)  # Skip connection
        
        d1 = self.up1(d2)            # [B, 64, H/2, W/2]
        d0 = self.up0(d1)            # [B, 32, H, W]
        
        out = self.final(d0)         # [B, num_classes, H, W]
        
        return out