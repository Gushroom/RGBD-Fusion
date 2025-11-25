import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


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


class ResNetUNetEarlyFusion(nn.Module):
    """ResNet18 U-Net with early fusion (RGB+Depth concatenated at input)"""
    
    def __init__(self, num_classes=32, pretrained=True):
        super().__init__()
        
        # Load ResNet18
        resnet = resnet18(weights=None)
        
        # Modify first conv for 4 channels (RGB + D)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, 
                                padding=3, bias=False)
        
        # If pretrained, initialize RGB channels
        if pretrained:
            pretrained_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            pretrained_conv1 = pretrained_resnet.conv1.weight.data
            
            # Copy RGB weights
            resnet.conv1.weight.data[:, :3, :, :] = pretrained_conv1
            # Initialize depth channel as average of RGB
            resnet.conv1.weight.data[:, 3:, :, :] = pretrained_conv1.mean(dim=1, keepdim=True)
        
        # Encoder
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Decoder (same as RGB-only)
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256 + 256, 128)
        self.up2 = UpBlock(128 + 128, 64)
        self.up1 = UpBlock(64 + 64, 64)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, rgb, depth):
        # Concatenate at input
        x = torch.cat([rgb, depth], dim=1)  # [B, 4, H, W]
        
        # Encoder
        x0 = self.conv1(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        
        d1 = self.up1(d2)
        d0 = self.up0(d1)
        
        out = self.final(d0)
        
        return out


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


# Test the models
if __name__ == '__main__':
    batch_size = 2
    num_classes = 9
    img_size = 224
    
    # Dummy data
    rgb = torch.randn(batch_size, 3, img_size, img_size)
    depth = torch.randn(batch_size, 1, img_size, img_size)
    
    print("Testing segmentation models...\n")
    
    # Test RGB-only U-Net
    print("1. ResNet U-Net (RGB only):")
    model_rgb = ResNetUNet(num_classes=num_classes, pretrained=False)
    out_rgb = model_rgb(rgb)
    print(f"   Input: {rgb.shape} -> Output: {out_rgb.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_rgb.parameters()) / 1e6:.2f}M")
    
    # Test Early Fusion U-Net
    print("\n2. ResNet U-Net Early Fusion (RGB+D):")
    model_early = ResNetUNetEarlyFusion(num_classes=num_classes, pretrained=False)
    out_early = model_early(rgb, depth)
    print(f"   Input: RGB {rgb.shape} + Depth {depth.shape}")
    print(f"   Output: {out_early.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_early.parameters()) / 1e6:.2f}M")
    
    print("\n✓ All models working correctly!")