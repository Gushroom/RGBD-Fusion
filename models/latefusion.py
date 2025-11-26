from models import ResNetUNet
import torch.nn as nn

class ResNetUNetLateFusion(nn.Module):
    """
    Late-Fusion model: two independent encoders and decoders.
    Fuse predictions at the end.
    """
    def __init__(self, num_classes=32, pretrained=True):
        super().__init__()

        # RGB U-Net
        self.rgb_unet = ResNetUNet(num_classes=num_classes,
                                   in_channels=3,
                                   pretrained=pretrained)

        # Depth U-Net (modify first layer for 1 channel)
        self.depth_unet = ResNetUNet(num_classes=num_classes,
                                     in_channels=1,
                                     pretrained=False)

        if pretrained:
            # initialize depth conv1 as grayscale weights from RGB model
            conv_rgb = self.rgb_unet.conv1[0].weight.data
            self.depth_unet.conv1[0].weight.data = conv_rgb.mean(dim=1, keepdim=True)

    def forward(self, rgb, depth):
        rgb_logits = self.rgb_unet(rgb)
        depth_logits = self.depth_unet(depth)

        # Late fusion: average or max
        return (rgb_logits + depth_logits) / 2.0


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
