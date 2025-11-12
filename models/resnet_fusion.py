import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetSingleModality(nn.Module):
    """ResNet-18 for single modality (RGB or Depth)"""
    
    def __init__(self, num_classes=30, in_channels=3, pretrained=True):
        super().__init__()
        
        # Load ResNet-18
        if pretrained and in_channels == 3:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.encoder = resnet18(weights=weights)
        else:
            self.encoder = resnet18(weights=None)
        
        # Modify first conv for depth (1 channel)
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace classifier
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.encoder(x)


class RGBDFusionModel(nn.Module):
    """ResNet-18 fusion model for RGB + Depth"""
    
    def __init__(
        self,
        num_classes=30,
        fusion_type='concat',  # 'concat', 'add', 'multiply', 'learned'
        pretrained=True
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # RGB encoder (pretrained on ImageNet)
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.rgb_encoder = resnet18(weights=weights)
        else:
            self.rgb_encoder = resnet18(weights=None)
        
        # Remove final FC layer
        self.rgb_encoder.fc = nn.Identity()
        
        # Depth encoder (trained from scratch)
        self.depth_encoder = resnet18(weights=None)
        # Modify for single channel input
        self.depth_encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.depth_encoder.fc = nn.Identity()
        
        # Feature dimension from ResNet-18
        feature_dim = 512
        
        # Fusion and classification
        if fusion_type == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim * 2, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif fusion_type == 'learned':
            # Learnable weighted fusion
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:  # 'add' or 'multiply'
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, rgb, depth):
        # Extract features
        rgb_features = self.rgb_encoder(rgb)      # [B, 512]
        depth_features = self.depth_encoder(depth)  # [B, 512]
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_features, depth_features], dim=1)  # [B, 1024]
        elif self.fusion_type == 'add':
            fused = rgb_features + depth_features  # [B, 512]
        elif self.fusion_type == 'multiply':
            fused = rgb_features * depth_features  # [B, 512]
        elif self.fusion_type == 'learned':
            alpha = torch.sigmoid(self.fusion_weight)
            fused = alpha * rgb_features + (1 - alpha) * depth_features  # [B, 512]
        
        # Classification
        output = self.classifier(fused)
        
        return output


class RGBDEarlyFusion(nn.Module):
    """Early fusion: Concatenate RGB + Depth at input"""
    
    def __init__(self, num_classes=30, pretrained=False):
        super().__init__()
        
        # ResNet-18 with 4 input channels (RGB + D)
        self.encoder = resnet18(weights=None)
        
        # Modify first conv for 4 channels
        self.encoder.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # If pretrained RGB weights available, initialize RGB channels
        if pretrained:
            pretrained_weights = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            pretrained_conv1 = pretrained_weights.conv1.weight.data
            
            # Initialize RGB channels with pretrained weights
            self.encoder.conv1.weight.data[:, :3, :, :] = pretrained_conv1
            # Initialize depth channel with average of RGB channels
            self.encoder.conv1.weight.data[:, 3:, :, :] = pretrained_conv1.mean(dim=1, keepdim=True)
        
        # Replace classifier
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, rgb, depth):
        # Concatenate RGB and Depth
        x = torch.cat([rgb, depth], dim=1)  # [B, 4, H, W]
        return self.encoder(x)


# Test the models
if __name__ == '__main__':
    batch_size = 4
    num_classes = 30
    
    # Dummy data
    rgb = torch.randn(batch_size, 3, 224, 224)
    depth = torch.randn(batch_size, 1, 224, 224)
    
    print("Testing models...\n")
    
    # Test single modality
    print("1. RGB-only model:")
    model_rgb = ResNetSingleModality(num_classes=num_classes, in_channels=3, pretrained=False)
    out_rgb = model_rgb(rgb)
    print(f"   Input: {rgb.shape} -> Output: {out_rgb.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_rgb.parameters()):,}")
    
    print("\n2. Depth-only model:")
    model_depth = ResNetSingleModality(num_classes=num_classes, in_channels=1, pretrained=False)
    out_depth = model_depth(depth)
    print(f"   Input: {depth.shape} -> Output: {out_depth.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_depth.parameters()):,}")
    
    print("\n3. Late fusion (concat):")
    model_late = RGBDFusionModel(num_classes=num_classes, fusion_type='concat', pretrained=False)
    out_late = model_late(rgb, depth)
    print(f"   Input: RGB {rgb.shape} + Depth {depth.shape} -> Output: {out_late.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_late.parameters()):,}")
    
    print("\n4. Early fusion:")
    model_early = RGBDEarlyFusion(num_classes=num_classes, pretrained=False)
    out_early = model_early(rgb, depth)
    print(f"   Input: RGB {rgb.shape} + Depth {depth.shape} -> Output: {out_early.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_early.parameters()):,}")
    
    print("\nAll models working correctly! ✓")