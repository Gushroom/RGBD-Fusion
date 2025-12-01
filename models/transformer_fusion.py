import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from models import UpBlock


class CrossModalTransformerFusion(nn.Module):
    """
    Cross-modal attention for RGB-D fusion
    Set up communication between RGB and D features
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Cross-attention layers
        self.rgb_to_depth_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.depth_to_rgb_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        # Layer norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.shape

        # Reshape to tokens. Flatten spatial dimensions for transformer
        rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)  # [B, N, C]
        depth_tokens = depth_feat.flatten(2).transpose(1, 2)  # [B, N, C]

        # Cross-attention: RGB attends to depth
        rgb_enhanced, _ = self.rgb_to_depth_attn(
            rgb_tokens, depth_tokens, depth_tokens
        )
        rgb_enhanced = self.norm1(rgb_tokens + rgb_enhanced) # residual connection

        # Cross-attention: Depth attends to RGB
        depth_enhanced, _ = self.depth_to_rgb_attn(
            depth_tokens, rgb_tokens, rgb_tokens
        )
        depth_enhanced = self.norm1(depth_tokens + depth_enhanced)

        # Fusion: Combine both modalities
        fused_tokens = rgb_enhanced + depth_enhanced

        # FFN
        fused_tokens = self.norm2(fused_tokens + self.ffn(fused_tokens))

        # Reshape back to feature map format for CNN decoder
        fused_feat = fused_tokens.transpose(1, 2).reshape(B, C, H, W)

        return fused_feat


class ResNetTransformerFusion(nn.Module):
    """
    Transformer-based RGB-D fusion with ResNet backbones
    """

    def __init__(self, num_classes=32, pretrained=True, embed_dim=256, num_heads=8, depth=4):
        super().__init__()

        # ===== RGB ENCODER =====
        # Using pretrained ResNet
        rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.rgb_conv1 = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu)
        self.rgb_maxpool = rgb.maxpool
        self.rgb_l1 = rgb.layer1  # 64
        self.rgb_l2 = rgb.layer2  # 128
        self.rgb_l3 = rgb.layer3  # 256
        self.rgb_l4 = rgb.layer4  # 512

        # ===== DEPTH ENCODER =====
        # first conv adjusted for single-channel depth
        depth_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        depth_net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            depth_net.conv1.weight.data = rgb.conv1.weight.data.mean(1, keepdim=True)

        self.depth_conv1 = nn.Sequential(depth_net.conv1, depth_net.bn1, depth_net.relu)
        self.depth_maxpool = depth_net.maxpool
        self.depth_l1 = depth_net.layer1
        self.depth_l2 = depth_net.layer2
        self.depth_l3 = depth_net.layer3
        self.depth_l4 = depth_net.layer4

        # ===== TRANSFORMER FUSION BLOCKS =====
        # Multi-scale fusion points
        self.fusion_l1 = CrossModalTransformerFusion(64, num_heads=num_heads)
        self.fusion_l2 = CrossModalTransformerFusion(128, num_heads=num_heads)
        self.fusion_l3 = CrossModalTransformerFusion(256, num_heads=num_heads)
        self.fusion_l4 = CrossModalTransformerFusion(512, num_heads=num_heads)

        # ===== BOTTLENECK TRANSFORMER =====
        # Additional transformer for final fusion
        self.bottleneck_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, nhead=num_heads, dim_feedforward=2048,
                batch_first=True, dropout=0.1
            ),
            num_layers=depth
        )

        # ===== DECODER =====
        # U-net style decoder with skip connections
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

        # Projection for bottleneck transformer
        self.bottleneck_proj = nn.Conv2d(512, 512, 1)

    def forward(self, rgb, depth):
        # ===== ENCODERS =====
        # RGB encoder
        r0 = self.rgb_conv1(rgb)
        r1 = self.rgb_l1(self.rgb_maxpool(r0))  # 64
        r2 = self.rgb_l2(r1)  # 128
        r3 = self.rgb_l3(r2)  # 256
        r4 = self.rgb_l4(r3)  # 512

        # Depth encoder parallel to RGB
        d0 = self.depth_conv1(depth)
        d1 = self.depth_l1(self.depth_maxpool(d0))  # 64
        d2 = self.depth_l2(d1)  # 128
        d3 = self.depth_l3(d2)  # 256
        d4 = self.depth_l4(d3)  # 512

        # ===== MULTI-SCALE TRANSFORMER FUSION =====
        # Fuse at multiple scales
        f1 = self.fusion_l1(r1, d1)  # 64
        f2 = self.fusion_l2(r2, d2)  # 128
        f3 = self.fusion_l3(r3, d3)  # 256
        f4 = self.fusion_l4(r4, d4)  # 512

        # ===== BOTTLENECK GLOBAL TRANSFORMER =====
        B, C, H, W = f4.shape
        bottleneck_tokens = f4.flatten(2).transpose(1, 2)  # [B, N, C]

        # Apply transformer
        enhanced_tokens = self.bottleneck_transformer(bottleneck_tokens)

        # Reshape back
        enhanced_feat = enhanced_tokens.transpose(1, 2).reshape(B, C, H, W)

        # ===== DECODER WITH SKIP CONNECTIONS =====
        # Use fused features at each level for skip connections
        u4 = self.up4(enhanced_feat)
        u4 = torch.cat([u4, f3], dim=1)  # Skip from fused L3

        u3 = self.up3(u4)
        u3 = torch.cat([u3, f2], dim=1)  # Skip from fused L2

        u2 = self.up2(u3)
        u2 = torch.cat([u2, f1], dim=1)  # Skip from fused L1

        u1 = self.up1(u2)
        u0 = self.up0(u1)

        return self.final(u0)
