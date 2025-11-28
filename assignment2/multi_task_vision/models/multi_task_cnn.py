"""
Multi-task model: ResNet backbone + U-Net segmentation head + FPN detection head.
Students modify: architecture components, dropout placement, activation functions.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .layers import ConvBlock, initialize_weights


class UNetDecoder(nn.Module):
    """Simplified U-Net decoder for segmentation."""

    def __init__(self, backbone_channels, num_classes=21, activation="relu", dropout_rate=0.0):
        super().__init__()

        # Decoder channels (match encoder stages)
        ch = [256, 128, 64, 32, 32]  # Added one more channel for final upsample

        # Upsampling and conv blocks
        # Each upsample doubles the spatial resolution
        self.up1 = nn.ConvTranspose2d(backbone_channels, ch[0], kernel_size=2, stride=2)
        self.conv1 = ConvBlock(ch[0], ch[0], activation, dropout_rate)

        self.up2 = nn.ConvTranspose2d(ch[0], ch[1], kernel_size=2, stride=2)
        self.conv2 = ConvBlock(ch[1], ch[1], activation, dropout_rate)

        self.up3 = nn.ConvTranspose2d(ch[1], ch[2], kernel_size=2, stride=2)
        self.conv3 = ConvBlock(ch[2], ch[2], activation, dropout_rate)

        self.up4 = nn.ConvTranspose2d(ch[2], ch[3], kernel_size=2, stride=2)
        self.conv4 = ConvBlock(ch[3], ch[3], activation, dropout_rate)

        # Final upsample to match input resolution (256x256)
        self.up5 = nn.ConvTranspose2d(ch[3], ch[4], kernel_size=2, stride=2)
        self.conv5 = ConvBlock(ch[4], ch[4], activation, dropout_rate)

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(ch[4], num_classes, kernel_size=1)

    def forward(self, features):
        # features is a dict with 'c2', 'c3', 'c4', 'c5' from backbone
        x = features['c5']  # Starting from 8x8 if input is 256x256

        x = self.up1(x)  # 16x16
        x = self.conv1(x)

        x = self.up2(x)  # 32x32
        x = self.conv2(x)

        x = self.up3(x)  # 64x64
        x = self.conv3(x)

        x = self.up4(x)  # 128x128
        x = self.conv4(x)

        x = self.up5(x)  # 256x256 - matches input resolution
        x = self.conv5(x)

        seg_output = self.segmentation_head(x)

        return seg_output


class FCNDecoder(nn.Module):
    """Simple FCN decoder for segmentation."""

    def __init__(self, backbone_channels, num_classes=21, activation="relu", dropout_rate=0.0):
        super().__init__()
        self.conv = ConvBlock(backbone_channels, 256, activation, dropout_rate)
        self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features):
        x = features['c5']
        x = self.conv(x)
        x = self.segmentation_head(x)
        # Upsample to match input resolution (assuming 32x downsampling in backbone)
        return nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)


class SimpleDetectionHead(nn.Module):
    """Simple detection head: predicts class scores and bbox offsets."""

    def __init__(self, backbone_channels, num_classes=20, activation="relu"):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature extraction
        self.conv = ConvBlock(backbone_channels, 256, activation, dropout_rate=0.0)

        # Classification and regression branches
        self.cls_head = nn.Conv2d(256, num_classes * 1, kernel_size=1)  # Simple per-pixel classification
        self.reg_head = nn.Conv2d(256, 4 * 1, kernel_size=1)  # Per-pixel bbox regression

        # Anchor generation (simplified grid)
        self.anchors = self._create_anchors()

    def _create_anchors(self):
        """Create simple anchor boxes."""
        # For educational purposes: 5x5 grid of anchors
        grid_size = 5
        anchors = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Center coordinates
                cx = (j + 0.5) / grid_size
                cy = (i + 0.5) / grid_size
                # Fixed size anchors (scaled for 224x224 input)
                anchors.append([cx, cy, 0.2, 0.2])
        return torch.tensor(anchors).float().unsqueeze(0)  # [1, 25, 4]

    def forward(self, features):
        x = features['c5']
        x = self.conv(x)

        # Global pooling for detection
        x = nn.AdaptiveAvgPool2d(1)(x)

        cls_logits = self.cls_head(x).squeeze(-1).squeeze(-1)  # [B, num_classes]
        bbox_pred = self.reg_head(x).squeeze(-1).squeeze(-1)  # [B, 4]

        return {
            'cls_logits': cls_logits,
            'bbox_pred': bbox_pred,
            'anchors': self.anchors.to(x.device)
        }


class MultiTaskCNN(nn.Module):
    """Main multi-task model."""

    def __init__(self, config):
        super().__init__()

        # Extract config
        model_cfg = config["model"]
        num_classes_seg = 21  # Pascal VOC classes + background
        num_classes_det = 20  # Pascal VOC classes

        # Backbone (ResNet)
        if model_cfg["backbone"] == "resnet18":
            backbone = models.resnet18()
            backbone_channels = 512
        elif model_cfg["backbone"] == "resnet34":
            backbone = models.resnet34()
            backbone_channels = 512
        else:
            raise ValueError(f"Unknown backbone: {model_cfg['backbone']}")

        # Extract features from different stages
        self.backbone = nn.ModuleDict({
            'c1': nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
            'c2': backbone.layer1,
            'c3': backbone.layer2,
            'c4': backbone.layer3,
            'c5': backbone.layer4
        })

        # Segmentation head
        if model_cfg["segmentation_head"] == "unet":
            self.segmentation_head = UNetDecoder(
                backbone_channels=backbone_channels,
                num_classes=num_classes_seg,
                activation=model_cfg["activation"],
                dropout_rate=model_cfg["dropout_rate"] if config["regularization"]["dropout_enabled"] else 0.0
            )
        elif model_cfg["segmentation_head"] == "fcn":
            self.segmentation_head = FCNDecoder(
                backbone_channels=backbone_channels,
                num_classes=num_classes_seg,
                activation=model_cfg["activation"],
                dropout_rate=model_cfg["dropout_rate"] if config["regularization"]["dropout_enabled"] else 0.0
            )
        else:
            raise ValueError(f"Unsupported segmentation head: {model_cfg['segmentation_head']}")

        # Detection head
        if model_cfg["detection_head"] in ["fpn", "simple"]:
            # Using our simple detection head for now
            self.detection_head = SimpleDetectionHead(
                backbone_channels=backbone_channels,
                num_classes=num_classes_det,
                activation=model_cfg["activation"]
            )
        else:
            raise ValueError(f"Unsupported detection head: {model_cfg['detection_head']}")

        # Initialize weights
        self._initialize_weights(model_cfg)

    def _initialize_weights(self, model_cfg):
        """Initialize custom layers."""
        if model_cfg["init_backbone"]:
            self.backbone.apply(lambda m: initialize_weights(m, model_cfg["init_scheme"]))
        self.segmentation_head.apply(lambda m: initialize_weights(m, model_cfg["init_scheme"]))
        self.detection_head.apply(lambda m: initialize_weights(m, model_cfg["init_scheme"]))

    def forward(self, x):
        # Extract backbone features
        features = {}
        x = self.backbone['c1'](x)
        features['c2'] = self.backbone['c2'](x)
        features['c3'] = self.backbone['c3'](features['c2'])
        features['c4'] = self.backbone['c4'](features['c3'])
        features['c5'] = self.backbone['c5'](features['c4'])

        # Task heads
        seg_output = self.segmentation_head(features)
        det_output = self.detection_head(features)

        return {
            'segmentation': seg_output,
            'detection': det_output
        }