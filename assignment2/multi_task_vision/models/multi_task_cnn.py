"""
Multi-task model: ResNet backbone + U-Net segmentation head + FPN detection head.
"""

import torch
import math
import torch.nn as nn
import torchvision.models as models
from .layers import ConvBlock, initialize_weights


class UNetDecoder(nn.Module):
    """Simplified U-Net decoder for segmentation."""

    def __init__(self, backbone_channels, num_classes=21, activation="relu", dropout_rate=0.0):
        super().__init__()

        # Decoder channels (match encoder stages)
        # ResNet18/34 channels: c5=512, c4=256, c3=128, c2=64, c1=64 (H/2)
        
        # Stage 1: Up c5 (512) -> 256. Concat c4 (256). Input to conv: 512. Output: 256.
        self.up1 = nn.ConvTranspose2d(backbone_channels, 256, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(256 + 256, 256, activation, dropout_rate)

        # Stage 2: Up (256) -> 128. Concat c3 (128). Input to conv: 256. Output: 128.
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(128 + 128, 128, activation, dropout_rate)

        # Stage 3: Up (128) -> 64. Concat c2 (64). Input to conv: 128. Output: 64.
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64 + 64, 64, activation, dropout_rate)

        # Stage 4: Up (64) -> 64. Concat c1 (64). Input to conv: 128. Output: 32.
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(64 + 64, 32, activation, dropout_rate)

        # Stage 5: Up (32) -> 32. Final resolution.
        self.up5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv5 = ConvBlock(32, 32, activation, dropout_rate)

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features):
        # features is a dict with 'c1', 'c2', 'c3', 'c4', 'c5' from backbone
        x = features['c5']  # 8x8 (512)
        c4 = features['c4'] # 16x16 (256)
        c3 = features['c3'] # 32x32 (128)
        c2 = features['c2'] # 64x64 (64)
        c1 = features['c1'] # 128x128 (64)

        # Up 1
        x = self.up1(x)     # 16x16 (256)
        x = torch.cat([x, c4], dim=1) # 16x16 (512)
        x = self.conv1(x)   # 16x16 (256)

        # Up 2
        x = self.up2(x)     # 32x32 (128)
        x = torch.cat([x, c3], dim=1) # 32x32 (256)
        x = self.conv2(x)   # 32x32 (128)

        # Up 3
        x = self.up3(x)     # 64x64 (64)
        x = torch.cat([x, c2], dim=1) # 64x64 (128)
        x = self.conv3(x)   # 64x64 (64)

        # Up 4
        x = self.up4(x)     # 128x128 (64)
        x = torch.cat([x, c1], dim=1) # 128x128 (128)
        x = self.conv4(x)   # 128x128 (32)

        # Up 5
        x = self.up5(x)     # 256x256 (32)
        x = self.conv5(x)

        seg_output = self.segmentation_head(x)

        return seg_output


class ImprovedDetectionHead(nn.Module):
    """Multi-scale detection with proper anchors"""
    
    def __init__(self, backbone_channels, num_classes=20, activation="relu"):
        super().__init__()
        self.num_classes = num_classes
        
        # Multi-scale anchors (3 scales × 3 aspect ratios = 9 anchors per location)
        self.anchor_scales = [0.1, 0.2, 0.4]  # Small, medium, large
        self.anchor_ratios = [0.5, 1.0, 2.0]  # Tall, square, wide
        
        # Detection at multiple feature map sizes
        self.grid_sizes = [8, 16, 32]  # 64x64, 32x32, 16x16 for 256x256 input
        
        # Shared conv layers
        # We need to project features to same channels first
        self.project_c5 = nn.Conv2d(backbone_channels, 256, kernel_size=1)
        self.project_c4 = nn.Conv2d(backbone_channels // 2, 256, kernel_size=1)
        self.project_c3 = nn.Conv2d(backbone_channels // 4, 256, kernel_size=1)
        
        self.conv1 = ConvBlock(256, 256, activation, 0.0)
        self.conv2 = ConvBlock(256, 256, activation, 0.0)
        
        # Per-anchor predictions (9 anchors per location)
        self.cls_head = nn.Conv2d(256, num_classes * 9, kernel_size=1)
        self.reg_head = nn.Conv2d(256, 4 * 9, kernel_size=1)
        
        # Generate anchors
        self.anchors = self._create_multi_scale_anchors()
    
    def _create_multi_scale_anchors(self):
        """Generate anchors at multiple scales and aspect ratios"""
        all_anchors = []
        
        for grid_size in self.grid_sizes:
            for i in range(grid_size):
                for j in range(grid_size):
                    cx = (j + 0.5) / grid_size
                    cy = (i + 0.5) / grid_size
                    
                    # Generate 9 anchors per grid cell
                    for scale in self.anchor_scales:
                        for ratio in self.anchor_ratios:
                            w = scale * math.sqrt(ratio)
                            h = scale / math.sqrt(ratio)
                            all_anchors.append([cx, cy, w, h])
        
        return torch.tensor(all_anchors).float()  # Shape: [N_anchors, 4]
    
    def forward(self, features):
        # Features from different stages for multi-scale detection
        # c5: 8x8 (for 32px grid)
        # c4: 16x16 (for 16px grid)
        # c3: 32x32 (for 8px grid)
        # Note: grid_sizes in _create_multi_scale_anchors are [8, 16, 32]
        # We need to process them in that order to match anchors
        
        feature_maps = [
            self.project_c5(features['c5']), 
            self.project_c4(features['c4']), 
            self.project_c3(features['c3'])
        ]
        
        all_cls_logits = []
        all_bbox_pred = []
        
        for x in feature_maps:
            # Process features
            x = self.conv1(x)
            x = self.conv2(x)
            
            # Predictions
            cls_logits = self.cls_head(x)  # [B, 20*9, H, W]
            bbox_pred = self.reg_head(x)   # [B, 4*9, H, W]
            
            B, _, H, W = cls_logits.shape
            
            # Reshape: [B, 9, 20, H, W] -> [B, H*W*9, 20]
            cls_logits = cls_logits.view(B, 9, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2).contiguous()
            cls_logits = cls_logits.view(B, -1, self.num_classes)
            
            # Reshape bbox: [B, 9, 4, H, W] -> [B, H*W*9, 4]
            bbox_pred = bbox_pred.view(B, 9, 4, H, W)
            bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).contiguous()
            bbox_pred = bbox_pred.view(B, -1, 4)
            
            all_cls_logits.append(cls_logits)
            all_bbox_pred.append(bbox_pred)
            
        # Concatenate all scales
        cls_logits = torch.cat(all_cls_logits, dim=1) # [B, Total_Anchors, 20]
        bbox_pred = torch.cat(all_bbox_pred, dim=1)   # [B, Total_Anchors, 4]
        
        # Apply sigmoid to bbox (keep in 0-1 range)
        bbox_pred = torch.sigmoid(bbox_pred)
        
        return {
            'cls_logits': cls_logits,
            'bbox_pred': bbox_pred,
            'anchors': self.anchors.unsqueeze(0).repeat(B, 1, 1).to(cls_logits.device)
        }


class SimpleDetectionHead(nn.Module):
    """Simple detection head: predicts class scores and bbox offsets per anchor."""

    def __init__(self, backbone_channels, num_classes=20, activation ="relu"):
        super().__init__()
        self.num_classes = num_classes
        
        # Use 7x7 grid of anchor points for dense predictions
        self.grid_size = 7
        self.num_anchors = self.grid_size * self.grid_size  # 49 anchors

        # Shared feature extraction
        self.conv = ConvBlock(backbone_channels, 256, activation, dropout_rate=0.0)

        # Classification and regression branches (per-anchor predictions)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)  # [B, 20, H, W]
        self.reg_head = nn.Conv2d(256, 4, kernel_size=1)  # [B, 4, H, W]

        # Anchor generation (simplified grid)
        self.anchors = self._create_anchors()

    def _create_anchors(self):
        """Create simple anchor boxes on a grid."""
        grid_size = self.grid_size
        anchors = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Center coordinates (normalized 0-1)
                cx = (j + 0.5) / grid_size
                cy = (i + 0.5) / grid_size
                # Fixed size anchors (normalized)
                anchors.append([cx, cy, 0.14, 0.14])  # 14% of image size
        return torch.tensor(anchors).float()  # [49, 4]

    def forward(self, features):
        x = features['c5']  # [B, C, H, W]
        x = self.conv(x)  # [B, 256, H, W]

        # Spatially adaptive pooling to grid_size x grid_size
        x = nn.AdaptiveAvgPool2d(self.grid_size)(x)  # [B, 256, 7, 7]

        cls_logits = self.cls_head(x)  # [B, 20, 7, 7]
        bbox_pred = self.reg_head(x)  # [B, 4, 7, 7]

        # Reshape to [B, num_anchors, ...]
        B = x.size(0)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous().view(B, self.num_anchors, self.num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(B, self.num_anchors, 4)
        
        # Apply sigmoid to constrain bbox predictions to [0, 1]
        bbox_pred = torch.sigmoid(bbox_pred)

        return {
            'cls_logits': cls_logits,  # [B, 49, 20]
            'bbox_pred': bbox_pred,  # [B, 49, 4]
            'anchors': self.anchors.unsqueeze(0).repeat(B, 1, 1).to(x.device)  # [B, 49, 4]
        }


class FCNHead(nn.Module):
    """Fully Convolutional Network head."""
    
    def __init__(self, backbone_channels, num_classes=21, activation="relu", dropout_rate=0.0):
        super().__init__()
        
        # FCN-32s style (simplified)
        # Reduce channels
        self.conv1 = ConvBlock(backbone_channels, 512, activation, dropout_rate)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Upsample to original size (32x)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32, bias=False)
        
    def forward(self, features):
        x = features['c5']
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


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
        # elif model_cfg["backbone"] == "resnet50":
        else:
            raise ValueError(f"Unknown backbone: {model_cfg['backbone']}")

        # Extract features from different stages
        self.backbone = nn.ModuleDict({
            'c1': nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu), # H/2, 64
            'c2': nn.Sequential(backbone.maxpool, backbone.layer1), # H/4, 64
            'c3': backbone.layer2, # H/8, 128
            'c4': backbone.layer3, # H/16, 256
            'c5': backbone.layer4  # H/32, 512
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
            self.segmentation_head = FCNHead(
                backbone_channels=backbone_channels,
                num_classes=num_classes_seg,
                activation=model_cfg["activation"],
                dropout_rate=model_cfg["dropout_rate"] if config["regularization"]["dropout_enabled"] else 0.0
            )
        else:
            raise ValueError(f"Unsupported segmentation head: {model_cfg['segmentation_head']}")

        # Detection head
        # Detection head
        if model_cfg["detection_head"] == "fpn":
            self.detection_head = ImprovedDetectionHead(
                backbone_channels=backbone_channels,
                num_classes=num_classes_det,
                activation=model_cfg["activation"]
            )
        elif model_cfg["detection_head"] == "simple":
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
        # ResNet structure:
        # x -> conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        
        x = self.backbone['c1'](x)
        features['c1'] = x
        
        x = self.backbone['c2'](x)
        features['c2'] = x
        
        x = self.backbone['c3'](x)
        features['c3'] = x
        
        x = self.backbone['c4'](x)
        features['c4'] = x
        
        x = self.backbone['c5'](x)
        features['c5'] = x

        # Task heads
        seg_output = self.segmentation_head(features)
        det_output = self.detection_head(features)

        return {
            'segmentation': seg_output,
            'detection': det_output
        }