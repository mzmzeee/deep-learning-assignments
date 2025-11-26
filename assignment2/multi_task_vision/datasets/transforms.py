"""
Data preprocessing and augmentation.
Students modify: augmentation strategies and preprocessing methods.
"""

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class SegmentationDetectionTransform:
    """
    Combined transforms for segmentation and detection.
    Handles both image and mask/bbox transforms.
    """

    def __init__(self, config, is_training=True):
        self.config = config
        self.is_training = is_training

        # Preprocessing
        self.preprocess = self._get_preprocess()

        # Augmentation
        self.augment = self._get_augmentation() if is_training else None

        # Resize to fixed size (NEW - fixes batching error)
        self.resize_size = (256, 256)  # You can adjust this size
        self.resize_image = transforms.Resize(
            self.resize_size,
            interpolation=TF.InterpolationMode.BILINEAR
        )
        self.resize_mask = transforms.Resize(
            self.resize_size,
            interpolation=TF.InterpolationMode.NEAREST  # Important for masks!
        )

        # Final tensor conversion
        self.to_tensor = transforms.ToTensor()

    def _get_preprocess(self):
        """Get preprocessing transform."""
        method = self.config["data"]["preprocessing"]

        if method == "standardize":
            # ImageNet statistics
            return transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif method == "normalize":
            # Scale to [0, 1]
            return transforms.Lambda(lambda x: x / 255.0)
        elif method == "none":
            return transforms.Lambda(lambda x: x)
        else:
            raise ValueError(f"Unknown preprocessing: {method}")

    def _get_augmentation(self):
        """Get augmentation pipeline based on config."""
        level = self.config["data"]["augmentation_level"]

        if level == "none":
            return None
        elif level == "basic":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        elif level == "heavy":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            ])
        else:
            raise ValueError(f"Unknown augmentation level: {level}")

    def __call__(self, image, mask=None, boxes=None):
        """Apply transforms to image and optionally mask and boxes.

        ``boxes`` are expected in normalized ``[cx, cy, w, h]`` format. Since
        resizing keeps coordinates normalized, only horizontal flips need to
        update them.
        """
        # Convert to tensors first
        image = self.to_tensor(image)
        if mask is not None:
            mask = torch.from_numpy(mask).long()

        # RESIZE to fixed size (NEW - this fixes the batching error)
        image = self.resize_image(image)
        if mask is not None:
            mask = self.resize_mask(mask.unsqueeze(0)).squeeze(0).long()

        # Apply augmentations (only spatial transforms)
        if self.is_training and self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
                if boxes is not None:
                    boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip cx

        # Apply preprocessing (normalize/standardize)
        image = self.preprocess(image)

        return image, mask, boxes

def get_transform(config, is_training=True):
    """Factory function for transforms."""
    return SegmentationDetectionTransform(config, is_training)