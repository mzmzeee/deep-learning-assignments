"""
Pascal VOC 2012 dataset loader for segmentation and detection.
Downloads and loads data automatically.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import VOCSegmentation
import numpy as np
from PIL import Image
from .transforms import get_transform

class VOCSegDetDataset(Dataset):
    """Combined VOC dataset for segmentation and detection."""

    def __init__(self, root, year="2012", image_set="train", transform=None, download=False):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        # Download segmentation data
        self.seg_dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=download
        )

        # Detection annotations (simplified)
        self.det_annotations = self._load_detection_annotations()

    def _load_detection_annotations(self):
        """Load simplified detection annotations."""
        # For educational purposes, we'll use segmentation masks to generate pseudo-bboxes
        # In practice, use VOC detection annotations
        return {}

    def __len__(self):
        return len(self.seg_dataset)

    def __getitem__(self, idx):
        image, mask = self.seg_dataset[idx]

        # Convert mask to tensor
        mask = np.array(mask)

        # Generate pseudo detection boxes from mask instances
        # (In real assignment, load actual VOC detection annotations)
        boxes = self._generate_pseudo_boxes(mask)

        # Apply transforms
        if self.transform:
            image, mask, boxes = self.transform(image, mask, boxes)

        # For detection, we'll use a simple format
        # cls_labels: [num_classes], bbox_targets: [4]
        det_target = {
            'cls_labels': torch.zeros(20),  # 20 VOC classes
            'bbox_targets': torch.zeros(4)  # Single bbox for simplicity
        }

        # If we have boxes, use the first one
        if len(boxes) > 0:
            # Simple mapping: if mask contains object class > 0, mark presence
            unique_classes = np.unique(mask.numpy() if isinstance(mask, torch.Tensor) else mask)
            for cls_id in unique_classes:
                if cls_id > 0 and cls_id <= 20:
                    det_target['cls_labels'][cls_id - 1] = 1  # Binary presence

            det_target['bbox_targets'] = torch.tensor(boxes[0], dtype=torch.float32)

        return image, mask, det_target

    def _generate_pseudo_boxes(self, mask):
        """Generate pseudo bounding boxes from segmentation mask."""
        # Find connected components (simplified)
        boxes = []
        for cls_id in np.unique(mask):
            if cls_id == 0:  # Background
                continue
            # Get coordinates for this class
            coords = np.where(mask == cls_id)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                # Normalize to [0, 1]
                h, w = mask.shape
                boxes.append([
                    x_min / w, y_min / h, x_max / w, y_max / h
                ])
                break  # Only take first object for simplicity

        return np.array(boxes) if boxes else np.zeros((0, 4))


def get_dataloaders(config):
    """Create train and validation dataloaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Transforms
    train_transform = get_transform(config, is_training=True)
    val_transform = get_transform(config, is_training=False)

    # Datasets
    train_dataset = VOCSegDetDataset(
        root=data_cfg["dataset_root"],
        image_set="train",
        transform=train_transform,
        download=False
    )

    val_dataset = VOCSegDetDataset(
        root=data_cfg["dataset_root"],
        image_set="val",
        transform=val_transform,
        download=False
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader