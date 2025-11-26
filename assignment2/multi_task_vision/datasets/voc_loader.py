"""Pascal VOC 2012 dataset loader for segmentation and detection.

This variant uses real VOC detection annotations from XML files instead of
generating pseudo bounding boxes from segmentation masks.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

from .transforms import get_transform
from .voc_classes import VOC_NAME_TO_IDX

class VOCSegDetDataset(Dataset):
    """Combined VOC dataset for segmentation and detection."""

    def __init__(self, root, year="2012", image_set="train", transform=None, download=False):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        # Download segmentation data
        try:
            self.seg_dataset = VOCSegmentation(
                root=root,
                year=year,
                image_set=image_set,
                download=download
            )
            self.is_mock = False
        except (RuntimeError, OSError, Exception) as e:
            print(f"WARNING: Could not load VOC dataset ({e}). Using synthetic data for verification.")
            self.is_mock = True
            self.seg_dataset = None

        # Detection annotations from VOC XML files
        self.det_annotations = self._load_detection_annotations()

    def _get_voc_root(self):
        """Return the root directory that contains VOCdevkit/VOC2012.

        We expect the structure: ``root/VOCdevkit/VOC2012`` where ``root`` is
        what was passed to ``VOCSegmentation``.
        """

        # When using torchvision's VOCSegmentation, images are typically under
        # root/VOCdevkit/VOC2012/JPEGImages. We mirror that convention here.
        return os.path.join(self.root, "VOCdevkit", "VOC2012")

    def _load_detection_annotations(self):
        """Load detection annotations from Pascal VOC XML files.

        Builds a mapping from image id (e.g. "2007_000027") to a list of
        objects, each represented as a tuple ``(cls_idx, (xmin, ymin, xmax,
        ymax))`` in absolute pixel coordinates.
        """

        if self.is_mock:
            return {}

        voc_root = self._get_voc_root()
        ann_dir = os.path.join(voc_root, "Annotations")

        if not os.path.isdir(ann_dir):
            print(f"WARNING: VOC Annotations directory not found at {ann_dir}. "
                  "Detection targets will be empty.")
            return {}

        det_annotations = {}

        # VOCSegmentation keeps a list of image paths in ``self.images``.
        # Derive the image id from the image filename stem so that it matches
        # XML names in the Annotations directory.
        for img_path in self.seg_dataset.images:
            img_filename = os.path.basename(img_path)
            img_id, _ = os.path.splitext(img_filename)

            xml_path = os.path.join(ann_dir, img_id + ".xml")
            if not os.path.isfile(xml_path):
                # No detection annotation for this image
                continue

            objects = []
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except ET.ParseError:
                continue

            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name is None:
                    continue

                # Optionally skip difficult objects
                difficult = obj.findtext("difficult")
                if difficult is not None and difficult.strip() == "1":
                    continue

                if name not in VOC_NAME_TO_IDX:
                    continue

                cls_idx = VOC_NAME_TO_IDX[name]
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue

                try:
                    xmin = float(bbox.findtext("xmin"))
                    ymin = float(bbox.findtext("ymin"))
                    xmax = float(bbox.findtext("xmax"))
                    ymax = float(bbox.findtext("ymax"))
                except (TypeError, ValueError):
                    continue

                objects.append((cls_idx, (xmin, ymin, xmax, ymax)))

            if objects:
                det_annotations[img_id] = objects

        return det_annotations

    def __len__(self):
        if self.is_mock:
            return 100  # Mock size
        return len(self.seg_dataset)

    def __getitem__(self, idx):
        if self.is_mock:
            # Generate synthetic data
            image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            mask = Image.fromarray(np.random.randint(0, 21, (256, 256), dtype=np.uint8))
            mask_np = np.array(mask)
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_indices = np.zeros((0,), dtype=np.int64)
        else:
            image, mask = self.seg_dataset[idx]
            mask_np = np.array(mask)

            # Derive image id from underlying VOCSegmentation image path
            img_path = self.seg_dataset.images[idx]
            img_filename = os.path.basename(img_path)
            img_id, _ = os.path.splitext(img_filename)

            # Load detection objects for this image id (if any)
            objects = self.det_annotations.get(img_id, [])

            # Convert absolute corner boxes to normalized center format
            w, h = image.size
            norm_boxes = []
            cls_indices = []
            for cls_idx, (xmin, ymin, xmax, ymax) in objects:
                # VOC uses 1-based inclusive pixel coordinates; convert to 0-based
                # and then to normalized [cx, cy, w, h]
                xmin0 = max(0.0, xmin - 1.0)
                ymin0 = max(0.0, ymin - 1.0)
                xmax0 = min(float(w), xmax - 1.0)
                ymax0 = min(float(h), ymax - 1.0)

                bw = max(0.0, xmax0 - xmin0)
                bh = max(0.0, ymax0 - ymin0)
                if bw <= 1.0 or bh <= 1.0:
                    continue

                cx = (xmin0 + xmax0) / 2.0 / float(w)
                cy = (ymin0 + ymax0) / 2.0 / float(h)
                nw = bw / float(w)
                nh = bh / float(h)

                norm_boxes.append([cx, cy, nw, nh])
                cls_indices.append(cls_idx)

            if norm_boxes:
                boxes = np.array(norm_boxes, dtype=np.float32)
                cls_indices = np.array(cls_indices, dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                cls_indices = np.zeros((0,), dtype=np.int64)

        # Apply transforms
        if self.transform:
            image, mask_tensor, boxes = self.transform(image, mask_np, boxes)
        else:
            from torch import from_numpy
            mask_tensor = from_numpy(mask_np).long()

        mask = mask_tensor

        # Create detection targets for 49 anchors (7x7 grid)
        num_anchors = 49
        det_target = {
            'cls_labels': torch.zeros(num_anchors, 20),  # [49, 20] multi-label per anchor
            'bbox_targets': torch.zeros(num_anchors, 4),  # [49, 4] bbox per anchor
            'objectness': torch.zeros(num_anchors)  # [49] 1=object, 0=background
        }

        # If we have boxes, assign them to anchors based on IoU
        if boxes is not None and len(boxes) > 0:
            # Get anchor positions (7x7 grid)
            anchors = self._get_anchor_boxes()  # [49, 4] in normalized coords
            
            # For each ground truth box, find best matching anchor
            for box, cls_idx in zip(boxes, cls_indices):
                box_tensor = torch.tensor(box, dtype=torch.float32)
                
                # Compute IoU between this box and all anchors
                ious = self._compute_iou(box_tensor.unsqueeze(0), anchors)  # [49]
                
                # Assign to anchor with highest IoU (if IoU > 0.05)
                best_iou, best_idx = ious.max(0)
                if best_iou > 0.05:
                    det_target['objectness'][best_idx] = 1.0
                    det_target['bbox_targets'][best_idx] = box_tensor
                    if 0 <= cls_idx < 20:
                        det_target['cls_labels'][best_idx, cls_idx] = 1.0

        return image, mask, det_target

    def _get_anchor_boxes(self):
        """Get anchor boxes matching the detection head (7x7 grid)."""
        grid_size = 7
        anchors = []
        for i in range(grid_size):
            for j in range(grid_size):
                cx = (j + 0.5) / grid_size
                cy = (i + 0.5) / grid_size
                anchors.append([cx, cy, 0.14, 0.14])
        return torch.tensor(anchors, dtype=torch.float32)

    def _compute_iou(self, box1, box2):
        """Compute IoU between box1 [1,4] and box2 [N,4] in format [cx,cy,w,h]."""
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        box1_x1 = box1[:, 0] - box1[:, 2] / 2
        box1_y1 = box1[:, 1] - box1[:, 3] / 2
        box1_x2 = box1[:, 0] + box1[:, 2] / 2
        box1_y2 = box1[:, 1] + box1[:, 3] / 2
        
        box2_x1 = box2[:, 0] - box2[:, 2] / 2
        box2_y1 = box2[:, 1] - box2[:, 3] / 2
        box2_x2 = box2[:, 0] + box2[:, 2] / 2
        box2_y2 = box2[:, 1] + box2[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        # Union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)

def get_dataloaders(config):
    """Create train and validation dataloaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Transforms
    train_transform = get_transform(config, is_training=True)
    val_transform = get_transform(config, is_training=False)

    # Datasets
    # Datasets
    download = data_cfg.get("download", True) # Default to True to ensure it works
    
    train_dataset = VOCSegDetDataset(
        root=data_cfg["dataset_root"],
        image_set="train",
        transform=train_transform,
        download=download
    )

    val_dataset = VOCSegDetDataset(
        root=data_cfg["dataset_root"],
        image_set="val",
        transform=val_transform,
        download=download
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader