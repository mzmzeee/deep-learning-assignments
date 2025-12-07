"""
Metric computation for segmentation and detection.
Students do not need to modify this file.
"""

import torch
import torch.nn.functional as F
import numpy as np


class SegmentationMetrics:
    """Compute segmentation metrics: Pixel Accuracy, mIoU."""

    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """
        Update confusion matrix.
        pred: [B, C, H, W] logits
        target: [B, H, W] long
        """
        pred = pred.argmax(dim=1).cpu().numpy()
        target = target.cpu().numpy()

        for i in range(pred.shape[0]):
            valid_mask = target[i] != 255  # Ignore void labels
            pred_valid = pred[i][valid_mask].ravel()
            target_valid = target[i][valid_mask].ravel()

            # Vectorized confusion matrix update using np.bincount
            # This is 10-100x faster than the nested loop
            indices = target_valid * self.num_classes + pred_valid
            counts = np.bincount(indices, minlength=self.num_classes * self.num_classes)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """Compute metrics."""
        cm = self.confusion_matrix

        # Pixel accuracy
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-6)

        # IoU per class
        iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-6)
        miou = np.nanmean(iou)

        return {
            "pixel_accuracy": pixel_acc,
            "miou": miou,
            "class_iou": iou.tolist()
        }


class DetectionMetrics:
    """Simplified detection metrics: mAP-like score."""

    def __init__(self, num_classes=20, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []

    def update(self, pred_dict, target_dict):
        """
        Update predictions and targets.
        pred_dict: dict with 'cls_logits', 'bbox_pred'
        target_dict: dict with 'cls_labels', 'bbox_targets'
        """
        cls_pred = torch.sigmoid(pred_dict['cls_logits']).cpu()
        bbox_pred = pred_dict['bbox_pred'].cpu()

        self.all_preds.append({
            'cls': cls_pred,
            'bbox': bbox_pred
        })

        self.all_targets.append({
            'cls': target_dict['cls_labels'].cpu(),
            'bbox': target_dict['bbox_targets'].cpu()
        })

    def compute(self):
        """Compute mAP-like metric (simplified)."""
        if not self.all_preds:
            return {"map": 0.0}

        # Concatenate all predictions
        all_cls_pred = torch.cat([p['cls'] for p in self.all_preds], dim=0)
        all_cls_target = torch.cat([t['cls'] for t in self.all_targets], dim=0)

        # For simplicity: compute AP for each class
        aps = []
        for c in range(self.num_classes):
            pred = all_cls_pred[:, c]
            target = all_cls_target[:, c]

            # Sort by confidence
            sorted_indices = pred.argsort(descending=True)
            pred = pred[sorted_indices]
            target = target[sorted_indices]

            # Compute precision-recall
            tp = (pred > 0.5) & (target == 1)
            fp = (pred > 0.5) & (target == 0)

            tp_cumsum = tp.float().cumsum(0)
            fp_cumsum = fp.float().cumsum(0)

            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / (target.sum() + 1e-6)

            # Compute AP (11-point interpolation)
            ap = 0.0
            for r in torch.arange(0, 1.1, 0.1):
                if torch.sum(recall >= r) == 0:
                    p = 0
                else:
                    p = torch.max(precision[recall >= r])
                ap += p / 11.0

            aps.append(ap.item())

        return {
            "map": np.mean(aps),
            "class_aps": aps
        }


class MultiTaskEvaluator:
    """Combined evaluator for both tasks."""

    def __init__(self):
        self.seg_metrics = SegmentationMetrics(num_classes=21)
        self.det_metrics = DetectionMetrics(num_classes=20)

    def reset(self):
        self.seg_metrics.reset()
        self.det_metrics.reset()

    def update(self, model_output, batch_targets):
        """Update metrics with batch data."""
        # Segmentation update
        seg_pred = model_output['segmentation']
        seg_target = batch_targets[1]
        self.seg_metrics.update(seg_pred, seg_target)

        # Detection update
        det_pred = model_output['detection']
        det_target = batch_targets[2]
        self.det_metrics.update(det_pred, det_target)

    def compute(self):
        """Compute all metrics."""
        seg_results = self.seg_metrics.compute()
        det_results = self.det_metrics.compute()

        return {
            **seg_results,
            **det_results
        }