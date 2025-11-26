"""
Training loop logic.
Students do not modify this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .evaluator import MultiTaskEvaluator
from .aim_logger import log_metrics, log_system_stats, save_checkpoint

import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # pred, target: [B, 4] (x1, y1, x2, y2)
        # Ensure coordinates are correct (x2 > x1, y2 > y1)
        
        # Intersection
        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred[:, 2], target[:, 2])
        y2 = torch.min(pred[:, 3], target[:, 3])
        
        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        # Union
        w1, h1 = pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1]
        w2, h2 = target[:, 2] - target[:, 0], target[:, 3] - target[:, 1]
        union = (w1 * h1) + (w2 * h2) - intersection + 1e-7
        
        iou = intersection / union
        
        # CIoU terms
        # Center distance
        c1_x, c1_y = pred[:, 0] + w1 / 2, pred[:, 1] + h1 / 2
        c2_x, c2_y = target[:, 0] + w2 / 2, target[:, 1] + h2 / 2
        rho2 = (c1_x - c2_x)**2 + (c1_y - c2_y)**2
        
        # Diagonal distance of enclosing box
        cw = torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0])
        ch = torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1])
        c2 = cw**2 + ch**2 + 1e-7
        
        # Aspect ratio consistency
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
            
        ciou = iou - (rho2 / c2) - alpha * v
        loss = 1 - ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class MultiTaskTrainer:
    """Handles multi-task training and validation."""

    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config["system"]["device"]

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Scheduler (StepLR as requested)
        # Default to step every 15 epochs, gamma 0.1 if not specified
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=15, gamma=0.1
        )

        # Loss functions
        self.seg_loss_fn = self._get_seg_loss()
        self.det_loss_fn = self._get_det_loss()

        # Metrics
        self.evaluator = MultiTaskEvaluator()

        # Best metric tracking
        self.best_miou = 0.0

    def _get_seg_loss(self):
        """Get segmentation loss function."""
        loss_name = self.config["training"]["seg_loss"]
        
        # Check for class weights
        weight = None
        if "seg_class_weights" in self.config["training"]:
            weights_list = self.config["training"]["seg_class_weights"]
            weight = torch.tensor(weights_list).to(self.device)
            
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        elif loss_name == "focal":
            # Note: FocalLoss implementation here doesn't support class weights directly in this version
            return FocalLoss(ignore_index=255)
        else:
            raise ValueError(f"Unknown seg loss: {loss_name}")

    def _get_det_loss(self):
        """Get detection loss function."""
        loss_name = self.config["training"]["det_loss"]
        if loss_name == "smooth_l1":
            return nn.SmoothL1Loss()
        elif loss_name == "ciou":
            return CIoULoss()
        else:
            raise ValueError(f"Unknown det loss: {loss_name}")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (images, seg_targets, det_targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            seg_targets = seg_targets.to(self.device, non_blocking=True)


            # Model forward pass
            outputs = self.model(images)

            # Compute losses
            seg_loss = self.seg_loss_fn(outputs['segmentation'], seg_targets)

            # Detection loss (multi-anchor format)
            det_pred = outputs['detection']
            # det_pred: cls_logits [B, 49, 20], bbox_pred [B, 49, 4]
            # det_targets: cls_labels [B, 49, 20], bbox_targets [B, 49, 4], objectness [B, 49]
            
            objectness = det_targets['objectness'].to(self.device)  # [B, 49]
            num_pos = objectness.sum().clamp(min=1.0)  # Avoid division by zero
            
            # Classification loss (only on positive anchors)
            det_cls_loss = nn.BCEWithLogitsLoss(reduction='none')(
                det_pred['cls_logits'],
                det_targets['cls_labels'].to(self.device)
            )  # [B, 49, 20]
            det_cls_loss = (det_cls_loss.mean(dim=-1) * objectness).sum() / num_pos
            
            # Bbox regression loss (only on positive anchors)
            bbox_mask = objectness.unsqueeze(-1).expand_as(det_pred['bbox_pred'])  # [B, 49, 4]
            
            pred_boxes = det_pred['bbox_pred'] * bbox_mask
            target_boxes = det_targets['bbox_targets'].to(self.device) * bbox_mask
            
            # Convert [cx, cy, w, h] to [x1, y1, x2, y2] for CIoU loss
            if isinstance(self.det_loss_fn, CIoULoss):
                # Pred
                p_cx, p_cy, p_w, p_h = pred_boxes.unbind(-1)
                p_x1 = p_cx - 0.5 * p_w
                p_y1 = p_cy - 0.5 * p_h
                p_x2 = p_cx + 0.5 * p_w
                p_y2 = p_cy + 0.5 * p_h
                pred_boxes = torch.stack([p_x1, p_y1, p_x2, p_y2], dim=-1)
                
                # Target
                t_cx, t_cy, t_w, t_h = target_boxes.unbind(-1)
                t_x1 = t_cx - 0.5 * t_w
                t_y1 = t_cy - 0.5 * t_h
                t_x2 = t_cx + 0.5 * t_w
                t_y2 = t_cy + 0.5 * t_h
                target_boxes = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=-1)

            det_bbox_loss = self.det_loss_fn(pred_boxes, target_boxes)

            # Combined loss
            loss_weights = self.config["training"]["loss_weights"]
            total_batch_loss = (
                loss_weights["seg"] * seg_loss +
                loss_weights["det"] * (det_cls_loss + det_bbox_loss)
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{total_batch_loss.item():.4f}",
                "Seg": f"{seg_loss.item():.4f}",
                "Det": f"{det_cls_loss.item() + det_bbox_loss.item():.4f}"
            })

            # Log system stats every 100 batches
            if batch_idx % 100 == 0:
                log_system_stats()

        return total_loss / num_batches

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        self.evaluator.reset()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

            for images, seg_targets, det_targets in pbar:
                images = images.to(self.device, non_blocking=True)
                seg_targets = seg_targets.to(self.device, non_blocking=True)

                # Model forward pass
                outputs = self.model(images)

                # Compute losses
                seg_loss = self.seg_loss_fn(outputs['segmentation'], seg_targets)

                det_pred = outputs['detection']
                objectness = det_targets['objectness'].to(self.device)
                num_pos = objectness.sum().clamp(min=1.0)
                
                det_cls_loss = nn.BCEWithLogitsLoss(reduction='none')(
                    det_pred['cls_logits'],
                    det_targets['cls_labels'].to(self.device)
                )
                det_cls_loss = (det_cls_loss.mean(dim=-1) * objectness).sum() / num_pos
                
                bbox_mask = objectness.unsqueeze(-1).expand_as(det_pred['bbox_pred'])
                
                pred_boxes = det_pred['bbox_pred'] * bbox_mask
                target_boxes = det_targets['bbox_targets'].to(self.device) * bbox_mask
                
                # Convert [cx, cy, w, h] to [x1, y1, x2, y2] for CIoU loss
                if isinstance(self.det_loss_fn, CIoULoss):
                    # Pred
                    p_cx, p_cy, p_w, p_h = pred_boxes.unbind(-1)
                    p_x1 = p_cx - 0.5 * p_w
                    p_y1 = p_cy - 0.5 * p_h
                    p_x2 = p_cx + 0.5 * p_w
                    p_y2 = p_cy + 0.5 * p_h
                    pred_boxes = torch.stack([p_x1, p_y1, p_x2, p_y2], dim=-1)
                    
                    # Target
                    t_cx, t_cy, t_w, t_h = target_boxes.unbind(-1)
                    t_x1 = t_cx - 0.5 * t_w
                    t_y1 = t_cy - 0.5 * t_h
                    t_x2 = t_cx + 0.5 * t_w
                    t_y2 = t_cy + 0.5 * t_h
                    target_boxes = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=-1)

                det_bbox_loss = self.det_loss_fn(pred_boxes, target_boxes)

                # Combined loss
                loss_weights = self.config["training"]["loss_weights"]
                total_batch_loss = (
                    loss_weights["seg"] * seg_loss +
                    loss_weights["det"] * (det_cls_loss + det_bbox_loss)
                )

                total_loss += total_batch_loss.item()

                # Update metrics
                self.evaluator.update(outputs, (images, seg_targets, det_targets))

                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{total_batch_loss.item():.4f}",
                    "Seg": f"{seg_loss.item():.4f}",
                    "Det": f"{det_cls_loss.item() + det_bbox_loss.item():.4f}"
                })

            # Log predictions (using last batch)
            # We use the convenience function from aim_logger
            from .aim_logger import log_predictions
            log_predictions(
                images, 
                outputs['segmentation'], 
                seg_targets,
                outputs['detection'],
                det_targets,
                epoch, 
                num_samples=4
            )

        # Compute final metrics
        metrics = self.evaluator.compute()

        return total_loss / num_batches, metrics

    def train(self):
        """Full training loop."""
        num_epochs = self.config["system"]["epochs"]

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*50}")

            # Train
            train_loss = self.train_epoch(epoch)
            
            # Step scheduler
            self.scheduler.step()

            # Validate
            val_loss, metrics = self.validate(epoch)

            # Log to Aim
            log_metrics(train_loss, val_loss, metrics, epoch)

            # Save checkpoint
            is_best = metrics["miou"] > self.best_miou
            if is_best:
                self.best_miou = metrics["miou"]

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                metrics,
                self.config,
                is_best=is_best
            )

            # Print summary
            print(f"\nValidation Summary:")
            print(f"  mIoU: {metrics['miou']:.4f} (Best: {self.best_miou:.4f})")
            print(f"  Pixel Acc: {metrics['pixel_accuracy']:.4f}")
            print(f"  mAP: {metrics['map']:.4f}")