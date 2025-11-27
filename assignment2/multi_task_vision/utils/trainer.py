"""
Training loop logic.
Students do not modify this file.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from .evaluator import MultiTaskEvaluator
from .aim_logger import log_metrics, log_system_stats, save_checkpoint

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
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(ignore_index=255)
        elif loss_name == "focal":
            # Simplified focal loss
            return nn.CrossEntropyLoss(ignore_index=255)  # Replace with real focal loss
        else:
            raise ValueError(f"Unknown seg loss: {loss_name}")

    def _get_det_loss(self):
        """Get detection loss function."""
        loss_name = self.config["training"]["det_loss"]
        if loss_name == "smooth_l1":
            return nn.SmoothL1Loss()
        elif loss_name == "ciou":
            return nn.SmoothL1Loss()  # Replace with real CIoU loss
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

            # Detection loss (simplified)
            det_pred = outputs['detection']
            det_cls_loss = nn.BCEWithLogitsLoss()(
                det_pred['cls_logits'],
                det_targets['cls_labels'].to(self.device)
            )
            det_bbox_loss = self.det_loss_fn(
                det_pred['bbox_pred'],
                det_targets['bbox_targets'].to(self.device)
            )

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
                det_cls_loss = nn.BCEWithLogitsLoss()(
                    det_pred['cls_logits'],
                    det_targets['cls_labels'].to(self.device)
                )
                det_bbox_loss = self.det_loss_fn(
                    det_pred['bbox_pred'],
                    det_targets['bbox_targets'].to(self.device)
                )

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

        # Compute final metrics
        metrics = self.evaluator.compute()

        return total_loss / num_batches, metrics

    def train(self):
        """Full training loop."""
        num_epochs = self.config["system"]["epochs"]

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            # Train
            train_loss = self.train_epoch(epoch)

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