"""
Training loop logic.
Students do not modify this file.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import logging
logger = logging.getLogger(__name__)
from .evaluator import MultiTaskEvaluator
import utils.aim_logger as aim_logger

class MultiTaskTrainer:
    """Handles multi-task training and validation."""

    def __init__(self, model, config, train_loader, val_loader, checkpoint_path=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader = val_loader
        self.device = config["system"]["device"]
        self.simple_logging = config["system"].get("simple_logging", False)
        self.start_epoch = 1
        self.checkpoint_path = checkpoint_path

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )

        # Mixed Precision Training
        self.use_amp = config["training"].get("use_amp", True)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip = config["training"].get("gradient_clip", 1.0)

        # Gradient accumulation
        self.accumulation_steps = config["training"].get("accumulation_steps", 1)

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = config["training"].get("patience", 10)
        self.patience_counter = 0

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

        pbar = None
        if not self.simple_logging:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", mininterval=5.0, ncols=100, leave=False)
            iterator = pbar
        else:
            iterator = self.train_loader

        for batch_idx, (images, seg_targets, det_targets) in enumerate(iterator):
            images = images.to(self.device, non_blocking=True)
            seg_targets = seg_targets.to(self.device, non_blocking=True)

            # Mixed Precision forward pass
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
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
                
                loss_weights = self.config["training"]["loss_weights"]
                total_batch_loss = (
                    loss_weights["seg"] * seg_loss +
                    loss_weights["det"] * (det_cls_loss + det_bbox_loss)
                )
                total_batch_loss = total_batch_loss / self.accumulation_steps

            # Scaled backward with gradient clipping
            self.scaler.scale(total_batch_loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += total_batch_loss.item() * self.accumulation_steps

            # Update progress string
            if self.simple_logging:
                if batch_idx % 10 == 0:  # Log every 10 batches
                    print(f"Batch {batch_idx}/{num_batches} | Loss: {total_batch_loss.item() * self.accumulation_steps:.4f} | Seg: {seg_loss.item():.4f} | Det: {det_cls_loss.item() + det_bbox_loss.item():.4f}", flush=True)
            else:
                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{total_batch_loss.item() * self.accumulation_steps:.4f}",
                    "Seg": f"{seg_loss.item():.4f}",
                    "Det": f"{det_cls_loss.item() + det_bbox_loss.item():.4f}"
                })

            # Log system stats every 100 batches
            if batch_idx % 100 == 0:
                aim_logger.log_system_stats()

        return total_loss / num_batches

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        self.evaluator.reset()

        with torch.no_grad():
            pbar = None
            if not self.simple_logging:
                pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", mininterval=5.0, ncols=100, leave=False)
                iterator = pbar
            else:
                iterator = self.val_loader

            for images, seg_targets, det_targets in iterator:
                images = images.to(self.device, non_blocking=True)
                seg_targets = seg_targets.to(self.device, non_blocking=True)

                # Mixed Precision forward pass
                with autocast('cuda', enabled=self.use_amp):
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

                if not self.simple_logging:
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

        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)

        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, metrics = self.validate(epoch)

            # Log to Aim
            aim_logger.log_metrics(train_loss, val_loss, metrics, epoch)

            # Save checkpoint
            is_best = metrics["miou"] > self.best_miou
            if is_best:
                self.best_miou = metrics["miou"]

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.warning(f"No improvement for {self.patience_counter}/{self.patience} epochs")

            aim_logger.save_checkpoint(
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

            # Check early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Clean GPU memory
            torch.cuda.empty_cache()

    def load_checkpoint(self, path):
        """Load checkpoint from path."""
        logger.info(f"Loading checkpoint from {path}")
        print(f"Loading checkpoint from {path}")
        # weights_only=False required for numpy scalars/older pytorch versions compat
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load other state if available
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_miou = checkpoint.get('metrics', {}).get('miou', 0.0)
        
        print(f"Resuming from epoch {self.start_epoch}")