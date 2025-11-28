"""
Aim logging utilities.
STUDENTS DO NOT MODIFY THIS FILE.
"""

from aim import Run, Image
import torch
import os
import numpy as np
from datetime import datetime


class AimLogger:
    """Wrapper for Aim experiment tracking."""

    def __init__(self, config, run_name=None):
        """
        Initialize Aim run.
        Students should not call this directly - use init_aim() instead.
        """
        self.config = config

        # Initialize Aim Run
        self.run = Run(
            repo=config["system"]["aim_repo"],
            experiment=config["system"]["aim_experiment"]
        )

        # Set run name
        if run_name:
            self.run.name = run_name

        # Log hyperparameters
        self.run["hparams"] = config

        # Log system info
        self.run["system/gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            self.run["system/gpu_name"] = torch.cuda.get_device_name(0)
        else:
            self.run["system/gpu_name"] = "CPU"

    def log_metrics(self, train_loss, val_loss, metrics, epoch):
        """
        Log all metrics to Aim.
        metrics: dict with "pixel_accuracy", "miou", "map", etc.
        """
        # Log all metrics at once with step=epoch
        self.run.track({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/pixel_accuracy": metrics["pixel_accuracy"],
            "val/miou": metrics["miou"],
            "val/map": metrics["map"],
        }, step=epoch)

        # Log class-wise metrics as separate series
        if "class_iou" in metrics:
            for i, iou in enumerate(metrics["class_iou"]):
                self.run.track({
                    f"val/iou_class_{i}": iou
                }, step=epoch)

        if "class_aps" in metrics:
            for i, ap in enumerate(metrics["class_aps"]):
                self.run.track({
                    f"val/ap_class_{i}": ap
                }, step=epoch)

    def log_system_stats(self):
        """Log GPU/CPU/memory statistics."""
        if torch.cuda.is_available():
            self.run.track({
                "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            })

    def save_checkpoint(self, model, optimizer, epoch, metrics, config, is_best=False):
        """
        Save model checkpoint locally. Logs metadata to Aim without relying on track_artifact.
        """
        if not config["system"]["save_checkpoint"]:
            return

        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config
        }

        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)

        # Save best checkpoint based on mIoU
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_path)

            # Log best model info as Aim parameters (always works)
            self.run["best_checkpoint/path"] = best_path
            self.run["best_checkpoint/epoch"] = epoch
            self.run["best_checkpoint/miou"] = metrics.get("miou", 0)
            print(f"Saved best checkpoint: {best_path}")

        # Log current checkpoint info
        self.run[f"checkpoints/epoch_{epoch}/path"] = latest_path
        self.run[f"checkpoints/epoch_{epoch}/miou"] = metrics.get("miou", 0)

    def finish(self):
        """Close Aim run."""
        self.run.close()

    def log_predictions(self, images, seg_preds, seg_targets, epoch, num_samples=4):
        """
        Log best and worst predictions as images to Aim.
        Shows random samples with their IoU scores.
        """
        # Convert tensors to numpy
        images = images.cpu()
        seg_preds = seg_preds.argmax(dim=1).cpu()  # [B, H, W]
        seg_targets = seg_targets.cpu()

        # Calculate per-sample IoU to find best/worst
        ious = []
        for i in range(len(images)):
            pred = seg_preds[i]
            target = seg_targets[i]

            # Calculate IoU for this sample (simplified)
            intersection = ((pred == target) & (target != 255)).sum().float()
            union = (target != 255).sum().float()
            iou = (intersection / (union + 1e-6)).item()
            ious.append(iou)

        # Sort by IoU
        sorted_indices = np.argsort(ious)
        worst_indices = sorted_indices[:num_samples]  # Lowest IoU (bad predictions)
        best_indices = sorted_indices[-num_samples:]  # Highest IoU (good predictions)

        # Log worst predictions
        for idx in worst_indices:
            # Create visualization image (overlay prediction on original)
            fig = self._create_pred_visualization(
                images[idx], seg_preds[idx], seg_targets[idx], ious[idx]
            )
            self.run.track(
                Image(fig),
                name="predictions/worst",
                epoch=epoch,
                context={"sample_idx": int(idx)}
            )

        # Log best predictions
        for idx in best_indices:
            fig = self._create_pred_visualization(
                images[idx], seg_preds[idx], seg_targets[idx], ious[idx]
            )
            self.run.track(
                Image(fig),
                name="predictions/best",
                epoch=epoch,
                context={"sample_idx": int(idx)}
            )

    def _create_pred_visualization(self, image, pred, target, iou):
        """
        Create a matplotlib figure showing image, prediction, and target.
        Returns a PIL Image for Aim logging.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import io

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        axes[0].imshow(image.permute(1, 2, 0).numpy())
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Prediction
        axes[1].imshow(pred.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[1].set_title(f"Prediction (IoU: {iou:.3f})")
        axes[1].axis('off')

        # Ground truth
        axes[2].imshow(target.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')

        plt.tight_layout()

        # Convert to PIL Image using BytesIO to avoid backend issues
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img


# Global logger instance (used by trainer)
_logger = None


def init_aim(config, run_name=None):
    """
    Initialize Aim logging.
    Students should call this at the start of train.py.
    """
    global _logger
    _logger = AimLogger(config, run_name)
    return _logger.run


def log_metrics(train_loss, val_loss, metrics, epoch):
    """Convenience function to log metrics using global logger."""
    global _logger
    if _logger is None:
        raise RuntimeError("Aim logger not initialized. Call init_aim() first.")
    _logger.log_metrics(train_loss, val_loss, metrics, epoch)


def log_system_stats():
    """Convenience function to log system stats."""
    global _logger
    if _logger:
        _logger.log_system_stats()


def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """Convenience function to save checkpoint."""
    global _logger
    if _logger:
        _logger.save_checkpoint(model, optimizer, epoch, metrics, config, is_best)


def finish():
    """Convenience function to close Aim run."""
    global _logger
    if _logger:
        _logger.finish()


def log_predictions(images, seg_preds, seg_targets, epoch, num_samples=4):
    """Convenience function to log predictions."""
    global _logger
    if _logger:
        _logger.log_predictions(images, seg_preds, seg_targets, epoch, num_samples)