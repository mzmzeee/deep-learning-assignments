"""
Aim logging utilities.
"""

try:
    from aim import Run, Figure
except ImportError:
    # Mock Aim for environments where it cannot be installed (e.g., Python 3.13)
    print("WARNING: Aim not installed. Using mock logger.")
    
    class Run:
        def __init__(self, repo=None, experiment=None):
            self.name = "mock_run"
            self.repo = repo
            self.experiment = experiment
            self._params = {}
        
        def __setitem__(self, key, value):
            self._params[key] = value
            
        def track(self, metrics, step=None, name=None, epoch=None, context=None):
            pass
            
        def close(self):
            pass

    class Figure:
        def __init__(self, fig):
            pass

import torch
import os
import numpy as np
from datetime import datetime


class AimLogger:
    """Wrapper for Aim experiment tracking."""

    def __init__(self, config, run_name=None):
        """
        Initialize Aim run.
        Use init_aim() instead.
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

        # Only save best checkpoint to save disk space
        if is_best:
            checkpoint_dir = "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config
            }

            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_path)

            # Log best model info as Aim parameters (always works)
            self.run["best_checkpoint/path"] = best_path
            self.run["best_checkpoint/epoch"] = epoch
            self.run["best_checkpoint/miou"] = metrics.get("miou", 0)
            print(f"Saved best checkpoint: {best_path}")

    def finish(self):
        """Close Aim run."""
        self.run.close()

    def log_predictions(self, images, seg_preds, seg_targets, det_preds, det_targets, epoch, num_samples=4):
        """
        Log best and worst predictions as images to Aim.
        Shows random samples with their IoU scores.
        Includes both segmentation masks and detection bounding boxes.
        """
        try:
            from aim import Image
        except ImportError:
            class Image:
                def __init__(self, img):
                    pass

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

        # Create output directory
        os.makedirs("test_results", exist_ok=True)
        
        # Log worst predictions
        for idx in worst_indices:
            # Create visualization image (overlay prediction on original)
            fig = self._create_pred_visualization(
                images[idx], seg_preds[idx], seg_targets[idx], ious[idx],
                det_preds, det_targets, idx
            )
            # Save to disk
            run_prefix = self.run.name if hasattr(self.run, 'name') else "unknown"
            fig.save(f"test_results/{run_prefix}_worst_epoch_{epoch}_sample_{idx}.png")
            
            # Track without context to avoid hashing errors
            self.run.track(
                Image(fig),
                name=f"predictions/worst_sample_{idx}",
                epoch=epoch
            )

        # Log best predictions
        for idx in best_indices:
            fig = self._create_pred_visualization(
                images[idx], seg_preds[idx], seg_targets[idx], ious[idx],
                det_preds, det_targets, idx
            )
            # Save to disk
            run_prefix = self.run.name if hasattr(self.run, 'name') else "unknown"
            fig.save(f"test_results/{run_prefix}_best_epoch_{epoch}_sample_{idx}.png")
            
            # Track without context to avoid hashing errors
            self.run.track(
                Image(fig),
                name=f"predictions/best_sample_{idx}",
                epoch=epoch
            )

    def _create_pred_visualization(self, image, seg_pred, seg_target, iou, det_preds, det_targets, sample_idx):
        """
        Create a matplotlib figure showing image, segmentation prediction/target, and detection boxes.
        Returns a PIL Image for Aim logging.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from PIL import Image

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        img_np = image.permute(1, 2, 0).numpy()

        # Create figure with 4 panels: Original | Seg Pred | Seg GT | Detection
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. Original image
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 2. Segmentation Ground Truth (Swapped with Pred)
        axes[1].imshow(seg_target.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[1].set_title("Seg Ground Truth")
        axes[1].axis('off')

        # 3. Segmentation Prediction (Swapped with GT)
        axes[2].imshow(seg_pred.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[2].set_title(f"Seg Pred (IoU: {iou:.3f})")
        axes[2].axis('off')

        # 4. Detection boxes on original image
        axes[3].imshow(img_np)
        axes[3].set_title("Detection Boxes")
        axes[3].axis('off')

        H, W = img_np.shape[:2]

        # Draw ground truth boxes (GREEN)
        if det_targets is not None:
            objectness = det_targets['objectness'][sample_idx].cpu()  # [49]
            bbox_targets = det_targets['bbox_targets'][sample_idx].cpu()  # [49, 4]
            cls_targets = det_targets['cls_labels'][sample_idx].cpu()  # [49, 20]
            
            # Draw boxes where objectness = 1
            for i in range(len(objectness)):
                if objectness[i] > 0.5:  # Has object
                    box = bbox_targets[i]  # [cx, cy, w, h]
                    cx, cy, w, h = box.tolist()
                    
                    # Convert to pixel coordinates [x, y, w, h]
                    x = (cx - w/2) * W
                    y = (cy - h/2) * H
                    w_px = w * W
                    h_px = h * H
                    
                    # Draw green rectangle
                    rect = patches.Rectangle(
                        (x, y), w_px, h_px,
                        linewidth=2, edgecolor='green', facecolor='none'
                    )
                    axes[3].add_patch(rect)
                    
                    # Get class (find which class has label=1)
                    cls_label = cls_targets[i].argmax().item() if cls_targets[i].sum() > 0 else -1
                    if cls_label >= 0:
                        axes[3].text(
                            x, y, f"GT: {cls_label}", 
                            color='white', fontsize=6,
                            bbox=dict(facecolor='green', alpha=0.7, pad=1)
                        )

        # Draw predicted boxes (RED)
        if det_preds is not None:
            det_cls_logits = det_preds['cls_logits'][sample_idx].cpu()  # [49, 20]
            det_bbox_pred = det_preds['bbox_pred'][sample_idx].cpu()  # [49, 4]
            det_probs = torch.sigmoid(det_cls_logits)  # [49, 20]
            
            # 1. Determine K = number of ground truth objects
            num_gt = 0
            if det_targets is not None:
                objectness = det_targets['objectness'][sample_idx].cpu()
                num_gt = int((objectness > 0.5).sum().item())
            
            # 2. Select Top K predictions
            if num_gt > 0:
                anchor_scores, anchor_classes = det_probs.max(dim=1)  # [49]
                
                # Get top K scores
                # We clamp K to be at most the number of anchors (49)
                k = min(num_gt, len(anchor_scores))
                top_k_scores, top_k_indices = torch.topk(anchor_scores, k=k)
                
                # Filter to keep only these indices
                boxes = det_bbox_pred[top_k_indices].clamp(0, 1)
                scores = top_k_scores
                classes = anchor_classes[top_k_indices]
                
                for box, cls, score in zip(boxes, classes, scores):
                    cx, cy, w, h = box.tolist()
                    
                    # Convert to pixel coordinates
                    x = (cx - w/2) * W
                    y = (cy - h/2) * H
                    w_px = w * W
                    h_px = h * H
                    
                    # Draw red rectangle
                    rect = patches.Rectangle(
                        (x, y), w_px, h_px,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    axes[3].add_patch(rect)
                    
                    # Draw label
                    axes[3].text(
                        x, y - 5, f"{cls.item()}: {score.item():.2f}",
                        color='white', fontsize=6,
                        bbox=dict(facecolor='red', alpha=0.7, pad=1)
                    )
            else:
                # If no GT objects, show no predictions (or maybe just the single best one if you wanted)
                # For now, we strictly follow "same number of boxes", so 0 GT -> 0 Preds.
                pass

        # Force axis limits to prevent expansion
        axes[3].set_xlim(0, W)
        axes[3].set_ylim(H, 0)

        plt.tight_layout()

        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        plt.close(fig)

        return Image.fromarray(img_array)


# Global logger instance (used by trainer)
_logger = None


def init_aim(config, run_name=None):
    """
    Initialize Aim logging.
    Call this at the start of train.py.
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


def log_predictions(images, seg_preds, seg_targets, det_preds, det_targets, epoch, num_samples=4):
    """Convenience function to log predictions using global logger."""
    global _logger
    if _logger:
        _logger.log_predictions(images, seg_preds, seg_targets, det_preds, det_targets, epoch, num_samples)