#!/usr/bin/env python3
"""
Inference and Visualization Script for Multi-Task Model.

Usage:
    # Visualize on random validation samples (with ground truth)
    python inference.py --checkpoint ./checkpoints/best_checkpoint.pth --data ./data/VOC2012 --mode dataset --num_samples 5

    # Visualize a single image (no ground truth)
    python inference.py --checkpoint ./checkpoints/best_checkpoint.pth --image path/to/your/image.jpg --mode single

    # Visualize with custom save path
    python inference.py --checkpoint ./checkpoints/best_checkpoint.pth --data ./data/VOC2012 --mode dataset --output ./results
"""

import argparse
import os
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from torch.distributions import transforms

from config import CONFIG
from models.multi_task_cnn import MultiTaskCNN
from datasets.voc_loader import VOCSegDetDataset
from datasets.transforms import get_transform


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create model
    model = MultiTaskCNN(config)
    model.to(device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best mIoU: {checkpoint['metrics']['miou']:.4f}")

    return model


def denormalize_image(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    return image.permute(1, 2, 0).numpy()


def create_overlay(image, mask, alpha=0.6):
    """Create overlay of segmentation mask on image."""
    from matplotlib.colors import ListedColormap

    # Convert mask to RGB
    cmap = plt.get_cmap('tab20')
    mask_colored = cmap(mask.cpu().numpy())[:, :, :3]

    # Blend
    overlay = (1 - alpha) * image + alpha * mask_colored
    return np.clip(overlay, 0, 1)


def visualize_predictions(images, seg_preds, det_preds, seg_targets=None, save_path=None):
    """
    Create visualization comparing predictions vs ground truth.
    Applies softmax to segmentation outputs for probability visualization.
    Returns a matplotlib figure.
    """
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))  # Added column for softmax

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    # Apply softmax to get probabilities
    seg_probs = torch.softmax(seg_preds, dim=1)

    for i in range(batch_size):
        # Denormalize image
        img_np = denormalize_image(images[i])

        # Get predictions
        seg_pred = seg_probs[i].argmax(dim=0).cpu()  # Argmax on probabilities
        seg_confidence = seg_probs[i].max(dim=0)[0].cpu()  # Confidence values

        # Get detection predictions for this specific sample
        det_cls_logits = det_preds['cls_logits'][i].cpu()
        det_bbox_pred = det_preds['bbox_pred'][i].cpu()
        det_probs = torch.sigmoid(det_cls_logits)

        # Original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # Segmentation prediction (argmax)
        seg_overlay = create_overlay(img_np, seg_pred)
        axes[i, 1].imshow(seg_overlay)
        axes[i, 1].set_title("Seg Prediction")
        axes[i, 1].axis('off')

        # Segmentation confidence map (NEW - shows softmax probabilities)
        conf_im = axes[i, 2].imshow(seg_confidence.numpy(), cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Seg Confidence\n(mean: {seg_confidence.mean():.3f})")
        axes[i, 2].axis('off')
        plt.colorbar(conf_im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Detection prediction
        det_class = det_probs.argmax().item()
        det_conf = det_probs.max().item()
        det_text = f"Det: cls={det_class}\nconf={det_conf:.2f}"
        axes[i, 3].imshow(img_np)
        axes[i, 3].set_title(det_text, fontsize=8)
        axes[i, 3].axis('off')

        # Ground truth or blank
        if seg_targets is not None:
            seg_target = seg_targets[i].cpu()
            target_overlay = create_overlay(img_np, seg_target)
            axes[i, 4].imshow(target_overlay)
            axes[i, 4].set_title("Ground Truth")
        else:
            axes[i, 4].imshow(np.ones_like(img_np) * 0.5)
            axes[i, 4].set_title("No Ground Truth")
        axes[i, 4].axis('off')

    plt.tight_layout()
    return fig


def process_dataset(model, data_root, config, device, num_samples=5, output_dir="./results"):
    """Process random samples from dataset with ground truth."""
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    transform = get_transform(config, is_training=False)
    dataset = VOCSegDetDataset(
        root=data_root,
        image_set="val",
        transform=transform,
        download=False
    )

    print(f"\nProcessing {num_samples} random samples from dataset...")

    # Randomly select samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        # Load data
        image, seg_target, det_target = dataset[idx]

        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_batch)
            seg_pred = outputs['segmentation']
            det_pred = outputs['detection']

        # Visualize
        fig = visualize_predictions(
            image_batch.cpu(),
            seg_pred.cpu(),
            det_pred,
            seg_target.unsqueeze(0),
            save_path=os.path.join(output_dir, f"sample_{idx}.png")
        )

        # Save figure
        fig_path = os.path.join(output_dir, f"sample_{idx}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {fig_path}")
        plt.close(fig)


def process_single_image(model, image_path, config, device, output_dir="./results"):
    """Process a single image without ground truth."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing single image: {image_path}")

    # Load and transform image
    image_pil = Image.open(image_path).convert('RGB')

    # Use validation transform (no augmentations)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform(image_pil)
    image_batch = image.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_batch)
        seg_pred = outputs['segmentation']
        det_pred = outputs['detection']

    # Create dummy targets for visualization function
    dummy_targets = torch.zeros_like(seg_pred.argmax(dim=1))

    # Visualize
    fig = visualize_predictions(
        image_batch.cpu(),
        seg_pred.cpu(),
        det_pred,
        seg_targets=dummy_targets,  # Won't be displayed
        save_path=os.path.join(output_dir, "single_image.png")
    )

    # Save figure
    fig_path = os.path.join(output_dir, "single_image.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {fig_path}")
    plt.show()  # Also display interactively
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Model Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["dataset", "single"], required=True,
        help="Mode: 'dataset' for val set samples, 'single' for one image"
    )
    parser.add_argument(
        "--data", type=str, default="./data/VOC2012",
        help="Path to VOC2012 dataset (for dataset mode)"
    )
    parser.add_argument(
        "--image", type=str,
        help="Path to single image (for single mode)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of random samples to visualize (dataset mode)"
    )
    parser.add_argument(
        "--output", type=str, default="./results",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu"
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, CONFIG, device)

    # Run inference
    if args.mode == "dataset":
        process_dataset(
            model, args.data, CONFIG, device,
            num_samples=args.num_samples,
            output_dir=args.output
        )
    elif args.mode == "single":
        if not args.image:
            raise ValueError("--image required for single mode")
        process_single_image(
            model, args.image, CONFIG, device,
            output_dir=args.output
        )

    print(f"\n✓ Done! Results saved to {args.output}")


if __name__ == "__main__":
    main()