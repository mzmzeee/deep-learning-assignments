"""
Main training script.
Students run this script with different config modifications.
"""

import torch
import random
import numpy as np
import argparse
from config import CONFIG
from models.multi_task_cnn import MultiTaskCNN
from datasets.voc_loader import get_dataloaders
# from utils.trainer import MultiTaskTrainer  # Replaced with CustomTrainer
from custom_trainer import CustomTrainer as MultiTaskTrainer
from utils.aim_logger import init_aim, finish

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--tags', type=str, nargs='+', default=[])
    args = parser.parse_args()

    run_name = args.run_name
    tags = args.tags if args.tags else []
    if tags:
        CONFIG["system"]["experiment_tags"] = tags

    # Set seed for reproducibility
    set_seed(CONFIG["system"]["seed"])

    init_aim(CONFIG, run_name=run_name)

    # Print configuration
    print("\n" + "="*50)
    print("Starting Multi-Task Training")
    print("="*50)
    print(f"Device: {CONFIG['system']['device']}")
    print(f"Learning Rate: {CONFIG['training']['lr']}")
    print(f"Batch Size: {CONFIG['training']['batch_size']}")
    print(f"Model: {CONFIG['model']['backbone']}")
    print(f"Aim Repo: {CONFIG['system']['aim_repo']}")
    print("="*50 + "\n")

    # Create model
    model = MultiTaskCNN(CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(CONFIG)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create trainer
    trainer = MultiTaskTrainer(model, CONFIG, train_loader, val_loader)

    try:
        # Train
        trainer.train()
        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        # Finish Aim run
        print("Finishing Aim run...")
        finish()

if __name__ == "__main__":
    main()