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
    parser.add_argument('--no-tqdm', action='store_true', help='Disable tqdm progress bars and use simple logging')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    run_name = args.run_name
    tags = args.tags if args.tags else []
    if tags:
        CONFIG["system"]["experiment_tags"] = tags

    # Set seed for reproducibility
    set_seed(CONFIG["system"]["seed"])

    # Monkeypatch save_checkpoint to prevent overwriting
    import utils.aim_logger as aim_logger_module
    original_save = aim_logger_module.save_checkpoint
    
    def unique_save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
        # We can't easily change the implementation inside aim_logger without editing it
        # But we can change the 'checkpoint_dir' logic if we could... 
        # Actually, aim_logger.save_checkpoint uses hardcoded "./checkpoints".
        # We can't change the path inside the function easily.
        # But we CAN re-implement the function here since it's simple.
        
        if not config["system"]["save_checkpoint"]:
            return

        import os
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Use run_name from args (it's in CONFIG now? No, CONFIG is static dict, but we can access `run_name` variable from outer scope if we are careful, or better: passed config doesn't have run_name usually?
        # unique_save_checkpoint is called by trainer. Since we are monkeypatching, we capture the global `run_name` from main() scope? No, `main` variables aren't global.
        # But we passed `run_name` to `init_aim`. `aim_logger._logger.run.name` exists!
        
        current_run_name = "experiment"
        if aim_logger_module._logger and aim_logger_module._logger.run:
            current_run_name = aim_logger_module._logger.run.name
            
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config
        }
        
        # Save latest
        latest_filename = f"{current_run_name}_latest.pth"
        latest_path = os.path.join(checkpoint_dir, latest_filename)
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_filename = f"{current_run_name}_best.pth"
            best_path = os.path.join(checkpoint_dir, best_filename)
            torch.save(checkpoint, best_path)
            
            # We also update the logger manually since we bypassed the original method
            if aim_logger_module._logger:
                aim_logger_module._logger.run["best_checkpoint/path"] = best_path
                aim_logger_module._logger.run["best_checkpoint/epoch"] = epoch
                aim_logger_module._logger.run["best_checkpoint/miou"] = metrics.get("miou", 0)
                print(f"Saved best checkpoint: {best_path}")

        if aim_logger_module._logger:
             aim_logger_module._logger.run[f"checkpoints/epoch_{epoch}/path"] = latest_path

    # Apply patch
    aim_logger_module.save_checkpoint = unique_save_checkpoint
    aim_logger_module._logger = None # Force re-init if needed, though init_aim called later

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
    CONFIG['system']['simple_logging'] = args.no_tqdm
    trainer = MultiTaskTrainer(model, CONFIG, train_loader, val_loader, checkpoint_path=args.checkpoint)

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