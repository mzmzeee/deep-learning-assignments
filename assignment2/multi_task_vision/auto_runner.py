import json
import torch
import traceback
from pathlib import Path
import time
from datetime import datetime
import subprocess
import sys
import shutil
import re
import os

class SafetyValidator:
    """Pre-flight checks and validation"""
    
    def check_environment(self):
        """Check CUDA, GPU, dependencies"""
        print("🔍 Checking environment...")
        if not torch.cuda.is_available():
            print(" CUDA not available! Training will be slow.")
        else:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            
        # Check dataset
        dataset_path = Path("./data/VOC2012")
        if not dataset_path.exists():
            print(" Dataset not found at ./data/VOC2012. Attempting to verify structure...")
            # In a real scenario, we might trigger download here, but we assume it's handled or present
        else:
            print("Dataset directory found.")

    def estimate_memory_usage(self, config):
        """Predict if config will OOM based on backbone and batch size"""
        backbone = config['model']['backbone']
        batch_size = config['training']['batch_size']
        
        # Rough estimates for 6GB VRAM (with AMP enabled, we can go higher)
        # ResNet18 + batch 8 ≈ 2.5GB (with AMP)
        # ResNet34 + batch 8 ≈ 3.5GB (with AMP)
        # ResNet34 + batch 16 ≈ 5GB (with AMP)
        # ResNet34 + batch 32 ≈ 7GB (with AMP)
        
        base_memory = {'resnet18': 2.5, 'resnet34': 3.5}
        estimated = base_memory.get(backbone, 3.0) * (batch_size / 8)
        
        limit = 7.5  # Allow more since AMP reduces memory usage
        
        if estimated > limit:
            return False, estimated
        return True, estimated

class ExperimentRunner:
    """Main training orchestrator"""
    
    def __init__(self, resume=False):
        self.resume = resume
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / "training_progress.json"
        self.validation_report_file = self.results_dir / "validation_report.json"
        self.summary_report_file = self.results_dir / "summary_report.json"
        self.failed_configs_file = self.results_dir / "failed_configs.json"
        
        self.progress = self.load_progress()
        self.validator = SafetyValidator()

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "failed": []}

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def update_config_file(self, config_data):
        """Write new config to config.py"""
        # Backup existing config
        if Path('config.py').exists():
            shutil.copy('config.py', 'config.py.bak')
            
        # Use pprint to format the dict as valid Python code
        import pprint
        config_str = f"CONFIG = {pprint.pformat(config_data, indent=4)}"
        
        with open('config.py', 'w') as f:
            f.write('"""\nCentral configuration for hyperparameter experiments.\n"""\n\n')
            f.write(config_str)

    def restore_config_file(self):
        if Path('config.py.bak').exists():
            shutil.move('config.py.bak', 'config.py')

    def run_phase1_validation(self, configs):
        """5-epoch safety checks for all configs"""
        print("\n" + "="*50)
        print("🚀 Starting Phase 1: Validation (5 Epochs)")
        print("="*50)
        
        validation_results = {"passed": [], "failed": []}
        
        for config_wrapper in configs:
            config = config_wrapper['config']
            config_id = config_wrapper['id']
            name = config_wrapper['name']
            
            if config_id in self.progress['completed'] and self.resume:
                print(f"✓ Skipping completed config {config_id}: {name}")
                validation_results['passed'].append(config_id)
                continue
            
            # Skip already-failed configs when resuming
            failed_ids = [f['id'] for f in self.progress.get('failed', [])]
            if config_id in failed_ids and self.resume:
                print(f"✗ Skipping previously failed config {config_id}: {name}")
                validation_results['failed'].append(config_id)
                continue

            # Memory Check (skip prediction - let actual OOM be caught)
            is_safe, estimated_mem = self.validator.estimate_memory_usage(config)
            if not is_safe:
                print(f"⚠ Skipping Config {config_id} ({name}): Predicted OOM ({estimated_mem:.1f}GB > 7.5GB)")
                self.progress['failed'].append({"id": config_id, "reason": "PREDICTED_OOM"})
                self.save_progress()
                validation_results['failed'].append(config_id)
                continue

            print(f"\n🧪 Validating Config {config_id}/{len(configs)}: {name}")
            print(f"   Backbone: {config['model']['backbone']} | Batch: {config['training']['batch_size']} | LR: {config['training']['lr']}")

            # Set epochs for full training
            config['system']['epochs'] = 60
            
            success, metrics = self.run_single_experiment(config_wrapper, phase="validation")
            
            if success:
                validation_results['passed'].append(config_id)
                # Save to progress so resume works
                self.progress['completed'].append(config_id)
                self.save_progress()
            else:
                validation_results['failed'].append(config_id)
                self.progress['failed'].append({"id": config_id, "reason": metrics.get('error', 'UNKNOWN')})
                self.save_progress()

        # Save validation report
        with open(self.validation_report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
            
        return validation_results

    def run_phase2_training(self, configs, validation_results):
        """50-epoch training for validated configs"""
        print("\n" + "="*50)
        print("🚀 Starting Phase 2: Full Training (50 Epochs)")
        print("="*50)
        
        passed_ids = validation_results['passed']
        
        for config_wrapper in configs:
            config_id = config_wrapper['id']
            
            if config_id not in passed_ids:
                continue
                
            if config_id in self.progress['completed'] and self.resume:
                print(f" Skipping completed config {config_id}")
                continue
                
            print(f"\n Training Config {config_id}: {config_wrapper['name']}")
            
            # Set epochs to 60 for training
            config_wrapper['config']['system']['epochs'] = 60
            
            success, metrics = self.run_single_experiment(config_wrapper, phase="training")
            
            if success:
                self.progress['completed'].append(config_id)
                self.save_progress()
                
                # Copy metrics to results folder
                results_file = Path(f"./results/training_report_{config_id}.json")
                try:
                    with open(results_file, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    print(f" Saved training report to {results_file}")
                except Exception as e:
                    print(f" Failed to save training report: {e}")
            else:
                self.progress['failed'].append({"id": config_id, "reason": metrics.get('error', 'UNKNOWN')})
                self.save_progress()

    def run_single_experiment(self, config_wrapper, phase="validation"):
        """Train one configuration"""
        config = config_wrapper['config']
        name = config_wrapper['name']
        
        # Construct descriptive run name
        model_params = config['model']
        train_params = config['training']
        # e.g. perfect_config_1_resnet34_unet_bs8_lr0.001_validation
        run_name = f"{name}_{model_params['backbone']}_{model_params['segmentation_head']}_bs{train_params['batch_size']}_{phase}"
        
        self.update_config_file(config)
        
        # Helper to find checkpoints
        checkpoints_dir = Path("./checkpoints")
        checkpoint_path = None
        
        # Check if we are resuming an interrupted run
        # Only do this if resume flag is ON and we are in training phase
        if self.resume and phase == "training":
             if checkpoints_dir.exists():
                # Only look for checkpoints belonging to the current run
                cps = list(checkpoints_dir.glob(f"{run_name}*.pth"))
                if cps:
                    print(f"\n Found {len(cps)} existing checkpoints for {run_name}.")
                    print("Do you want to resume from a checkpoint? (y/n)")
                    # Since we are in auto_runner, maybe we shouldn't block?
                    # But the user EXPLICITLY asked: "give me to chose"
                    # We will use input() with a timeout if possible, but standard input() blocks.
                    # Given the user instruction, we should assume interactive mode is fine for this choice.
                    
                    try:
                        choice = input("Resume? [y/n]: ")
                        if choice.lower() == 'y':
                            # List checkpoints
                            print("Available checkpoints:")
                            for i, cp in enumerate(cps):
                                # Try to get mIoU if possible (from filename or just list them)
                                # Filename format isn't strict, so we just list paths
                                # Best would be to load them, but that's slow.
                                # User said "based on mIoU". Our checkpoints are just "best_checkpoint.pth" and "latest_checkpoint.pth"
                                # We can't easily know mIoU without loading.
                                print(f" {i}: {cp.name}")
                            
                            idx_str = input(f"Enter index to resume (0-{len(cps)-1}): ")
                            if idx_str.isdigit() and 0 <= int(idx_str) < len(cps):
                                checkpoint_path = str(cps[int(idx_str)])
                                print(f"Resuming from {checkpoint_path}")
                    except Exception as e:
                        print(f"Skipping resume prompt due to error: {e}")

        # Build command with args instead of stdin
        cmd = ['uv', 'run', 'python', 'train.py', 
               '--run_name', run_name, '--no-tqdm']
        if checkpoint_path:
            cmd.extend(['--checkpoint', checkpoint_path])
            
        if phase and config_wrapper.get('category'):
            cmd.extend(['--tags', phase, config_wrapper['category']])
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) # stdin=subprocess.PIPE not needed now because we use args
            
            # Monitor output
            metrics = {"miou": 0.0, "map": 0.0, "loss": 0.0}
            loss_history = []
            
            for line in iter(process.stdout.readline, ''):
                print(line, end='') # Echo to console
                
                # Safety Checks & Monitoring
                
                # 1. OOM Detection
                if "CUDA out of memory" in line:
                    print("💀 OOM detected! Terminating...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False, {"error": "OOM"}
                
                # 2. Loss Monitoring
                if "Loss:" in line:
                    try:
                        # Regex to capture Loss, Seg, Det values regardless of position
                        loss_match = re.search(r"Loss:\s*([\d\.]+)", line)
                        if loss_match:
                            loss_val = float(loss_match.group(1))
                            metrics["loss"] = loss_val
                            loss_history.append(loss_val)
                            
                            # Exploding Gradient Check
                            if loss_val > 1000 or torch.isnan(torch.tensor(loss_val)):
                                print("💥 Exploding gradients detected! Terminating...")
                                process.terminate()
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                return False, {"error": "EXPLODING_GRADIENTS"}
                    except Exception as e:
                        pass

                # 3. Metrics Monitoring
                if "mIoU:" in line:
                    try:
                        # Regex for mIoU
                        miou_match = re.search(r"mIoU:\s*([\d\.]+)", line)
                        if miou_match:
                            metrics["miou"] = float(miou_match.group(1))
                    except: pass
                if "mAP:" in line:
                    try:
                        # Regex for mAP
                        map_match = re.search(r"mAP:\s*([\d\.]+)", line)
                        if map_match:
                            metrics["map"] = float(map_match.group(1))
                    except: pass

            process.wait()
            
            if process.returncode != 0:
                print(f"Process failed with return code {process.returncode}")
                # return False, {"error": f"Process exited with code {process.returncode}"}
                # If we have gathered metrics, we might want to return them? 
                # But usually code 1 means crash.
                return False, {"error": f"Process exited with code {process.returncode}"}
            
            # Phase 1 Specific Checks
            if phase == "validation":
                # Vanishing Gradient Check (Loss plateau)
                if len(loss_history) > 300: # Assuming enough batches
                    recent_loss = loss_history[-100:]
                    if max(recent_loss) - min(recent_loss) < 1e-4:
                        return False, {"error": "VANISHING_GRADIENTS"}
                
                # Note: Removed BAD_INITIALIZATION check - mIoU < 0.05 is too strict for 2 epochs
                # The model needs more epochs to converge

            return True, metrics

        except Exception as e:
            print(f"❌ Error running experiment: {e}")
            traceback.print_exc()
            return False, {"error": str(e)}
        finally:
            self.restore_config_file()
            torch.cuda.empty_cache()
            import gc
            gc.collect()

def load_configs(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data['configs']

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--phase', choices=['1', '2', 'both'], default='both')
    args = parser.parse_args()
    
    # Safety checks
    validator = SafetyValidator()
    validator.check_environment()
    
    # Load configs
    configs = load_configs('experiment_configs.json')
    
    # Run experiments
    runner = ExperimentRunner(resume=args.resume)
    
    validation_results = {"passed": [], "failed": []}
    
    if args.phase in ['1', 'both']:
        validation_results = runner.run_phase1_validation(configs)
    
    # If skipping phase 1, we need to load validation results or assume all passed (risky)
    # For now, if phase 2 only, we assume we have a report
    if args.phase == '2':
        if runner.validation_report_file.exists():
            with open(runner.validation_report_file, 'r') as f:
                validation_results = json.load(f)
        else:
            print(" No validation report found. Running all configs in Phase 2 (Risky!)")
            validation_results['passed'] = [c['id'] for c in configs]

    if args.phase in ['2', 'both']:
        runner.run_phase2_training(configs, validation_results)
    
    print(" All experiments complete!")

if __name__ == "__main__":
    main()
