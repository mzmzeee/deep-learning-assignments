
import torch
import gc
import json
import time
import shutil
import argparse
import traceback
import os
from datetime import datetime, timedelta
from pathlib import Path
from experiment_configs import EXPERIMENT_CONFIGS, get_full_config
from train import main

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

class ExperimentRunner:
    def __init__(self):
        self.results = self.load_progress()
    
    def load_progress(self):
        try:
            with open("experiment_results.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"completed": [], "failed": [], "timings": {}}
    
    def cleanup_gpu(self):
        """Aggressive GPU cleanup for RTX 4050"""
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def check_resources(self):
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU: {allocated:.2f}GB / {reserved:.2f}GB")
            if reserved > 5.0:
                print("WARNING: High GPU memory usage, forcing cleanup")
                self.cleanup_gpu()
        
        # Disk space
        os.makedirs("./aim_logs", exist_ok=True)
        stat = shutil.disk_usage("./aim_logs")
        free_gb = stat.free / (1024**3)
        if free_gb < 10:
            raise RuntimeError(f"Low disk: {free_gb:.1f}GB")

    def check_aim_logs_size(self):
        """Check Aim logs aren't consuming too much space"""
        try:
            total_size = sum(
                f.stat().st_size 
                for f in Path("./aim_logs").rglob("*") 
                if f.is_file()
            ) / (1024**3)  # Convert to GB
            
            if total_size > 20:
                print(f"⚠️  WARNING: Aim logs are {total_size:.1f}GB")
            
            return total_size
        except:
            return 0
    
    def validate_config(self, name, config):
        """Validate experiment config before running"""
        errors = []
        
        # Check required sections
        required = ["system", "model", "training", "data", "regularization"]
        for section in required:
            if section not in config:
                errors.append(f"Missing section: {section}")
        
        # Validate value ranges
        if "lr" in config["training"]:
             if config["training"]["lr"] > 0.1 or config["training"]["lr"] < 1e-6:
                errors.append(f"LR out of range: {config['training']['lr']}")
        
        if "batch_size" in config["training"]:
            if config["training"]["batch_size"] not in [4, 8, 16, 32]:
                errors.append(f"Invalid batch size: {config['training']['batch_size']}")
        
        if "dropout_rate" in config["model"]:
            if config["model"]["dropout_rate"] > 0.5:
                errors.append(f"Dropout too high: {config['model']['dropout_rate']}")
        
        if errors:
            raise ValueError(f"Config validation failed for {name}:\n" + "\n".join(errors))
    
    def run_experiments(self, experiment_names=None, dry_run=False):
        self.check_resources()
        
        # Get experiments to run
        all_exps = experiment_names or list(EXPERIMENT_CONFIGS.keys())
        # Filter out completed and failed (failed ones need manual intervention or explicit retry)
        # Actually, for failed ones, we might want to retry if explicitly requested, but default behavior
        # should be to skip what's already in the results file to avoid duplicates.
        # However, if user explicitly passes names, we should probably run them even if failed before?
        # Let's stick to the plan: skip attempted.
        attempted = set(self.results["completed"]) | set(e.split(':')[0] for e in self.results["failed"])
        
        # If specific experiments requested, ignore history check? 
        # No, better to be safe. User can delete from json if they want to re-run.
        to_run = [e for e in all_exps if e not in attempted]
        
        if not to_run:
            print("All requested experiments already completed or attempted!")
            return
        
        print(f"\nRunning {len(to_run)} experiments")
        print(f"Already completed: {len(self.results['completed'])}")
        print(f"Previously failed: {len(self.results['failed'])}\n")
        
        start_time = time.time()
        
        iterator = tqdm(to_run, desc="Experiments") if HAS_TQDM else to_run

        try:
            for i, exp_name in enumerate(iterator, 1):
                if not HAS_TQDM:
                    print(f"\n{'='*60}")
                    print(f"[{i}/{len(to_run)}] {exp_name}")
                    print(f"{'='*60}")
                
                exp_config = get_full_config(EXPERIMENT_CONFIGS[exp_name])
                
                # Dry run override
                if dry_run:
                    exp_config["system"]["epochs"] = 2
                
                try:
                    self.validate_config(exp_name, exp_config)
                    self.check_resources()
                    
                    # Check log size every 5 runs
                    if i % 5 == 0:
                        log_size = self.check_aim_logs_size()
                        if not HAS_TQDM:
                            print(f"Aim logs size: {log_size:.2f}GB")

                    exp_start = time.time()
                    main(experiment_name=exp_name, experiment_config=exp_config)
                    exp_duration = time.time() - exp_start
                    
                    self.results["completed"].append(exp_name)
                    self.results["timings"][exp_name] = exp_duration
                    
                    # ETA calculation
                    avg_time = sum(self.results["timings"].values()) / len(self.results["timings"])
                    remaining_secs = (len(to_run) - i) * avg_time
                    eta = datetime.now() + timedelta(seconds=remaining_secs)
                    
                    if not HAS_TQDM:
                        print(f"\n✓ Completed in {exp_duration/60:.1f}min")
                        print(f"ETA for all: {eta.strftime('%Y-%m-%d %H:%M')}")
                    
                except Exception as e:
                    error = f"{exp_name}: {str(e)}\n{traceback.format_exc()}"
                    self.results["failed"].append(error)
                    if not HAS_TQDM:
                        print(f"\n✗ FAILED: {str(e)}")
                    
                    # Write failure log immediately
                    with open("failed_experiments.txt", "a") as f:
                        f.write(f"\n{'='*60}\n{error}\n")
                
                finally:
                    self.cleanup_gpu()
                    self.save_results()
                    
                    # Close Aim run just in case train.py didn't (it should, but safety first)
                    try:
                        from utils.aim_logger import finish
                        finish()
                    except:
                        pass

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            print(f"Progress saved: {len(self.results['completed'])} completed")
            print("Run again to resume from where you left off")
            self.save_results()
            return
        
        # Final summary
        total_time = (time.time() - start_time) / 3600
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"Total time: {total_time:.2f} hours")
        print(f"Completed: {len(self.results['completed'])}")
        print(f"Failed: {len(self.results['failed'])}")
        print(f"{'='*60}\n")
    
    def save_results(self):
        with open("experiment_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--experiments', nargs='+', default=None)
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    runner.run_experiments(args.experiments, args.dry_run)
