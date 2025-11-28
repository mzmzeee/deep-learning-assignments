from utils.trainer import MultiTaskTrainer
from utils.aim_logger import log_predictions
import torch

class CustomTrainer(MultiTaskTrainer):
    """
    Custom trainer that logs predictions to Aim during validation.
    """
    def validate(self, epoch):
        # Run standard validation
        val_loss, metrics = super().validate(epoch)
        
        # Log predictions for a few samples
        try:
            # Get a batch from validation loader
            # We create a new iterator to ensure we get data without messing up the loader state if it was used elsewhere
            # But val_loader is an iterable, so iter(val_loader) is fine.
            images, seg_targets, det_targets = next(iter(self.val_loader))
            
            images = images.to(self.device)
            seg_targets = seg_targets.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)
                seg_preds = outputs['segmentation']
                
                # Log to Aim
                log_predictions(images, seg_preds, seg_targets, epoch)
                
        except Exception as e:
            print(f"Failed to log prediction images: {e}")
            
        return val_loss, metrics
