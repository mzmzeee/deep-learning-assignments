"""
Central configuration for hyperparameter experiments.
"""

CONFIG = {
    # --- System & Logging (DO NOT MODIFY THESE) ---
    "system": {
        "epochs": 50,
        "device": "cuda",  # or "cpu"
        "aim_repo": "./aim_logs",  # Local directory for Aim database
        "aim_experiment": "cv-hyperparam-study",  # Experiment name
        "save_checkpoint": True,
        "seed": 42,
    },

    # --- Phase 1: Network Architecture ---
    "model": {
        "backbone": "resnet18",          # Options: "resnet18", "resnet34"
        "segmentation_head": "unet",     # Options: "unet", "fcn"
        "detection_head": "fpn",         # Options: "fpn", "simple"
        "activation": "relu",            # Options: "relu", "leaky_relu", "gelu"
        "init_scheme": "kaiming",        # Options: "kaiming", "xavier", "normal"
        "init_backbone": True,           # Options: True, False
        "dropout_rate": 0.0,             # Range: 0.0 to 0.5
    },

    # --- Phase 2: Optimization ---
    "training": {
        "lr": 1e-3,                      # Range: 1e-5 to 1e-1
        "batch_size": 16,                # Options: 4, 8, 16, 32
        "weight_decay": 0.0,             # Range: 0.0 to 1e-2 (L2 regularization)
        "loss_weights": {"seg": 1.0, "det": 1.0},  # Multi-task loss balancing

        # Class weights for segmentation (0.1 for background, 1.0 for others)
        "seg_class_weights": [0.1] + [1.0] * 20,

        # Loss function choices
        "seg_loss": "cross_entropy",     # Options: "cross_entropy", "focal"
        "det_loss": "smooth_l1",         # Options: "smooth_l1", "ciou"
    },

    # --- Phase 3: Data Pipeline ---
    "data": {
        "dataset_root": "./data/VOC2012",  # Download Pascal VOC here
        "preprocessing": "none",    # Options: "standardize", "normalize", "none"
        "augmentation_level": "none",      # Options: "none", "basic", "heavy"
    },

    # --- Phase 4: Regularization ---
    "regularization": {
        "dropout_enabled": False,          # Boolean: Enable/disable dropout layers
    }
}