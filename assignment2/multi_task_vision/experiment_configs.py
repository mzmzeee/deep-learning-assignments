
import copy
from config import CONFIG

def get_full_config(exp_config):
    """
    Deep merge experiment config with base config.
    """
    full_config = copy.deepcopy(CONFIG)
    
    for section, settings in exp_config.items():
        if section not in full_config:
            full_config[section] = {}
        for key, value in settings.items():
            full_config[section][key] = value
            
    return full_config

EXPERIMENT_CONFIGS = {}

# =============================================================================
# 1. BASELINE (5 experiments)
# Pure defaults and minor variations of the starting point.
# =============================================================================
EXPERIMENT_CONFIGS.update({
    "baseline_01_default": {},  # Pure default
    "baseline_02_seed_1": {"system": {"seed": 1}},
    "baseline_03_seed_2": {"system": {"seed": 2}},
    "baseline_04_seed_3": {"system": {"seed": 3}},
    "baseline_05_seed_4": {"system": {"seed": 4}},
})

# =============================================================================
# 2. OPTIMAL (15 experiments)
# Best practices: ResNet34, Heavy Aug, Correct Pairings, etc.
# =============================================================================
EXPERIMENT_CONFIGS.update({
    "optimal_01_golden_standard": {
        "model": {"backbone": "resnet34", "activation": "relu", "init_scheme": "kaiming", "init_backbone": True, "dropout_rate": 0.3},
        "training": {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-4, "seg_loss": "focal", "det_loss": "ciou"},
        "data": {"preprocessing": "standardize", "augmentation_level": "heavy"},
        "regularization": {"dropout_enabled": True}
    },
    "optimal_02_gelu_xavier": {
        "model": {"backbone": "resnet34", "activation": "gelu", "init_scheme": "xavier", "init_backbone": True, "dropout_rate": 0.3},
        "training": {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-4},
        "data": {"preprocessing": "standardize", "augmentation_level": "heavy"}
    },
    "optimal_03_leaky_relu_kaiming": {
        "model": {"backbone": "resnet18", "activation": "leaky_relu", "init_scheme": "kaiming", "init_backbone": True, "dropout_rate": 0.2},
        "training": {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-4},
        "data": {"preprocessing": "standardize", "augmentation_level": "basic"}
    },
    "optimal_04_resnet34_fcn_fpn": {
        "model": {"backbone": "resnet34", "segmentation_head": "fcn", "detection_head": "fpn", "init_backbone": True},
        "training": {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-4},
        "data": {"augmentation_level": "heavy"}
    },
    "optimal_05_heavy_reg": {
        "model": {"backbone": "resnet34", "dropout_rate": 0.5},
        "training": {"weight_decay": 1e-3},
        "regularization": {"dropout_enabled": True},
        "data": {"augmentation_level": "heavy"}
    },
    "optimal_06_focal_ciou_balanced": {
        "training": {"seg_loss": "focal", "det_loss": "ciou", "loss_weights": {"seg": 1.0, "det": 1.0}},
        "model": {"init_backbone": True}
    },
    "optimal_07_batch32_lr_scaled": {
        "training": {"batch_size": 32, "lr": 2e-3, "weight_decay": 1e-4}, # Linear scaling rule
        "model": {"backbone": "resnet18", "init_backbone": True}
    },
    "optimal_08_standardize_only": {
        "data": {"preprocessing": "standardize", "augmentation_level": "basic"},
        "model": {"init_backbone": True}
    },
    "optimal_09_normalize_heavy": {
        "data": {"preprocessing": "normalize", "augmentation_level": "heavy"},
        "model": {"init_backbone": True}
    },
    "optimal_10_resnet34_basic": {
        "model": {"backbone": "resnet34", "init_backbone": True},
        "data": {"augmentation_level": "basic"}
    },
    "optimal_11_leaky_relu_dropout": {
        "model": {"activation": "leaky_relu", "dropout_rate": 0.3},
        "regularization": {"dropout_enabled": True},
        "training": {"weight_decay": 1e-4}
    },
    "optimal_12_gelu_kaiming_ok": { # GELU works ok with Kaiming too sometimes
        "model": {"activation": "gelu", "init_scheme": "kaiming", "init_backbone": True},
        "training": {"lr": 1e-3}
    },
    "optimal_13_seg_focused": {
        "training": {"loss_weights": {"seg": 2.0, "det": 1.0}, "seg_loss": "focal"},
        "model": {"init_backbone": True}
    },
    "optimal_14_det_focused": {
        "training": {"loss_weights": {"seg": 1.0, "det": 2.0}, "det_loss": "ciou"},
        "model": {"init_backbone": True}
    },
    "optimal_15_conservative_best": {
        "model": {"backbone": "resnet18", "init_backbone": True},
        "training": {"lr": 5e-4, "weight_decay": 1e-4},
        "data": {"augmentation_level": "basic"}
    }
})

# =============================================================================
# 3. REASONABLE (10 experiments)
# Acceptable choices, moderate performance expected.
# =============================================================================
EXPERIMENT_CONFIGS.update({
    "reasonable_01_resnet18_basic": {
        "model": {"backbone": "resnet18", "init_backbone": True},
        "data": {"augmentation_level": "basic"}
    },
    "reasonable_02_lr_low": {
        "training": {"lr": 1e-4},
        "model": {"init_backbone": True}
    },
    "reasonable_03_batch8": {
        "training": {"batch_size": 8},
        "model": {"init_backbone": True}
    },
    "reasonable_04_dropout_02": {
        "model": {"dropout_rate": 0.2},
        "regularization": {"dropout_enabled": True}
    },
    "reasonable_05_smooth_l1_ce": {
        "training": {"seg_loss": "cross_entropy", "det_loss": "smooth_l1"}
    },
    "reasonable_06_normalize": {
        "data": {"preprocessing": "normalize"}
    },
    "reasonable_07_xavier_relu": { # Not ideal but common enough
        "model": {"activation": "relu", "init_scheme": "xavier"}
    },
    "reasonable_08_fcn_simple": {
        "model": {"segmentation_head": "fcn", "detection_head": "simple"}
    },
    "reasonable_09_resnet34_no_aug": {
        "model": {"backbone": "resnet34", "init_backbone": True},
        "data": {"augmentation_level": "none"}
    },
    "reasonable_10_balanced_loss": {
        "training": {"loss_weights": {"seg": 1.0, "det": 1.0}}
    }
})

# =============================================================================
# 4. MISMATCHED (15 experiments)
# Deliberately bad pairings.
# =============================================================================
EXPERIMENT_CONFIGS.update({
    "mismatch_01_relu_xavier": {
        "model": {"activation": "relu", "init_scheme": "xavier"}
    },
    "mismatch_02_gelu_kaiming": {
        "model": {"activation": "gelu", "init_scheme": "kaiming"}
    },
    "mismatch_03_no_aug_high_dropout": {
        "data": {"augmentation_level": "none"},
        "model": {"dropout_rate": 0.5},
        "regularization": {"dropout_enabled": True}
    },
    "mismatch_04_heavy_aug_no_dropout": {
        "data": {"augmentation_level": "heavy"},
        "regularization": {"dropout_enabled": False}
    },
    "mismatch_05_high_lr_no_decay": {
        "training": {"lr": 1e-2, "weight_decay": 0.0}
    },
    "mismatch_06_tiny_batch_heavy_aug": {
        "training": {"batch_size": 4},
        "data": {"augmentation_level": "heavy"}
    },
    "mismatch_07_resnet34_scratch": {
        "model": {"backbone": "resnet34", "init_backbone": False}
    },
    "mismatch_08_normal_init_scratch": {
        "model": {"init_scheme": "normal", "init_backbone": False}
    },
    "mismatch_09_unbalanced_seg": {
        "training": {"loss_weights": {"seg": 5.0, "det": 0.2}}
    },
    "mismatch_10_unbalanced_det": {
        "training": {"loss_weights": {"seg": 0.2, "det": 5.0}}
    },
    "mismatch_11_leaky_relu_xavier": {
        "model": {"activation": "leaky_relu", "init_scheme": "xavier"}
    },
    "mismatch_12_fcn_fpn_scratch": {
        "model": {"segmentation_head": "fcn", "detection_head": "fpn", "init_backbone": False}
    },
    "mismatch_13_batch32_low_lr": {
        "training": {"batch_size": 32, "lr": 1e-5} # Too slow updates
    },
    "mismatch_14_batch4_high_lr": {
        "training": {"batch_size": 4, "lr": 1e-2} # Unstable
    },
    "mismatch_15_heavy_aug_scratch": {
        "data": {"augmentation_level": "heavy"},
        "model": {"init_backbone": False} # Hard to learn from scratch with heavy aug
    }
})

# =============================================================================
# 5. EXTREME (10 experiments)
# Stress test boundaries.
# =============================================================================
EXPERIMENT_CONFIGS.update({
    "extreme_01_lr_tiny": {"training": {"lr": 1e-5}},
    "extreme_02_lr_huge": {"training": {"lr": 1e-1}},
    "extreme_03_batch_4": {"training": {"batch_size": 4}},
    "extreme_04_batch_32": {"training": {"batch_size": 32}},
    "extreme_05_dropout_05": {"model": {"dropout_rate": 0.5}, "regularization": {"dropout_enabled": True}},
    "extreme_06_weight_decay_huge": {"training": {"weight_decay": 1e-2}},
    "extreme_07_seg_weight_10": {"training": {"loss_weights": {"seg": 10.0, "det": 0.1}}},
    "extreme_08_det_weight_10": {"training": {"loss_weights": {"seg": 0.1, "det": 10.0}}},
    "extreme_09_all_aug_max": {"data": {"augmentation_level": "heavy", "preprocessing": "standardize"}},
    "extreme_10_no_reg_at_all": {"training": {"weight_decay": 0.0}, "regularization": {"dropout_enabled": False}, "data": {"augmentation_level": "none"}}
})

# =============================================================================
# 6. ABLATION (10 experiments)
# Isolate single variables from baseline.
# =============================================================================
EXPERIMENT_CONFIGS.update({
    "ablation_01_backbone_34": {"model": {"backbone": "resnet34"}},
    "ablation_02_activation_gelu": {"model": {"activation": "gelu"}},
    "ablation_03_init_xavier": {"model": {"init_scheme": "xavier"}},
    "ablation_04_aug_basic": {"data": {"augmentation_level": "basic"}},
    "ablation_05_aug_heavy": {"data": {"augmentation_level": "heavy"}},
    "ablation_06_dropout_02": {"model": {"dropout_rate": 0.2}, "regularization": {"dropout_enabled": True}},
    "ablation_07_dropout_04": {"model": {"dropout_rate": 0.4}, "regularization": {"dropout_enabled": True}},
    "ablation_08_lr_1e4": {"training": {"lr": 1e-4}},
    "ablation_09_lr_1e2": {"training": {"lr": 1e-2}},
    "ablation_10_batch_32": {"training": {"batch_size": 32}}
})
