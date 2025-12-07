"""
Central configuration for hyperparameter experiments.
"""

CONFIG = {   'data': {   'augmentation_level': 'basic',
                'dataset_root': './data/VOC2012',
                'preprocessing': 'standardize'},
    'model': {   'activation': 'relu',
                 'backbone': 'resnet18',
                 'detection_head': 'simple',
                 'dropout_rate': 0.2,
                 'init_backbone': True,
                 'init_scheme': 'xavier',
                 'segmentation_head': 'unet'},
    'regularization': {'dropout_enabled': True},
    'system': {   'aim_experiment': 'cv-hyperparam-study',
                  'aim_repo': './aim_logs',
                  'device': 'cuda',
                  'epochs': 1,
                  'save_checkpoint': True,
                  'seed': 42},
    'training': {   'batch_size': 4,
                    'det_loss': 'smooth_l1',
                    'loss_weights': {'det': 1.0, 'seg': 1.0},
                    'lr': 0.002,
                    'seg_loss': 'cross_entropy',
                    'weight_decay': 0.0}}