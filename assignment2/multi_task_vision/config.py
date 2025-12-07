"""
Central configuration for hyperparameter experiments.
"""

CONFIG = {   'data': {   'augmentation_level': 'heavy',
                'dataset_root': './data/VOC2012',
                'preprocessing': 'standardize'},
    'model': {   'activation': 'relu',
                 'backbone': 'resnet34',
                 'detection_head': 'simple',
                 'dropout_rate': 0.4,
                 'init_backbone': True,
                 'init_scheme': 'kaiming',
                 'segmentation_head': 'unet'},
    'regularization': {'dropout_enabled': True},
    'system': {   'aim_experiment': 'cv-hyperparam-study',
                  'aim_repo': './aim_logs',
                  'device': 'cuda',
                  'epochs': 2,
                  'save_checkpoint': True,
                  'seed': 42},
    'training': {   'batch_size': 8,
                    'det_loss': 'ciou',
                    'loss_weights': {'det': 1.0, 'seg': 1.0},
                    'lr': 0.0005,
                    'seg_loss': 'cross_entropy',
                    'weight_decay': 1e-05}}