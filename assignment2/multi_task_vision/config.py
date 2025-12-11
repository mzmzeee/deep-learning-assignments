"""
Central configuration for hyperparameter experiments.
"""

CONFIG = {   'data': {   'augmentation_level': 'heavy',
                'dataset_root': './data/VOC2012',
                'preprocessing': 'normalize'},
    'model': {   'activation': 'leaky_relu',
                 'backbone': 'resnet34',
                 'detection_head': 'fpn',
                 'dropout_rate': 0.2,
                 'init_backbone': True,
                 'init_scheme': 'kaiming',
                 'segmentation_head': 'unet'},
    'regularization': {'dropout_enabled': True},
    'system': {   'aim_experiment': 'cv-hyperparam-study',
                  'aim_repo': './aim_logs',
                  'device': 'cuda',
                  'epochs': 60,
                  'save_checkpoint': True,
                  'seed': 42},
    'training': {   'batch_size': 32,
                    'det_loss': 'ciou',
                    'loss_weights': {'det': 1.0, 'seg': 1.0},
                    'lr': 0.001,
                    'seg_loss': 'focal',
                    'weight_decay': 0.0001}}