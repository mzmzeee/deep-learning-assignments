"""
Central configuration for hyperparameter experiments.
"""

CONFIG = {   'data': {   'augmentation_level': 'heavy',
                'dataset_root': './data/VOC2012',
                'preprocessing': 'normalize'},
    'model': {   'activation': 'gelu',
                 'backbone': 'resnet34',
                 'detection_head': 'fpn',
                 'dropout_rate': 0.2,
                 'init_backbone': True,
                 'init_scheme': 'kaiming',
                 'segmentation_head': 'fcn'},
    'regularization': {'dropout_enabled': True},
    'system': {   'aim_experiment': 'cv-hyperparam-study',
                  'aim_repo': './aim_logs',
                  'device': 'cuda',
                  'epochs': 5,
                  'save_checkpoint': True,
                  'seed': 42},
    'training': {   'batch_size': 8,
                    'det_loss': 'ciou',
                    'loss_weights': {'det': 1.5, 'seg': 1.0},
                    'lr': 0.001,
                    'seg_loss': 'focal',
                    'weight_decay': 0.0001}}