# Assignment 1 Part C Report

## MLP Results
- Architecture: 64-input -> Linear(128) -> ReLU -> Linear(10) -> Softmax
- Training: 1000 epochs, lr=0.05, full-batch SGD
- Test accuracy: **0.9499**

## CNN Baseline Results
- Architecture: Conv(1→8, 3×3, pad=1) → ReLU → MaxPool(2)
  → Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2)
  → Flatten → Linear(64→10) → Softmax
- Training: 50 epochs, batch size 32, lr=0.01
- Test accuracy: **0.9573**

## CNN Experiments
1. Increased filter counts to 16/32/64 and added a third Conv block.
2. Added a third MaxPooling layer to reduce spatial size to 1×1 before the classifier.
3. Raised learning rate to 0.02 and trained for 80 epochs with batch size 32.

## Best CNN Results
- Architecture: Conv(1→16, 3×3, pad=1) → ReLU → MaxPool(2)
  → Conv(16→32, 3×3, pad=1) → ReLU → MaxPool(2)
  → Conv(32→64, 3×3, pad=1) → ReLU → MaxPool(2)
  → Flatten → Linear(64→10) → Softmax
- Training: 80 epochs, batch size 32, lr=0.02
- Test accuracy: **0.9796**
- Plots: `assingment1/figure/c_task4_best_cnn_loss.png`, `assingment1/figure/c_task4_best_cnn_cm.png`

## Comparison
The CNN variants consistently outperform the MLP. Convolutions exploit spatial locality and share parameters across the image, enabling the network to detect translation-invariant patterns with far fewer weights than a dense layer operating on flattened pixels. Pooling layers build a hierarchy of increasingly abstract features, improving generalization. The deeper, wider CNN captures richer structures and delivers the highest accuracy, highlighting why CNNs are a better fit for image classification than MLPs that lack built-in spatial bias.
