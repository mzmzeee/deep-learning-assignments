"""
Custom layers and weight initialization schemes.
Students modify the activation functions and initialization here.
"""

import torch.nn as nn
import torch.nn.init as init


def get_activation(name):
    """Return activation function based on config."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


def initialize_weights(module, scheme="kaiming"):
    """
    Initialize weights of a module.
    Students can modify this function to experiment with different init schemes.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if scheme == "kaiming":
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif scheme == "xavier":
            init.xavier_normal_(module.weight)
        elif scheme == "normal":
            init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init scheme: {scheme}")

        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


class ConvBlock(nn.Module):
    """Basic conv block with activation and optional dropout."""

    def __init__(self, in_channels, out_channels, activation="relu", dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

        # Dropout for regularization experiments
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x