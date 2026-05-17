"""
Baseline DP training algorithm: Simple CNN with DP-SGD on CIFAR-10.

This baseline uses a simple 4-layer CNN with GroupNorm (required for DP-SGD,
since BatchNorm is incompatible with per-sample gradient computation).
The model is trained with Opacus DP-SGD at privacy budget epsilon=8.

Agents should modify this file to improve test accuracy under the DP constraint:
- Better model architectures (wider, deeper, different normalization)
- Learning rate scheduling
- Gradient clipping strategy
- Noise multiplier tuning
- Data augmentation
- Better optimizers
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with GroupNorm for DP-SGD compatibility.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=10):
    """Return the model to be trained with DP-SGD."""
    return SimpleCNN(num_classes=num_classes)


def get_training_config():
    """Return training hyperparameters."""
    return {
        'epochs': 20,
        'lr': 0.1,
        'momentum': 0.9,
        'max_grad_norm': 1.0,
        'target_epsilon': 8.0,
        'target_delta': 1e-5,
        'batch_size': 512,
    }
