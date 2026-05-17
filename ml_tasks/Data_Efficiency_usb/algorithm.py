"""
Baseline semi-supervised learning algorithm: FixMatch on CIFAR-100.

FixMatch combines pseudo-labeling with consistency regularization:
1. Generate pseudo-labels from weakly-augmented unlabeled data
2. Only keep pseudo-labels with confidence above a threshold
3. Train on both labeled data (cross-entropy) and unlabeled data
   (cross-entropy on pseudo-labels with strongly-augmented input)

Agents should modify this file to improve semi-supervised learning:
- Better pseudo-labeling strategies (adaptive threshold, curriculum)
- Different augmentation strategies
- Better loss formulations
- Novel semi-supervised methods (MixMatch, ReMixMatch, FlexMatch, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np


class WideResNetBlock(nn.Module):
    """Basic block for WideResNet."""

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.dropout(F.relu(self.bn2(out)))
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    """WideResNet-28-2 for CIFAR."""

    def __init__(self, depth=28, widen_factor=2, num_classes=100, dropout_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.layer1 = self._make_layer(n, channels[0], channels[1], 1, dropout_rate)
        self.layer2 = self._make_layer(n, channels[1], channels[2], 2, dropout_rate)
        self.layer3 = self._make_layer(n, channels[2], channels[3], 2, dropout_rate)
        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, n, in_channels, out_channels, stride, dropout_rate):
        layers = [WideResNetBlock(in_channels, out_channels, stride, dropout_rate)]
        for _ in range(1, n):
            layers.append(WideResNetBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def get_model(num_classes=100):
    """Return WideResNet-28-2."""
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes)


def get_training_config():
    """Return FixMatch training hyperparameters."""
    return {
        'total_steps': 15000,
        'eval_every': 1500,
        'batch_size_labeled': 64,
        'batch_size_unlabeled': 64,
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'threshold': 0.95,     # Pseudo-label confidence threshold
        'lambda_u': 1.0,       # Weight for unlabeled loss
        'T': 1.0,              # Temperature for sharpening
        'num_labels': 200,     # Total labeled samples (4 per class)
    }


class RandAugment:
    """Simple RandAugment for strong augmentation."""

    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augmentations = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=1.0),
            transforms.RandomAutocontrast(p=1.0),
            transforms.RandomEqualize(p=1.0),
        ]

    def __call__(self, img):
        ops = np.random.choice(self.augmentations, self.n, replace=False)
        for op in ops:
            img = op(img)
        return img
