"""
Baseline OOD detection algorithm: MSP (Maximum Softmax Probability).

MSP is the simplest OOD detection baseline — it uses the maximum softmax
probability as the OOD score. Higher confidence = more likely in-distribution.

Agents should modify this file to improve OOD detection:
- Energy score (LogSumExp instead of max softmax)
- ODIN (temperature scaling + input perturbation)
- ReAct (feature truncation at a threshold)
- ASH (activation shaping)
- Mahalanobis distance
- KNN-based detection
- Training-time methods (outlier exposure, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 images)."""

    def __init__(self, num_classes=10):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Adapt for CIFAR: smaller conv1, no maxpool
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits

    def get_features(self, x):
        """Extract penultimate features."""
        return self.backbone(x)


def get_model(num_classes=10):
    """Return classifier model."""
    return ResNet18CIFAR(num_classes=num_classes)


def get_training_config():
    """Return training hyperparameters."""
    return {
        'epochs': 100,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_milestones': [50, 75, 90],
        'lr_gamma': 0.1,
    }


def compute_ood_score(model, x):
    """
    Compute OOD score for input batch.

    Returns scores where HIGHER = more likely IN-DISTRIBUTION.
    (Framework will negate for OOD detection: higher OOD score = more OOD)

    Args:
        model: trained classifier
        x: input tensor (batch_size, C, H, W)

    Returns:
        scores: tensor of shape (batch_size,)
    """
    with torch.no_grad():
        logits = model(x)
        # MSP: max softmax probability
        probs = F.softmax(logits, dim=1)
        scores = probs.max(dim=1).values
    return scores
