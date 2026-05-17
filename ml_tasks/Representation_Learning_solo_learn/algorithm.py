"""
Baseline self-supervised learning algorithm: Barlow Twins on CIFAR-100.

Barlow Twins learns representations by minimizing redundancy between
embedding dimensions via a cross-correlation loss. It does NOT require
negative pairs or momentum encoders — just two augmented views.

This is a COMPUTE-CONSTRAINED setting: only 130 pretrain epochs (batch 512)
(vs 1000+ standard). Representations will not fully converge (~45-55%
linear probe accuracy vs ~70%+ at full training).

Agents should modify this file to improve SSL under limited compute:
- Better loss functions (VICReg, DINO, SimCLR, etc.)
- Better augmentation strategies
- Better projection head design
- Learning rate scheduling
- Multi-crop strategies
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss: cross-correlation matrix → identity."""

    def __init__(self, lambda_param=5e-3, embedding_dim=2048):
        super().__init__()
        self.lambda_param = lambda_param
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, z1, z2):
        z1 = self.bn(z1)
        z2 = self.bn(z2)
        batch_size = z1.shape[0]

        # Cross-correlation matrix
        c = z1.T @ z2 / batch_size

        # Loss: on-diagonal → 1, off-diagonal → 0
        on_diag = (1 - c.diagonal()).pow(2).sum()
        off_diag = c.flatten()[:-1].view(c.shape[0] - 1, c.shape[0] + 1)[:, 1:].flatten().pow(2).sum()
        return on_diag + self.lambda_param * off_diag


class ProjectionHead(nn.Module):
    """3-layer projection head for Barlow Twins."""

    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwinsModel(nn.Module):
    """ResNet-18 backbone + projection head for Barlow Twins."""

    def __init__(self, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        # ResNet-18 backbone (remove final FC)
        backbone = models.resnet18(weights=None)
        self.backbone_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        # Adapt for CIFAR-100 (32x32): smaller first conv, no maxpool
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()
        self.backbone = backbone
        self.projector = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def get_model():
    """Return Barlow Twins model."""
    return BarlowTwinsModel(proj_hidden_dim=2048, proj_output_dim=2048)


def get_training_config():
    """Return Barlow Twins training config (compute-constrained)."""
    return {
        'pretrain_epochs': 130,
        'linear_eval_epochs': 100,
        'batch_size': 512,
        'lr': 0.4,
        'weight_decay': 1e-4,
        'lambda_param': 5e-3,       # Barlow Twins off-diagonal penalty
        'linear_lr': 0.1,
        'linear_batch_size': 256,
    }


def get_augmentation():
    """Return augmentation transforms for Barlow Twins."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
