"""
Training and evaluation script for solo-learn Barlow Twins task.

Dataset: CIFAR-100 (unlabeled pretrain + labeled linear eval)
Baseline: Barlow Twins with ResNet-18, 200 pretrain epochs (compute-constrained)
Split: Visible pool (train + val portion of test) for agent; hidden test held out
Metrics: Linear probe top-1 accuracy

Usage:
    python train_eval_baseline.py --split val
    python train_eval_baseline.py --split test
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import get_model, get_training_config, get_augmentation, BarlowTwinsLoss


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.057, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


class TransformSubset(Dataset):
    """Dataset from numpy arrays with a transform applied."""
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class TwoViewSubset(Dataset):
    """Dataset from numpy arrays that returns two augmented views (for contrastive pretraining)."""
    def __init__(self, data, targets, augmentation):
        self.data = data
        self.targets = targets
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return self.augmentation(img), self.augmentation(img)


def pretrain(model, train_loader, config, device):
    """Barlow Twins pretraining."""
    criterion = BarlowTwinsLoss(lambda_param=config['lambda_param'],
                                 embedding_dim=2048).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['pretrain_epochs']
    )

    model.train()
    for epoch in range(1, config['pretrain_epochs'] + 1):
        total_loss = 0
        for x1, x2 in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 20 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            print(f"Pretrain Epoch {epoch}/{config['pretrain_epochs']}: Loss={avg_loss:.4f}")

    return model


def linear_eval(model, train_loader, eval_loader, config, device, seed=42):
    """Linear evaluation: freeze backbone, train linear classifier."""
    torch.manual_seed(seed)

    # Freeze backbone
    model.eval()
    backbone_dim = model.backbone_dim

    # Extract features
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                feat, _ = model(inputs.to(device))
                features.append(feat.cpu())
                labels.append(targets)
        return torch.cat(features), torch.cat(labels)

    train_feat, train_labels = extract_features(train_loader)
    eval_feat, eval_labels = extract_features(eval_loader)

    # Train linear classifier
    classifier = nn.Linear(backbone_dim, 100).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=config['linear_lr'],
                                momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['linear_eval_epochs']
    )

    # Create feature dataloaders
    train_ds = torch.utils.data.TensorDataset(train_feat, train_labels)
    train_dl = DataLoader(train_ds, batch_size=config['linear_batch_size'], shuffle=True)

    for epoch in range(config['linear_eval_epochs']):
        classifier.train()
        for feat, labels in train_dl:
            feat, labels = feat.to(device), labels.to(device)
            logits = classifier(feat)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        logits = classifier(eval_feat.to(device))
        _, predicted = logits.max(1)
        accuracy = predicted.eq(eval_labels.to(device)).float().mean().item()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Barlow Twins SSL evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_training_config()
    augmentation = get_augmentation()

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ---- Step 1: Load CIFAR-100 train + test as numpy arrays ----
    raw_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True
    )
    raw_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True
    )

    train_data = np.array(raw_train.data)       # (50000, 32, 32, 3)
    train_targets = np.array(raw_train.targets)  # (50000,)
    test_data = np.array(raw_test.data)          # (10000, 32, 32, 3)
    test_targets = np.array(raw_test.targets)    # (10000,)

    # ---- Step 2: Carve hidden test from test set (70%, hardcoded seed=42) ----
    n_test = len(test_data)
    rng_split = np.random.RandomState(42)
    perm = rng_split.permutation(n_test)
    n_val_portion = int(0.3 * n_test)  # 3000
    val_portion_idx = perm[:n_val_portion]
    hidden_test_idx = perm[n_val_portion:]

    val_portion_data = test_data[val_portion_idx]
    val_portion_targets = test_targets[val_portion_idx]
    hidden_test_data = test_data[hidden_test_idx]
    hidden_test_targets = test_targets[hidden_test_idx]

    # ---- Step 3: Visible pool = train (50K) + val portion of test (3K) = 53K ----
    visible_data = np.concatenate([train_data, val_portion_data], axis=0)
    visible_targets = np.concatenate([train_targets, val_portion_targets], axis=0)

    print(f"Train: {len(train_data)}, Val portion: {len(val_portion_data)}, "
          f"Hidden test: {len(hidden_test_data)}, Visible pool: {len(visible_data)}")

    CHECKPOINT_DIR = 'model_checkpoint'

    if args.split == 'val':
        # ---- Step 4 (val): Read split_config, split visible pool ----
        split_cfg = load_split_config()
        val_ratio = split_cfg['val_ratio']
        val_seed = split_cfg['val_seed']

        n_visible = len(visible_data)
        rng_val = np.random.RandomState(val_seed)
        indices = rng_val.permutation(n_visible)
        n_agent_val = int(val_ratio * n_visible)
        agent_val_idx = indices[:n_agent_val]
        agent_train_idx = indices[n_agent_val:]

        agent_train_data = visible_data[agent_train_idx]
        agent_train_targets = visible_targets[agent_train_idx]
        agent_val_data = visible_data[agent_val_idx]
        agent_val_targets = visible_targets[agent_val_idx]

        print(f"Agent train: {len(agent_train_data)}, Agent val: {len(agent_val_data)} "
              f"(val_ratio={val_ratio}, val_seed={val_seed})")

        # Pretraining dataset (two views)
        pretrain_dataset = TwoViewSubset(agent_train_data, agent_train_targets, augmentation)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=config['batch_size'],
                                     shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

        # Linear eval loaders
        train_labeled_dataset = TransformSubset(agent_train_data, agent_train_targets, transform_eval)
        train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=256,
                                           shuffle=False, num_workers=4)
        val_dataset = TransformSubset(agent_val_data, agent_val_targets, transform_eval)
        eval_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

        # Step 1: Pretrain Barlow Twins (expensive: ~25 min)
        print(f"\n=== Pretraining Barlow Twins for {config['pretrain_epochs']} epochs ===")
        model = get_model().to(device)
        model = pretrain(model, pretrain_loader, config, device)

        # Save pretrained backbone for test split reuse
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'backbone.pt'))
        print(f"Saved pretrained backbone to {CHECKPOINT_DIR}/backbone.pt")

        # Step 2: Linear evaluation (fast: ~3 min)
        linear_seed = 100
        print(f"\n=== Linear evaluation ({config['linear_eval_epochs']} epochs, seed={linear_seed}) ===")
        accuracy = linear_eval(model, train_labeled_loader, eval_loader, config, device, seed=linear_seed)

    else:
        # ---- Step 4 (test): Load checkpoint, evaluate on hidden test ----
        print(f"\n=== Loading pretrained backbone from {CHECKPOINT_DIR}/backbone.pt ===")
        model = get_model().to(device)
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'backbone.pt'), map_location=device))
        print("Loaded pretrained backbone (skipping pretraining)")

        # For linear eval training, use full visible pool
        train_labeled_dataset = TransformSubset(visible_data, visible_targets, transform_eval)
        train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=256,
                                           shuffle=False, num_workers=4)

        # Evaluate on hidden test
        test_dataset = TransformSubset(hidden_test_data, hidden_test_targets, transform_eval)
        eval_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

        # Linear evaluation
        linear_seed = 200
        print(f"\n=== Linear evaluation ({config['linear_eval_epochs']} epochs, seed={linear_seed}) ===")
        accuracy = linear_eval(model, train_labeled_loader, eval_loader, config, device, seed=linear_seed)

    print(f"\n=== {args.split.upper()} Results ===")
    print(f"  Linear probe accuracy: {accuracy:.6f}")

    results = {
        "cifar100_ssl": {
            "means": {
                "linear_eval_acc_mean": float(accuracy),
            },
            "stderrs": {
                "linear_eval_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "linear_eval_acc": float(accuracy),
            }
        }
    }

    os.makedirs('results_tmp', exist_ok=True)
    output_path = f'results_tmp/{args.split}_info.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
