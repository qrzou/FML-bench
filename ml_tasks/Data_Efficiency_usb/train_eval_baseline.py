"""
Training and evaluation script for USB semi-supervised learning task.

Dataset: CIFAR-100 with 200 labeled samples (4 per class) + full unlabeled set
Baseline: FixMatch (pseudo-labeling + strong augmentation) with WRN-28-2
Split: Fixed labeled/val/test split with seed
Metrics: test accuracy

Usage:
    python train_eval_baseline.py --split val
    python train_eval_baseline.py --split test
"""
import argparse
import json
import os
import sys
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import get_model, get_training_config, RandAugment


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SSLDataset(Dataset):
    """Wrapper to apply different transforms for weak/strong augmentation."""
    def __init__(self, dataset, transform_weak, transform_strong):
        self.dataset = dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        # img is a PIL Image (no transform applied yet)
        weak = self.transform_weak(img)
        strong = self.transform_strong(img)
        return weak, strong, target


def main():
    parser = argparse.ArgumentParser(description='USB FixMatch evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_training_config()

    # Transforms
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load CIFAR-100 WITHOUT transform (raw PIL)
    raw_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=None
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_eval
    )

    # Split: 200 labeled (4 per class) + remaining unlabeled
    # Also split test set into val/test: 30/70
    targets = np.array(raw_train.targets)
    num_classes = 100
    labels_per_class = config['num_labels'] // num_classes  # 4

    labeled_idx = []
    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        np.random.shuffle(class_idx)
        labeled_idx.extend(class_idx[:labels_per_class])

    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.setdiff1d(np.arange(len(raw_train)), labeled_idx)

    # Val/test split from CIFAR-100 test set (isolated RNG for determinism)
    n_test = len(test_dataset)
    split_rng = np.random.RandomState(42)
    test_indices = split_rng.permutation(n_test)
    n_val = int(0.3 * n_test)
    val_idx = test_indices[:n_val]
    test_idx = test_indices[n_val:]

    val_subset = Subset(test_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    # Create labeled and unlabeled datasets with appropriate transforms
    labeled_dataset = Subset(raw_train, labeled_idx)
    unlabeled_dataset = SSLDataset(
        Subset(raw_train, unlabeled_idx), transform_weak, transform_strong
    )

    # Labeled loader wraps with weak transform
    class LabeledDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img, target = self.subset[idx]
            return self.transform(img), target

    labeled_dataset = LabeledDataset(Subset(raw_train, labeled_idx), transform_weak)

    labeled_loader = DataLoader(labeled_dataset, batch_size=config['batch_size_labeled'],
                                shuffle=True, num_workers=2, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config['batch_size_unlabeled'],
                                  shuffle=True, num_workers=2, drop_last=True)

    eval_dataset = val_subset if args.split == 'val' else test_subset
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=2)

    print(f"Labeled: {len(labeled_idx)}, Unlabeled: {len(unlabeled_idx)}, "
          f"Val: {len(val_subset)}, Test: {len(test_subset)}")

    CHECKPOINT_DIR = 'model_checkpoint'
    model = get_model(num_classes=num_classes).to(device)

    if args.split == 'test':
        # Load checkpoint from val run (no training)
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'model.pt')
        print(f"Loading checkpoint from {ckpt_path} (skipping training)")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Jump to evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        print(f"\n=== TEST Results ===")
        print(f"  Accuracy: {test_acc:.6f}")

        results = {
            "cifar100_ssl": {
                "means": {"test_acc_mean": float(test_acc)},
                "stderrs": {"test_acc_stderr": 0.0},
                "final_info_dict": {"test_acc": float(test_acc)}
            }
        }
        os.makedirs('results_tmp', exist_ok=True)
        with open('results_tmp/test_info.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to results_tmp/test_info.json")
        return

    # --- VAL: Train + save checkpoint ---
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'],
                                nesterov=True)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['total_steps']
    )

    # Training loop (FixMatch)
    print(f"Training FixMatch for {config['total_steps']} steps...")
    model.train()
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(1, config['total_steps'] + 1):
        # Get labeled batch
        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)

        # Get unlabeled batch
        try:
            x_uw, x_us, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_uw, x_us, _ = next(unlabeled_iter)

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_uw, x_us = x_uw.to(device), x_us.to(device)

        # Supervised loss
        logits_l = model(x_l)
        loss_l = F.cross_entropy(logits_l, y_l)

        # Unsupervised loss (FixMatch)
        with torch.no_grad():
            logits_uw = model(x_uw)
            probs = torch.softmax(logits_uw / config['T'], dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            mask = max_probs.ge(config['threshold']).float()

        logits_us = model(x_us)
        loss_u = (F.cross_entropy(logits_us, pseudo_labels, reduction='none') * mask).mean()

        # Total loss
        loss = loss_l + config['lambda_u'] * loss_u
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % config['eval_every'] == 0 or step == config['total_steps']:
            # Quick eval
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in eval_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc = correct / total
            mask_rate = mask.mean().item() if mask.numel() > 0 else 0
            print(f"Step {step}/{config['total_steps']}: "
                  f"Loss_l={loss_l.item():.4f}, Loss_u={loss_u.item():.4f}, "
                  f"Mask={mask_rate:.2f}, {args.split}_acc={acc:.4f}")
            model.train()

    # Save checkpoint after training
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model.pt'))
    print(f"Saved checkpoint to {CHECKPOINT_DIR}/model.pt")

    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct / total
    print(f"\n=== {args.split.upper()} Results ===")
    print(f"  Accuracy: {test_acc:.6f}")

    results = {
        "cifar100_ssl": {
            "means": {
                "test_acc_mean": float(test_acc),
            },
            "stderrs": {
                "test_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "test_acc": float(test_acc),
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
