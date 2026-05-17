"""
Training and evaluation script for Opacus differential privacy task.

Dataset: CIFAR-10 (auto-downloaded via torchvision)
Baseline: Simple CNN + DP-SGD, epsilon=8, delta=1e-5
Split: Hidden test (21% of 50K, fixed seed=42) carved first.
       Visible pool (79% = 39,500) split into agent_train / agent_val
       via split_config.json (default: val_ratio=0.114, val_seed=42).
       The original 10K test set is NOT used — we split the 50K train set.
Metrics: test_accuracy, epsilon (privacy budget consumed)

Usage:
    python train_eval_baseline.py --split val
    python train_eval_baseline.py --split test
"""
import argparse
import json
import os
import sys
import warnings

# Ensure we use pip-installed opacus, not the local repo source
# (running from within the opacus repo would shadow the pip package)
cwd = os.getcwd()
sys.path = [p for p in sys.path if not (p == cwd and os.path.exists(os.path.join(p, 'opacus', '__init__.py')))]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

# Re-add cwd for algorithm.py import
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from algorithm import get_model, get_training_config


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.114, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Opacus DP training evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_training_config()

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 train set (50K images)
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    # ===== Step 1: Carve out hidden test (FIXED, ignores split_config) =====
    # Use a separate RandomState so hidden test indices are deterministic
    # regardless of split_config or global np.random state.
    n = len(full_dataset)
    hidden_rng = np.random.RandomState(42)
    all_indices = hidden_rng.permutation(n)
    n_test = int(0.21 * n)  # 10,500 hidden test samples
    test_idx = all_indices[:n_test]
    visible_idx = all_indices[n_test:]  # 39,500 visible pool

    if args.split == 'val':
        # ===== Step 2a: Read split_config, split visible pool =====
        cfg = load_split_config()
        visible_rng = np.random.RandomState(cfg["val_seed"])
        visible_perm = visible_rng.permutation(len(visible_idx))
        n_val = int(cfg["val_ratio"] * len(visible_idx))
        val_local = visible_perm[:n_val]
        train_local = visible_perm[n_val:]
        train_idx = visible_idx[train_local]
        val_idx = visible_idx[val_local]
        eval_idx = val_idx
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)} (hidden)")
        print(f"split_config: val_ratio={cfg['val_ratio']}, val_seed={cfg['val_seed']}")
    else:
        # ===== Step 2b (test): Use entire visible pool for training context, eval on hidden test =====
        train_idx = visible_idx
        eval_idx = test_idx
        print(f"Visible pool (train): {len(train_idx)}, Hidden test: {len(test_idx)}")

    train_dataset = Subset(full_dataset, train_idx)

    # For eval, use test transform (no augmentation)
    eval_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_test
    )
    eval_dataset = Subset(eval_full, eval_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=2)

    CHECKPOINT_DIR = 'model_checkpoint'
    final_epsilon = config['target_epsilon']

    if args.split == 'test':
        # Load checkpoint from val run (no training)
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'model.pt')
        print(f"Loading checkpoint from {ckpt_path} (skipping training)")
        model = get_model(num_classes=10).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        # Load saved epsilon
        meta_path = os.path.join(CHECKPOINT_DIR, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            final_epsilon = meta.get('epsilon', config['target_epsilon'])
    else:
        # --- VAL: Train with DP-SGD + save checkpoint ---
        model = get_model(num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                    momentum=config['momentum'])
        criterion = nn.CrossEntropyLoss()

        # Wrap with Opacus DP
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=config['epochs'],
            target_epsilon=config['target_epsilon'],
            target_delta=config['target_delta'],
            max_grad_norm=config['max_grad_norm'],
        )

        # Training loop
        print(f"Training with DP-SGD for {config['epochs']} epochs...")
        for epoch in range(config['epochs']):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epsilon = privacy_engine.get_epsilon(config['target_delta'])
            train_acc = 100. * correct / total
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Loss={running_loss/(batch_idx+1):.4f}, "
                  f"Train Acc={train_acc:.2f}%, "
                  f"ε={epsilon:.2f}")

        final_epsilon = privacy_engine.get_epsilon(config['target_delta'])

        # Save checkpoint (unwrap Opacus GradSampleModule)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        raw_model = model._module if hasattr(model, '_module') else model
        torch.save(raw_model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model.pt'))
        with open(os.path.join(CHECKPOINT_DIR, 'meta.json'), 'w') as f:
            json.dump({'epsilon': float(final_epsilon)}, f)
        print(f"Saved checkpoint to {CHECKPOINT_DIR}/model.pt (ε={final_epsilon:.4f})")

    # Evaluate
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
    print(f"  Epsilon: {final_epsilon:.4f}")

    # Format results
    results = {
        "cifar10_dp": {
            "means": {
                "test_acc_mean": float(test_acc),
                "epsilon_mean": float(final_epsilon),
            },
            "stderrs": {
                "test_acc_stderr": 0.0,
                "epsilon_stderr": 0.0,
            },
            "final_info_dict": {
                "test_acc": float(test_acc),
                "epsilon": float(final_epsilon),
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
