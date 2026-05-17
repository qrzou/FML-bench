"""
Training and evaluation script for OpenOOD OOD detection task.

Dataset: CIFAR-10 as in-distribution, CIFAR-100 + SVHN as OOD
Baseline: MSP (Maximum Softmax Probability) with ResNet-18
Split: Hidden ID test carved first (70% of CIFAR-10 test = 7K, hardcoded),
       visible pool (CIFAR-10 train 50K + remaining 3K) split by split_config.json.
       OOD datasets: CIFAR-100 for val, SVHN for test (unchanged).
Metrics: AUROC, FPR95

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
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import get_model, get_training_config, compute_ood_score


class TransformSubset(Dataset):
    """A subset of data (numpy arrays) with a specific transform applied."""
    def __init__(self, data, targets, transform=None):
        self.data = data        # numpy array, uint8, (N, H, W, C)
        self.targets = targets  # numpy array, int
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[idx])


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


def compute_ood_metrics(id_scores, ood_scores):
    """Compute AUROC and FPR95 from ID/OOD score distributions."""
    # Labels: 1 = in-distribution, 0 = OOD
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    # FPR95: false positive rate at 95% true positive rate
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    tp_target = int(0.95 * n_pos)
    tp_count = 0
    fp_count = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
        if tp_count >= tp_target:
            break
    fpr95 = fp_count / n_neg if n_neg > 0 else 0.0

    return auroc, fpr95


def main():
    parser = argparse.ArgumentParser(description='OpenOOD MSP evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_training_config()

    # Transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ===== Step 1: Load CIFAR-10 train + test as numpy =====
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    train_data = np.array(cifar_train.data)       # (50000, 32, 32, 3) uint8
    train_targets = np.array(cifar_train.targets)  # (50000,)
    test_data = np.array(cifar_test.data)          # (10000, 32, 32, 3) uint8
    test_targets = np.array(cifar_test.targets)    # (10000,)

    # ===== Step 2: Carve hidden ID test (70% of 10K = 7K, FIXED seed) =====
    rng_hidden = np.random.RandomState(42)
    n_test = len(test_data)
    perm = rng_hidden.permutation(n_test)
    n_val_portion = int(0.3 * n_test)  # 3K for visible pool
    val_portion_idx = perm[:n_val_portion]
    hidden_test_idx = perm[n_val_portion:]  # 7K hidden

    hidden_test_data = test_data[hidden_test_idx]
    hidden_test_targets = test_targets[hidden_test_idx]
    val_portion_data = test_data[val_portion_idx]
    val_portion_targets = test_targets[val_portion_idx]

    # ===== Step 3: Visible pool = train (50K) + val portion (3K) = 53K =====
    visible_data = np.concatenate([train_data, val_portion_data], axis=0)
    visible_targets = np.concatenate([train_targets, val_portion_targets], axis=0)

    print(f"Visible pool: {len(visible_data)}, Hidden ID test: {len(hidden_test_data)}")

    # OOD datasets (unchanged)
    ood_val = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_eval
    )
    rng_svhn = np.random.RandomState(42)
    ood_test_full = torchvision.datasets.SVHN(
        root='./data/svhn', split='test', download=True, transform=transform_eval
    )
    from torch.utils.data import Subset
    svhn_idx = rng_svhn.permutation(len(ood_test_full))[:5000]
    ood_test = Subset(ood_test_full, svhn_idx)

    CHECKPOINT_DIR = 'model_checkpoint'
    model = get_model(num_classes=10).to(device)

    if args.split == 'val':
        # ===== Step 4a: Read split_config, split visible pool =====
        cfg = load_split_config()
        n_visible = len(visible_data)
        rng_split = np.random.RandomState(cfg["val_seed"])
        split_perm = rng_split.permutation(n_visible)
        n_agent_val = int(cfg["val_ratio"] * n_visible)
        agent_val_idx = split_perm[:n_agent_val]
        agent_train_idx = split_perm[n_agent_val:]

        agent_train_dataset = TransformSubset(
            visible_data[agent_train_idx], visible_targets[agent_train_idx], transform_train
        )
        agent_val_dataset = TransformSubset(
            visible_data[agent_val_idx], visible_targets[agent_val_idx], transform_eval
        )

        print(f"Agent train: {len(agent_train_dataset)}, Agent val: {len(agent_val_dataset)}")
        print(f"split_config: val_ratio={cfg['val_ratio']}, val_seed={cfg['val_seed']}")

        train_loader = DataLoader(agent_train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)

        # Train classifier
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config['lr_milestones'], gamma=config['lr_gamma']
        )

        print(f"\nTraining classifier for {config['epochs']} epochs...")
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            scheduler.step()
            if epoch % 20 == 0 or epoch == config['epochs']:
                print(f"Epoch {epoch}/{config['epochs']}: Acc={100.*correct/total:.2f}%")

        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model.pt'))
        print(f"Saved checkpoint to {CHECKPOINT_DIR}/model.pt")

        # Evaluate: ID = agent_val, OOD = CIFAR-100
        id_eval_loader = DataLoader(agent_val_dataset, batch_size=256, shuffle=False, num_workers=4)
        ood_eval_loader = DataLoader(ood_val, batch_size=256, shuffle=False, num_workers=4)
        ood_name = "CIFAR-100"

    else:
        # ===== Step 4b: Test — load checkpoint, use hidden test =====
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'model.pt')
        print(f"Loading checkpoint from {ckpt_path} (skipping training)")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        hidden_test_dataset = TransformSubset(
            hidden_test_data, hidden_test_targets, transform_eval
        )
        print(f"Hidden ID test: {len(hidden_test_dataset)}")

        id_eval_loader = DataLoader(hidden_test_dataset, batch_size=256, shuffle=False, num_workers=4)
        ood_eval_loader = DataLoader(ood_test, batch_size=256, shuffle=False, num_workers=4)
        ood_name = "SVHN"

    # Compute OOD scores
    model.eval()
    id_scores = []
    for inputs, _ in id_eval_loader:
        scores = compute_ood_score(model, inputs.to(device))
        id_scores.append(scores.cpu().numpy())
    id_scores = np.concatenate(id_scores)

    ood_scores = []
    for batch in ood_eval_loader:
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        scores = compute_ood_score(model, inputs.to(device))
        ood_scores.append(scores.cpu().numpy())
    ood_scores = np.concatenate(ood_scores)

    auroc, fpr95 = compute_ood_metrics(id_scores, ood_scores)

    print(f"\n=== {args.split.upper()} Results (OOD: {ood_name}) ===")
    print(f"  AUROC: {auroc:.6f}")
    print(f"  FPR95: {fpr95:.6f}")

    results = {
        "openood": {
            "ood_dataset": ood_name,  # "CIFAR-100" (val) or "SVHN" (test)
            "means": {
                "auroc_mean": float(auroc),
                "fpr95_mean": float(fpr95),
            },
            "stderrs": {
                "auroc_stderr": 0.0,
                "fpr95_stderr": 0.0,
            },
            "final_info_dict": {
                "auroc": float(auroc),
                "fpr95": float(fpr95),
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
