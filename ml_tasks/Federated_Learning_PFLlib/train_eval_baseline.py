"""
Training and evaluation script for Federated Learning task (PFLlib).

Dataset: CIFAR-10, 20 clients, Dirichlet alpha=0.1 (non-IID)
Baseline: FedAvg + FedAvgCNN, 100 global rounds, 1 local epoch
Split: Hidden test = PFLlib's test/{0..19}.npz (25% per client, fixed).
       Visible pool = PFLlib's train/{0..19}.npz (75% per client).
       Agent controls train/val split of visible pool via split_config.json.
Metrics: test_acc (average across clients)

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
from torch.utils.data import TensorDataset

warnings.filterwarnings('ignore')

# Ensure cwd is on path for algorithm.py import
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from algorithm import get_fl_config, FLClient, FLServer


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.3, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_client_data(data_dir, client_id, is_train=True):
    """Load per-client data from PFLlib's generated npz files."""
    subdir = "train" if is_train else "test"
    path = os.path.join(data_dir, subdir, f"{client_id}.npz")
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].item()
    x = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    return TensorDataset(x, y)


def split_dataset(dataset, val_ratio, val_seed):
    """Split a TensorDataset into train/val portions."""
    n = len(dataset)
    rng = np.random.RandomState(val_seed)
    perm = rng.permutation(n)
    n_val = int(val_ratio * n)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    x_all, y_all = dataset.tensors
    train_ds = TensorDataset(x_all[train_indices], y_all[train_indices])
    val_ds = TensorDataset(x_all[val_indices], y_all[val_indices])
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description='Federated Learning evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_fl_config()
    num_clients = config['num_clients']
    assert num_clients == 20, (
        f"num_clients must be 20 (fixed by data partition), got {num_clients}"
    )

    # Data directory (PFLlib convention)
    data_dir = os.path.join('dataset', 'Cifar10')
    if not os.path.exists(os.path.join(data_dir, 'train', '0.npz')):
        raise RuntimeError(
            f"CIFAR-10 federated data not found at {data_dir}. "
            "Run: cd dataset && python generate_Cifar10.py noniid - dir"
        )

    CHECKPOINT_DIR = 'model_checkpoint'

    if args.split == 'val':
        # Load split_config for train/val split of visible pool
        split_cfg = load_split_config()
        print(f"split_config: val_ratio={split_cfg['val_ratio']}, val_seed={split_cfg['val_seed']}")

        # Load visible pool (train data) and split into actual_train / val
        client_train_data = []
        client_val_data = []
        for i in range(num_clients):
            visible_pool = load_client_data(data_dir, i, is_train=True)
            train_ds, val_ds = split_dataset(visible_pool, split_cfg['val_ratio'], split_cfg['val_seed'])
            client_train_data.append(train_ds)
            client_val_data.append(val_ds)
            print(f"  Client {i}: train={len(train_ds)}, val={len(val_ds)}")

        # Create FL server and clients
        server = FLServer(device=device)
        clients = []
        for i in range(num_clients):
            client = FLClient(i, client_train_data[i], device=device)
            clients.append(client)

        # FL training loop
        print(f"\nStarting FedAvg training: {config['global_rounds']} rounds, "
              f"{num_clients} clients, {config['local_epochs']} local epochs")

        for rnd in range(config['global_rounds']):
            # Select clients (all by default with join_ratio=1.0)
            n_selected = max(1, int(num_clients * config['join_ratio']))
            selected = np.random.choice(num_clients, n_selected, replace=False)

            # Send global model, local training
            global_state = server.get_global_state()
            client_states = []
            client_weights = []
            for idx in selected:
                clients[idx].set_model(global_state)
                state, n_samples = clients[idx].local_train()
                client_states.append(state)
                client_weights.append(n_samples)

            # Aggregate
            server.aggregate(client_states, client_weights)

            # Evaluate periodically
            if rnd % 10 == 0 or rnd == config['global_rounds'] - 1:
                accs = server.evaluate(client_val_data)
                avg_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f"  Round {rnd}: avg_acc={avg_acc:.4f}, std={std_acc:.4f}")

        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(server.global_model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, 'global_model.pt'))
        print(f"Saved checkpoint to {CHECKPOINT_DIR}/global_model.pt")

        # Final evaluation on val data
        eval_data = client_val_data

    else:
        # Test: load checkpoint, evaluate on hidden test data
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'global_model.pt')
        print(f"Loading checkpoint from {ckpt_path} (skipping training)")
        server = FLServer(device=device)
        server.global_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True))

        # Load hidden test data
        eval_data = []
        for i in range(num_clients):
            test_ds = load_client_data(data_dir, i, is_train=False)
            eval_data.append(test_ds)
            print(f"  Client {i}: test={len(test_ds)}")

    # Evaluate
    client_accs = server.evaluate(eval_data)
    avg_acc = float(np.mean(client_accs))
    std_acc = float(np.std(client_accs))

    print(f"\n=== {args.split.upper()} Results ===")
    print(f"  Average accuracy: {avg_acc:.6f}")
    print(f"  Std across clients: {std_acc:.6f}")

    # Format results
    results = {
        "cifar10_fl": {
            "means": {
                "test_acc_mean": avg_acc,
                "test_acc_std_mean": std_acc,
            },
            "stderrs": {
                "test_acc_stderr": 0.0,
                "test_acc_std_stderr": 0.0,
            },
            "final_info_dict": {
                "test_acc": avg_acc,
                "test_acc_std": std_acc,
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
