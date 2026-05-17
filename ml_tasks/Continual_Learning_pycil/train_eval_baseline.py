"""
Training and evaluation script for PyCIL continual learning task.

Dataset: CIFAR-100, class-incremental (50 base + 5×10 incremental)
Baseline: iCaRL with reduced epochs (60 base + 50 per stage)
Split: CIFAR-100 test set split 30/70 stratified (val=3000, test=7000).
       Training is deterministic (same seed, same data), so both paths
       produce identical models — only evaluation subset differs.
Metrics: Average incremental accuracy (top1 avg across all stages)

Usage:
    python train_eval_baseline.py --split val
    python train_eval_baseline.py --split test
"""
import argparse
import json
import os
import sys
import re
import numpy as np


def split_cifar100_test(test_targets, split_seed=42, val_ratio=0.3):
    """Split CIFAR-100 test set indices into val/test, stratified by class.

    Returns (val_indices, test_indices): 30% val, 70% test per class.
    """
    rng = np.random.RandomState(split_seed)
    val_indices = []
    test_indices = []

    for class_idx in range(100):
        class_mask = np.where(test_targets == class_idx)[0]
        shuffled = rng.permutation(class_mask)
        n_val = int(val_ratio * len(shuffled))
        val_indices.extend(shuffled[:n_val])
        test_indices.extend(shuffled[n_val:])

    return np.array(val_indices), np.array(test_indices)


def main():
    parser = argparse.ArgumentParser(description='PyCIL continual learning evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    # Import algorithm config
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from algorithm import get_pycil_config

    config = get_pycil_config()
    CHECKPOINT_DIR = 'model_checkpoint'

    # --- Monkey-patch DataManager to restrict test data to val or test subset ---
    from utils.data_manager import DataManager
    original_setup = DataManager._setup_data

    split_mode = args.split

    def patched_setup(self, dataset_name, shuffle, seed):
        original_setup(self, dataset_name, shuffle, seed)
        val_idx, test_idx = split_cifar100_test(self._test_targets)
        if split_mode == 'val':
            self._test_data = self._test_data[val_idx]
            self._test_targets = self._test_targets[val_idx]
            print(f"[SPLIT] Using val subset: {len(val_idx)} test samples (30%)")
        else:
            self._test_data = self._test_data[test_idx]
            self._test_targets = self._test_targets[test_idx]
            print(f"[SPLIT] Using test subset: {len(test_idx)} test samples (70%)")

    DataManager._setup_data = patched_setup

    # --- Train with patched evaluation ---
    config['seed'] = [1993]

    # Patch iCaRL module-level epoch variables before training
    import models.icarl as icarl_module
    icarl_module.init_epoch = config.get('init_epoch', 60)
    icarl_module.init_lr = config.get('init_lr', 0.1)
    icarl_module.init_milestones = config.get('init_milestones', [20, 40, 50])
    icarl_module.init_lr_decay = config.get('init_lr_decay', 0.1)
    icarl_module.init_weight_decay = config.get('init_weight_decay', 0.0005)
    icarl_module.epochs = config.get('epochs', 50)
    icarl_module.lrate = config.get('lrate', 0.1)
    icarl_module.milestones = config.get('milestones', [20, 35])
    icarl_module.lrate_decay = config.get('lrate_decay', 0.1)
    icarl_module.batch_size = config.get('batch_size', 128)
    icarl_module.weight_decay = config.get('weight_decay', 0.0002)

    print(f"Patched iCaRL: init_epoch={icarl_module.init_epoch}, epochs={icarl_module.epochs}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Run PyCIL training via its trainer module
    from trainer import train as pycil_train

    # Capture stdout to parse accuracy
    import io

    captured = io.StringIO()

    class TeeWriter:
        """Write to both stdout and capture buffer."""
        def __init__(self, original, capture):
            self.original = original
            self.capture = capture
        def write(self, text):
            self.original.write(text)
            self.capture.write(text)
        def flush(self):
            self.original.flush()
            self.capture.flush()

    old_stdout = sys.stdout
    sys.stdout = TeeWriter(old_stdout, captured)

    try:
        pycil_train(config)
    except ValueError as e:
        # PyCIL bug: forgetting matrix shape mismatch when init_cls != increment.
        # Occurs after all accuracy data is printed. Safe to continue.
        if "broadcast" in str(e):
            print(f"Known PyCIL bug (accuracy already captured): {e}")
        else:
            raise
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()

    # Parse average accuracy from output
    avg_acc_cnn = None
    avg_acc_nme = None

    for line in output.split('\n'):
        if 'Average Accuracy (CNN):' in line:
            match = re.search(r'Average Accuracy \(CNN\):\s*([\d.]+)', line)
            if match:
                avg_acc_cnn = float(match.group(1))
        if 'Average Accuracy (NME):' in line:
            match = re.search(r'Average Accuracy \(NME\):\s*([\d.]+)', line)
            if match:
                avg_acc_nme = float(match.group(1))

    # iCaRL uses NME (Nearest Mean Exemplar) for classification
    primary_acc = avg_acc_nme if avg_acc_nme is not None else avg_acc_cnn
    if primary_acc is None:
        print("ERROR: Could not parse accuracy from PyCIL output!", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== {args.split.upper()} Results ===")
    print(f"  Average Incremental Accuracy (CNN): {avg_acc_cnn}")
    print(f"  Average Incremental Accuracy (NME): {avg_acc_nme}")
    print(f"  Primary metric: {primary_acc:.4f}")

    # Format results
    results = {
        "cifar100_incremental": {
            "means": {
                "avg_incremental_acc_mean": float(primary_acc) / 100.0,
                "avg_acc_cnn_mean": float(avg_acc_cnn) / 100.0 if avg_acc_cnn else 0.0,
                "avg_acc_nme_mean": float(avg_acc_nme) / 100.0 if avg_acc_nme else 0.0,
            },
            "stderrs": {
                "avg_incremental_acc_stderr": 0.0,
                "avg_acc_cnn_stderr": 0.0,
                "avg_acc_nme_stderr": 0.0,
            },
            "final_info_dict": {
                "avg_incremental_acc": float(primary_acc) / 100.0,
                "avg_acc_cnn": float(avg_acc_cnn) / 100.0 if avg_acc_cnn else 0.0,
                "avg_acc_nme": float(avg_acc_nme) / 100.0 if avg_acc_nme else 0.0,
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
