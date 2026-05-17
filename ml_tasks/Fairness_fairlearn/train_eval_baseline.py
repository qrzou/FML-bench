"""
Training and evaluation script for Fairlearn fairness task.

Dataset: Adult Census Income (auto-downloaded via sklearn)
Baseline: Unconstrained LogisticRegression (no fairness)
Split: Hidden test carved first (21%, hardcoded), visible pool (79%) split
       by split_config.json (agent-controlled). Default: ~11.4% val from pool.
Metrics: balanced_accuracy, equalized_odds_difference, demographic_parity_difference

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
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Add the ml_tasks directory to path so we can import algorithm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import FairnessAlgorithm


def load_adult_data():
    """Load Adult Census Income dataset with all features (including categorical)."""
    from sklearn.datasets import fetch_openml
    import pandas as pd
    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    X = data.data
    y = (data.target == '>50K').astype(int)

    # Extract sensitive feature before encoding
    sensitive_feature = X['sex'].copy()

    # One-hot encode categorical features (including sex — the baseline is fairness-unaware)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill numeric NaN with median
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Fill any remaining NaN
    X = X.fillna(0)

    return X, y, sensitive_feature


def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    """Compute fairness and accuracy metrics."""
    from fairlearn.metrics import (
        equalized_odds_difference,
        demographic_parity_difference,
    )
    from sklearn.metrics import balanced_accuracy_score, accuracy_score

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'equalized_odds_diff': float(eod),
        'demographic_parity_diff': float(dpd),
        'abs_equalized_odds_diff': float(abs(eod)),
        'abs_demographic_parity_diff': float(abs(dpd)),
    }


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.114, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def main():
    parser = argparse.ArgumentParser(description='Fairlearn fairness task evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True,
                        help='Which evaluation split to use: val or test')
    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(42)

    # Load data
    print("Loading Adult Census Income dataset...")
    X, y, sensitive = load_adult_data()

    # ===== Step 1: Carve out hidden test (FIXED, ignores split_config) =====
    X_visible, X_test, y_visible, y_test, s_visible, s_test = train_test_split(
        X, y, sensitive, test_size=0.21, random_state=42, stratify=y
    )

    CHECKPOINT_DIR = 'model_checkpoint'

    if args.split == 'val':
        # ===== Step 2a: Read split_config, split visible pool =====
        cfg = load_split_config()
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X_visible, y_visible, s_visible,
            test_size=cfg["val_ratio"],
            random_state=cfg["val_seed"],
            stratify=y_visible
        )
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} (hidden)")
        print(f"split_config: val_ratio={cfg['val_ratio']}, val_seed={cfg['val_seed']}")

        # Train model and save checkpoint
        print("Training model...")
        model = FairnessAlgorithm()
        model.fit(X_train, y_train, sensitive_features=s_train)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        import pickle
        with open(os.path.join(CHECKPOINT_DIR, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved checkpoint to {CHECKPOINT_DIR}/model.pkl")

        X_eval, y_eval, s_eval = X_val, y_val, s_val
    else:
        # ===== Step 2b: Completely ignores split_config =====
        import pickle
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'model.pkl')
        print(f"Loading checkpoint from {ckpt_path} (skipping training)")
        with open(ckpt_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Test: {len(X_test)} (hidden)")

        X_eval, y_eval, s_eval = X_test, y_test, s_test

    y_pred = model.predict(X_eval)
    metrics = compute_fairness_metrics(y_eval, y_pred, s_eval)

    print(f"\n=== {args.split.upper()} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Format results in FML-bench standard format
    results = {
        "adult": {
            "means": {
                "accuracy_mean": metrics['accuracy'],
                "balanced_accuracy_mean": metrics['balanced_accuracy'],
                "equalized_odds_diff_mean": metrics['equalized_odds_diff'],
                "demographic_parity_diff_mean": metrics['demographic_parity_diff'],
                "abs_equalized_odds_diff_mean": metrics['abs_equalized_odds_diff'],
                "abs_demographic_parity_diff_mean": metrics['abs_demographic_parity_diff'],
            },
            "stderrs": {
                "accuracy_stderr": 0.0,
                "balanced_accuracy_stderr": 0.0,
                "equalized_odds_diff_stderr": 0.0,
                "demographic_parity_diff_stderr": 0.0,
                "abs_equalized_odds_diff_stderr": 0.0,
                "abs_demographic_parity_diff_stderr": 0.0,
            },
            "final_info_dict": {
                "accuracy": metrics['accuracy'],
                "balanced_accuracy": metrics['balanced_accuracy'],
                "equalized_odds_diff": metrics['equalized_odds_diff'],
                "demographic_parity_diff": metrics['demographic_parity_diff'],
                "abs_equalized_odds_diff": metrics['abs_equalized_odds_diff'],
                "abs_demographic_parity_diff": metrics['abs_demographic_parity_diff'],
            }
        }
    }

    # Save results
    os.makedirs('results_tmp', exist_ok=True)
    output_path = f'results_tmp/{args.split}_info.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
