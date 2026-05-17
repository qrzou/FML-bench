import argparse
import json
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.special import expit, logit
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from causalml.inference.tf import DragonNet
from causalml.metrics import *
from causalml.propensity import ElasticNetPropensityModel

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(seed=42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

plt.style.use('fivethirtyeight')


def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (np.sin(np.pi * X[:, 0] * X[:, 1])
         + 2 * (X[:, 2] - 0.5) ** 2
         + X[:, 3]
         + 0.5 * X[:, 4])
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)))
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2
    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)
    return y, X, w, tau, b, e


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.114, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def carve_test_and_split(*arrays, test_fraction=0.21, random_state=42,
                         val_ratio=0.114, val_seed=42):
    """Step 1: carve hidden test (fixed). Step 2: split visible pool (agent-controlled)."""
    # Fixed hidden test
    split1 = train_test_split(*arrays, test_size=test_fraction,
                              random_state=random_state, shuffle=True)
    visible_arrays = split1[0::2]
    test_arrays = split1[1::2]
    # Agent-controlled val split from visible pool
    split2 = train_test_split(*visible_arrays, test_size=val_ratio,
                              random_state=val_seed, shuffle=True)
    train_arrays = split2[0::2]
    val_arrays = split2[1::2]
    return train_arrays, val_arrays, test_arrays


def run_ihdp_experiment(split, split_cfg=None):
    print("\n=== IHDP Experiment ===")
    df = pd.read_csv('docs/examples/data/ihdp_npci_3.csv', header=None)
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f'x{i}' for i in range(1, 26)]
    df.columns = cols

    X = df.loc[:, 'x1':]
    treatment = df['treatment']
    y = df['y_factual']
    tau = df.apply(lambda d: d['y_factual'] - d['y_cfactual'] if d['treatment'] == 1
                   else d['y_cfactual'] - d['y_factual'], axis=1)

    cfg = split_cfg or {"val_ratio": 0.114, "val_seed": 42}
    (X_train, treatment_train, y_train, tau_train), \
    (X_val, treatment_val, y_val, tau_val), \
    (X_test, treatment_test, y_test, tau_test) = \
        carve_test_and_split(X, treatment, y, tau,
                             val_ratio=cfg["val_ratio"], val_seed=cfg["val_seed"])

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    p_model = ElasticNetPropensityModel()
    p = p_model.fit_predict(X, treatment)

    dragon = DragonNet(neurons_per_layer=200, val_split=0.3, targeted_reg=True)
    dragon.fit(X_train, treatment_train, y_train)

    if split == 'val':
        X_eval, tau_eval, treatment_eval, y_eval = X_val, tau_val, treatment_val, y_val
    else:
        X_eval, tau_eval, treatment_eval, y_eval = X_test, tau_test, treatment_test, y_test

    dragon_ite_eval = dragon.predict_tau(X_eval).flatten()
    dragon_ate_eval = dragon_ite_eval.mean()

    df_preds = pd.DataFrame(
        [dragon_ite_eval.ravel(), tau_eval.ravel(), treatment_eval.ravel(), y_eval.ravel()],
        index=['dragonnet', 'tau', 'w', 'y']).T
    df_cumgain = get_cumgain(df_preds)

    df_result = pd.DataFrame(
        [dragon_ate_eval, tau_eval.mean()],
        index=['dragonnet', 'actual'], columns=['ATE'])
    df_result['MAE'] = [mean_absolute_error(tau_eval, dragon_ite_eval.ravel())] + [None]
    df_result['MSE'] = [mse(tau_eval, dragon_ite_eval.ravel())] + [None]
    df_result['AUUC'] = auuc_score(df_preds)
    df_result['Abs % Error of ATE'] = [np.abs((dragon_ate_eval / tau_eval.mean()) - 1)] + [None]

    print(f"\nIHDP {split} Results:")
    print(df_result)

    return {
        "ate_mean": float(df_result.loc['dragonnet', 'ATE']),
        "mae_mean": float(df_result.loc['dragonnet', 'MAE']),
        "mse_mean": float(df_result.loc['dragonnet', 'MSE']),
        "auuc_mean": float(df_result.loc['dragonnet', 'AUUC']),
        "abs_pct_error_of_ate_mean": float(df_result.loc['dragonnet', 'Abs % Error of ATE']),
    }


def run_synthetic_experiment(split, split_cfg=None):
    print("\n=== Synthetic Data Experiment ===")
    y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=1000, seed=42)

    cfg = split_cfg or {"val_ratio": 0.114, "val_seed": 42}
    (X_train, y_train, w_train, tau_train, b_train, e_train), \
    (X_val, y_val, w_val, tau_val, b_val, e_val), \
    (X_test, y_test, w_test, tau_test, b_test, e_test) = \
        carve_test_and_split(X, y, w, tau, b, e,
                             val_ratio=cfg["val_ratio"], val_seed=cfg["val_seed"])

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    dragonnet = DragonNet(neurons_per_layer=200, val_split=0.3, targeted_reg=True)
    dragonnet.fit(X_train, treatment=w_train, y=y_train)

    if split == 'val':
        X_eval, tau_eval, w_eval, y_eval = X_val, tau_val, w_val, y_val
    else:
        X_eval, tau_eval, w_eval, y_eval = X_test, tau_test, w_test, y_test

    tau_pred = dragonnet.predict_tau(X=X_eval).flatten()

    ate = float(tau_pred.mean())
    ate_actual = float(tau_eval.mean())
    mse_val = float(mse(tau_pred, tau_eval))
    mae_val = float(mean_absolute_error(tau_pred, tau_eval))
    abs_perc_error_ate = float(np.abs((ate / ate_actual) - 1))

    stacked = np.hstack((tau_pred, tau_eval))
    bins = np.linspace(np.percentile(stacked, 0.1), np.percentile(stacked, 99.9), 100)
    distr = np.histogram(tau_pred, bins=bins)[0]
    distr = np.clip(distr / distr.sum(), 0.001, 0.999)
    true_distr = np.histogram(tau_eval, bins=bins)[0]
    true_distr = np.clip(true_distr / true_distr.sum(), 0.001, 0.999)
    kl = float(entropy(distr, true_distr))

    df_preds = pd.DataFrame(
        [tau_pred.ravel(), tau_eval.ravel(), w_eval.ravel(), y_eval.ravel()],
        index=['DragonNet', 'tau', 'w', 'y']).T
    auuc = float(auuc_score(df_preds).iloc[0])

    print(f"\nSynthetic {split} Results:")
    print(f"  ATE: {ate:.6f}, MSE: {mse_val:.6f}, KL: {kl:.6f}, AUUC: {auuc:.6f}")

    return {
        "ate_mean": ate,
        "mse_mean": mse_val,
        "mae_mean": mae_val,
        "abs_pct_error_of_ate_mean": abs_perc_error_ate,
        "kl_divergence_mean": kl,
        "auuc_mean": auuc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    os.makedirs('results_tmp', exist_ok=True)

    # Read split_config for val path; test path ignores it (fixed hidden test)
    split_cfg = load_split_config() if args.split == 'val' else {"val_ratio": 0.114, "val_seed": 42}
    print(f"split_config: {split_cfg}")

    ihdp_metrics = run_ihdp_experiment(args.split, split_cfg=split_cfg)
    synthetic_metrics = run_synthetic_experiment(args.split, split_cfg=split_cfg)

    results = {
        "ihdp_test": {
            "means": {k: v for k, v in ihdp_metrics.items()},
            "stderrs": {k.replace('_mean', '_stderr'): 0.0 for k in ihdp_metrics},
            "final_info_dict": {k.replace('_mean', ''): [v] for k, v in ihdp_metrics.items()},
        },
        "synthetic_test": {
            "means": {k: v for k, v in synthetic_metrics.items()},
            "stderrs": {k.replace('_mean', '_stderr'): 0.0 for k in synthetic_metrics},
            "final_info_dict": {k.replace('_mean', ''): [v] for k, v in synthetic_metrics.items()},
        },
    }

    output_path = f'results_tmp/{args.split}_info.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
