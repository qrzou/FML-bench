"""
Training and evaluation script for gCastle causal discovery task.

Dataset: Synthetic nonlinear (MLP-based) structural equation model
         50 nodes, ~100 edges, n=500 samples
Baseline: NOTEARS-linear (intentionally mismatched to nonlinear data)
Split: Different random seeds for val/test data generation
       (same true DAG structure, different noise realizations)
Metrics: SHD, F1, FDR, TPR

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

warnings.filterwarnings('ignore')

# Add the ml_tasks directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_random_dag(n_nodes, n_edges, seed=42):
    """Generate a random DAG using Erdos-Renyi model."""
    rng = np.random.RandomState(seed)

    # Generate a random permutation (topological order)
    perm = rng.permutation(n_nodes)

    # Sample edges only from lower-index to higher-index (ensures DAG)
    dag = np.zeros((n_nodes, n_nodes), dtype=int)
    possible_edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            possible_edges.append((perm[i], perm[j]))

    # Randomly select n_edges from possible edges
    n_edges = min(n_edges, len(possible_edges))
    selected = rng.choice(len(possible_edges), size=n_edges, replace=False)
    for idx in selected:
        i, j = possible_edges[idx]
        dag[i, j] = 1

    return dag


def generate_nonlinear_data(dag, n_samples, seed=42):
    """
    Generate data from nonlinear (MLP-based) structural equations.

    Each node X_j = f_j(parents(X_j)) + noise
    where f_j is a random 1-hidden-layer MLP.
    """
    rng = np.random.RandomState(seed)
    n_nodes = dag.shape[0]
    X = np.zeros((n_samples, n_nodes))

    # Topological sort
    in_degree = dag.sum(axis=0)
    order = []
    queue = list(np.where(in_degree == 0)[0])
    temp_in = in_degree.copy()
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in range(n_nodes):
            if dag[node, child] == 1:
                temp_in[child] -= 1
                if temp_in[child] == 0:
                    queue.append(child)

    # Generate data in topological order
    for node in order:
        parents = np.where(dag[:, node] == 1)[0]
        noise = rng.randn(n_samples) * 0.5

        if len(parents) == 0:
            X[:, node] = noise
        else:
            # Random 1-hidden-layer MLP: X_node = tanh(X_parents @ W1 + b1) @ W2 + noise
            n_parents = len(parents)
            hidden_dim = max(4, n_parents)
            W1 = rng.randn(n_parents, hidden_dim) * 0.8
            b1 = rng.randn(hidden_dim) * 0.3
            W2 = rng.randn(hidden_dim, 1) * 0.8

            parent_data = X[:, parents]
            hidden = np.tanh(parent_data @ W1 + b1)
            X[:, node] = (hidden @ W2).flatten() + noise

    return X


def compute_metrics(true_dag, estimated_dag):
    """Compute causal discovery evaluation metrics."""
    true_edges = set(zip(*np.where(true_dag == 1)))
    est_edges = set(zip(*np.where(estimated_dag == 1)))

    # SHD: Structural Hamming Distance
    # Count: missing edges + extra edges + reversed edges
    shd = 0
    all_edges = true_edges | est_edges
    for edge in all_edges:
        i, j = edge
        if true_dag[i, j] == 1 and estimated_dag[i, j] == 0:
            # Missing edge
            if estimated_dag[j, i] == 1 and true_dag[j, i] == 0:
                shd += 1  # Reversed
            else:
                shd += 1  # Missing
        elif true_dag[i, j] == 0 and estimated_dag[i, j] == 1:
            if (j, i) not in true_edges:
                shd += 1  # Extra edge

    # TP, FP, FN for undirected skeleton
    tp = len(true_edges & est_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        'shd': int(shd),
        'f1': float(f1),
        'fdr': float(fdr),
        'tpr': float(recall),
        'precision': float(precision),
        'n_true_edges': len(true_edges),
        'n_est_edges': len(est_edges),
    }


def main():
    parser = argparse.ArgumentParser(description='gCastle causal discovery evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True,
                        help='Which evaluation split to use: val or test')
    args = parser.parse_args()

    # Configuration
    n_nodes = 50
    n_edges = 100  # ~degree 4
    n_samples = 500
    dag_seed = 42  # Same DAG structure for both val and test
    val_data_seed = 100
    test_data_seed = 200

    # Generate ground truth DAG (same for val and test)
    print(f"Generating random DAG: {n_nodes} nodes, {n_edges} edges...")
    true_dag = generate_random_dag(n_nodes, n_edges, seed=dag_seed)
    print(f"True DAG has {true_dag.sum()} edges")

    # Generate data with split-specific seed
    data_seed = val_data_seed if args.split == 'val' else test_data_seed
    print(f"Generating nonlinear data: n={n_samples}, seed={data_seed}...")
    X = generate_nonlinear_data(true_dag, n_samples, seed=data_seed)

    # Run causal discovery algorithm
    print("Running causal discovery algorithm...")
    from algorithm import run_causal_discovery
    estimated_dag = run_causal_discovery(X, true_dag=true_dag)

    # Compute metrics
    metrics = compute_metrics(true_dag, estimated_dag)

    print(f"\n=== {args.split.upper()} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Format results
    results = {
        "synthetic_dag": {
            "means": {
                "shd_mean": float(metrics['shd']),
                "f1_mean": metrics['f1'],
                "fdr_mean": metrics['fdr'],
                "tpr_mean": metrics['tpr'],
            },
            "stderrs": {
                "shd_stderr": 0.0,
                "f1_stderr": 0.0,
                "fdr_stderr": 0.0,
                "tpr_stderr": 0.0,
            },
            "final_info_dict": {
                "shd": float(metrics['shd']),
                "f1": metrics['f1'],
                "fdr": metrics['fdr'],
                "tpr": metrics['tpr'],
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
