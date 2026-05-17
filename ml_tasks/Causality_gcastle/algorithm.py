"""
Baseline causal discovery algorithm: NOTEARS (linear) on nonlinear data.

This baseline uses the linear NOTEARS algorithm, which assumes linear
structural equations. Since the data is generated from nonlinear (MLP-based)
structural equations, this baseline is intentionally mismatched and will
perform poorly (high SHD).

Agents should modify this file to improve causal discovery, e.g.:
- Switch to NOTEARS-MLP (nonlinear variant)
- Try DAGMA, GraNDAG, or other nonlinear methods
- Adjust thresholding strategy
- Modify regularization or acyclicity penalty
"""
import numpy as np


def run_causal_discovery(X, true_dag=None):
    """
    Run causal structure learning on observed data.

    Args:
        X: np.ndarray of shape (n_samples, n_nodes) — observational data
        true_dag: np.ndarray of shape (n_nodes, n_nodes) — ground truth DAG
                  (adjacency matrix, provided for reference but should NOT
                  be used by the algorithm)

    Returns:
        estimated_dag: np.ndarray of shape (n_nodes, n_nodes) — estimated
                       binary adjacency matrix (1 = edge, 0 = no edge)
    """
    from castle.algorithms import Notears

    # NOTEARS linear — continuous optimization with acyclicity constraint
    model = Notears()
    model.learn(X)

    # Get the weighted adjacency matrix
    causal_matrix = model.causal_matrix

    # Threshold to binary (default: edges with |weight| > 0.3)
    threshold = 0.3
    estimated_dag = (np.abs(causal_matrix) > threshold).astype(int)

    # Remove self-loops
    np.fill_diagonal(estimated_dag, 0)

    return estimated_dag
