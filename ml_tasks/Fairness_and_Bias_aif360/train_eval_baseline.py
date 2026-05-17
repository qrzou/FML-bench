import argparse
import json
import os
import random
import warnings
import logging

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger().setLevel(logging.CRITICAL)

import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)

set_seed(seed=233)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    dataset_orig = load_preproc_data_compas()

    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    # 3-way split: 70/9/21 (eval pool split 30/70)
    dataset_orig_train, dataset_orig_eval_pool = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_val, dataset_orig_test = dataset_orig_eval_pool.split([0.3], shuffle=True)

    print("### Training Dataset shape:", dataset_orig_train.features.shape)
    print("### Val Dataset shape:", dataset_orig_val.features.shape)
    print("### Test Dataset shape:", dataset_orig_test.features.shape)

    # Train
    sess = tf.Session()
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

    debiased_model = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name='debiased_classifier',
        debias=True,
        sess=sess,
    )
    debiased_model.fit(dataset_orig_train)

    # Evaluate
    eval_dataset = dataset_orig_val if args.split == 'val' else dataset_orig_test
    # Strip ground truth labels before prediction to prevent identity-function exploit:
    # without this, an agent can make predict() return the input unchanged, yielding
    # predictions = ground truth → perfect metrics.
    eval_for_predict = eval_dataset.copy(deepcopy=True)
    eval_for_predict.labels = np.zeros_like(eval_for_predict.labels)
    eval_for_predict.scores = np.zeros_like(eval_for_predict.scores)
    dataset_debiasing_eval = debiased_model.predict(eval_for_predict)

    cm = ClassificationMetric(eval_dataset, dataset_debiasing_eval,
                              unprivileged_groups=unprivileged_groups,
                              privileged_groups=privileged_groups)

    cls_acc = cm.accuracy()
    TPR = cm.true_positive_rate()
    TNR = cm.true_negative_rate()
    balanced_cls_acc = 0.5 * (TPR + TNR)
    disparate_impact = cm.disparate_impact()
    abs_disparate_impact_diff = abs(disparate_impact - 1)
    eod = cm.equal_opportunity_difference()
    abs_eod = abs(eod)
    aod = cm.average_odds_difference()
    abs_aod = abs(aod)
    theil_index = cm.theil_index()
    abs_theil_index = abs(theil_index)

    # Baseline accuracy constraint.
    # - Val: strict, no tolerance. Violation → fairness=None + raise ValueError.
    # - Test: 1% tolerance. Violation → report baseline fairness + preserve originals.
    baseline_cls_acc = 0.63510101010101

    # Save original fairness before any replacement
    original_fairness = {
        "disparate_impact": disparate_impact,
        "abs_disparate_impact_diff": abs_disparate_impact_diff,
        "eod": eod, "abs_eod": abs_eod,
        "aod": aod, "abs_aod": abs_aod,
        "theil_index": theil_index, "abs_theil_index": abs_theil_index,
    }

    constraint_violated = False

    if args.split == 'val' and cls_acc < baseline_cls_acc:
        # Val: strict — set fairness to None, will raise after saving
        constraint_violated = True
        disparate_impact = None; abs_disparate_impact_diff = None
        eod = None; abs_eod = None
        aod = None; abs_aod = None
        theil_index = None; abs_theil_index = None
    elif args.split == 'test' and cls_acc < baseline_cls_acc - 0.01:
        # Test: 1% tolerance — replace with baseline fairness
        constraint_violated = True
        bt = {
            "disparate_impact": 0.6753074541440007,
            "abs_disparate_impact_diff": 0.32469254585599927,
            "eod": -0.1890634357580504, "abs_eod": 0.1890634357580504,
            "aod": -0.23986259586405548, "abs_aod": 0.23986259586405548,
            "theil_index": 0.2058732707585479, "abs_theil_index": 0.2058732707585479,
        }
        disparate_impact = bt["disparate_impact"]
        abs_disparate_impact_diff = bt["abs_disparate_impact_diff"]
        eod = bt["eod"]; abs_eod = bt["abs_eod"]
        aod = bt["aod"]; abs_aod = bt["abs_aod"]
        theil_index = bt["theil_index"]; abs_theil_index = bt["abs_theil_index"]

    results = {
        "compas": {
            "means": {
                "cls_acc_mean": cls_acc,
                "balanced_cls_acc_mean": balanced_cls_acc,
                "disparate_impact_mean": disparate_impact,
                "abs_disparate_impact_diff_mean": abs_disparate_impact_diff,
                "eod_mean": eod,
                "abs_eod_mean": abs_eod,
                "aod_mean": aod,
                "abs_aod_mean": abs_aod,
                "theil_index_mean": theil_index,
                "abs_theil_index_mean": abs_theil_index,
                "constraint_violated": constraint_violated,
            },
            "stderrs": {k.replace("_mean", "_stderr"): 0.0 for k in [
                "cls_acc_mean", "balanced_cls_acc_mean", "disparate_impact_mean",
                "abs_disparate_impact_diff_mean", "eod_mean", "abs_eod_mean",
                "aod_mean", "abs_aod_mean", "theil_index_mean", "abs_theil_index_mean",
            ]},
            "final_info_dict": {
                "cls_acc": cls_acc,
                "balanced_cls_acc": balanced_cls_acc,
                "disparate_impact": disparate_impact,
                "abs_disparate_impact_diff": abs_disparate_impact_diff,
                "eod": eod,
                "abs_eod": abs_eod,
                "aod": aod,
                "abs_aod": abs_aod,
                "theil_index": theil_index,
                "abs_theil_index": abs_theil_index,
                "constraint_violated": constraint_violated,
            },
        }
    }

    # Test constraint violated: preserve agent's original metrics for observability
    if args.split == 'test' and constraint_violated:
        results["compas"]["original_metrics"] = original_fairness

    os.makedirs('results_tmp', exist_ok=True)
    output_path = f'results_tmp/{args.split}_info.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if args.split == 'val' and constraint_violated:
        print(f"[WARNING] Invalid results: cls_acc ({cls_acc}) < baseline_cls_acc ({baseline_cls_acc})")
        raise ValueError(f"Invalid results: cls_acc ({cls_acc}) < baseline_cls_acc ({baseline_cls_acc})")
    elif constraint_violated:
        print(f"[WARNING] Test constraint violated: cls_acc ({cls_acc:.4f}) < threshold ({baseline_cls_acc - 0.01:.4f}). "
              f"Reporting baseline fairness metrics. Original preserved in 'original_metrics'.")
    print(f"[INFO] Results saved to {output_path}")


if __name__ == '__main__':
    main()
