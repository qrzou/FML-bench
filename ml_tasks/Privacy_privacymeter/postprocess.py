"""
Postprocess MIA results with val/test split and constraint checks.

Splits target model results into val (30%) / test (70%):
- Shuffled with fixed seed=42 for unbiased assignment
- val: 30% of target models
- test: 70% of target models
"""
import re
import json
import argparse
import numpy as np
import os


def main(exp_dir: str, split: str):
    log_file = os.path.join(exp_dir, "report", "log_time_analysis.log")

    auc_list = []
    tpr01_list = []
    tpr0_list = []
    test_acc_list = []

    auc_pattern = re.compile(
        r"Target Model \d+: AUC ([0-9.]+), TPR@0.1%FPR of ([0-9.]+), TPR@0.0%FPR of ([0-9.]+)"
    )
    test_acc_pattern = re.compile(r"Test accuracy (\d+\.\d+)")

    with open(log_file, "r") as f:
        for line in f:
            auc_match = auc_pattern.search(line)
            if auc_match:
                auc_list.append(float(auc_match.group(1)))
                tpr01_list.append(float(auc_match.group(2)))
                tpr0_list.append(float(auc_match.group(3)))
            acc_match = test_acc_pattern.search(line)
            if acc_match:
                test_acc_list.append(float(acc_match.group(1)))

    if not auc_list:
        raise ValueError("No AUC values found in log file.")
    if not test_acc_list:
        raise ValueError("No test accuracy values found in log file.")

    # Split by target model index (shuffled for unbiased 30/70 split)
    all_indices = list(range(len(auc_list)))
    rng = np.random.RandomState(42)
    rng.shuffle(all_indices)
    split_point = int(0.3 * len(auc_list))
    if split_point < 1:
        split_point = 1
    if split == 'val':
        indices = sorted(all_indices[:split_point])
    else:
        indices = sorted(all_indices[split_point:])

    if len(auc_list) <= 1:
        indices = list(range(len(auc_list)))

    auc_split = [auc_list[i] for i in indices]
    tpr01_split = [tpr01_list[i] for i in indices]
    tpr0_split = [tpr0_list[i] for i in indices]
    # Use all test_acc values (shared across splits since it's model accuracy)
    test_acc_split = test_acc_list

    auc_mean = sum(auc_split) / len(auc_split) if auc_split else 0.0
    tpr01_mean = sum(tpr01_split) / len(tpr01_split) if tpr01_split else 0.0
    tpr0_mean = sum(tpr0_split) / len(tpr0_split) if tpr0_split else 0.0
    test_acc_mean = sum(test_acc_split) / len(test_acc_split) if test_acc_split else 0.0

    auc_gap = abs(auc_mean - 0.5)
    auc_gap_list = [abs(auc - 0.5) for auc in auc_split]

    # Constraint checks.
    # - Val: strict, no tolerance. Violation → auc_gap=None + raise ValueError.
    # - Test: tolerant. Violation → report baseline AUC_gap + preserve originals.
    baseline_tpr01 = 0.0
    baseline_tpr0 = 0.0
    baseline_test_acc = 0.610
    violations = []

    # Save original metrics before any replacement
    original_metrics = {
        "AUC_gap": auc_gap,
        "AUC_gap_list": auc_gap_list,
        "TPR@0.1%FPR": tpr01_mean,
        "TPR@0.0%FPR": tpr0_mean,
        "test_acc": test_acc_mean,
    }

    if split == 'val':
        # Val: strict — no tolerance
        if tpr01_mean > baseline_tpr01 or tpr0_mean > baseline_tpr0:
            auc_gap = None
            violations.append(f"TPR constraint violated: TPR@0.1%={tpr01_mean}, TPR@0.0%={tpr0_mean}")
        if test_acc_mean < baseline_test_acc:
            auc_gap = None
            violations.append(f"Accuracy constraint violated: {test_acc_mean} < {baseline_test_acc}")
    else:
        # Test: tolerant — TPR@0.1% ≤ 0.02, TPR@0.0% ≤ 0.005, acc ≥ 0.600
        if tpr01_mean > 0.02 or tpr0_mean > 0.005:
            violations.append(f"TPR constraint violated: TPR@0.1%={tpr01_mean}, TPR@0.0%={tpr0_mean}")
        if test_acc_mean < baseline_test_acc - 0.01:
            violations.append(f"Accuracy constraint violated: {test_acc_mean} < {baseline_test_acc - 0.01}")

        if violations:
            # Fallback to baseline test AUC_gap
            auc_gap = 0.3213666666666667
            auc_gap_list = [0.32030000000000003, 0.31799999999999995, 0.3258]

    constraint_violated = bool(violations)

    final_info = {
        "cifar10": {
            "means": {
                "AUC_gap_mean": auc_gap,
                "TPR@0.1%FPR_mean": tpr01_mean,
                "TPR@0.0%FPR_mean": tpr0_mean,
                "test_acc_mean": test_acc_mean,
                "constraint_violated": constraint_violated,
            },
            "stderrs": {
                "AUC_gap_stderr": 0.0,
                "TPR@0.1%FPR_stderr": 0.0,
                "TPR@0.0%FPR_stderr": 0.0,
                "test_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "AUC_gap": auc_gap_list,
                "TPR@0.1%FPR": tpr01_split,
                "TPR@0.0%FPR": tpr0_split,
                "test_acc": test_acc_split,
                "constraint_violated": constraint_violated,
            },
        }
    }

    # Test constraint violated: preserve agent's original metrics for observability
    if split == 'test' and constraint_violated:
        final_info["cifar10"]["original_metrics"] = original_metrics

    output_file = os.path.join(exp_dir, f"{split}_info.json")
    with open(output_file, "w") as f:
        json.dump(final_info, f, indent=2)

    if split == 'val' and violations:
        violation_reason = " AND ".join(violations)
        print(f"[WARNING] Invalid results: {violation_reason}")
        raise ValueError(f"Invalid results: {violation_reason}")
    elif constraint_violated:
        violation_reason = " AND ".join(violations)
        print(f"[WARNING] Test constraint violated: {violation_reason}. "
              f"Reporting baseline AUC_gap. Original preserved in 'original_metrics'.")
    print(f"Saved {split} results to {output_file}")
    print(f"AUC_gap_mean: {auc_gap}, test_acc_mean: {test_acc_mean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()
    main(args.exp_dir, args.split)
