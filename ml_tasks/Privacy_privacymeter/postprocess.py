import re
import json
import argparse
import os

def main(exp_dir: str):
    log_file = os.path.join(exp_dir, "report", "log_time_analysis.log")
    output_file = os.path.join(exp_dir, "final_info.json")

    auc_list = []
    tpr01_list = []
    tpr0_list = []

    pattern = re.compile(
        r"Target Model \d+: AUC ([0-9.]+), TPR@0.1%FPR of ([0-9.]+), TPR@0.0%FPR of ([0-9.]+)"
    )

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                auc = float(match.group(1))
                tpr01 = float(match.group(2))
                tpr0 = float(match.group(3))
                auc_list.append(auc)
                tpr01_list.append(tpr01)
                tpr0_list.append(tpr0)

    auc_mean = sum(auc_list) / len(auc_list)
    tpr01_mean = sum(tpr01_list) / len(tpr01_list)
    tpr0_mean = sum(tpr0_list) / len(tpr0_list)

    final_info = {
        "cifar10": {
            "means": {
                "AUC_mean": auc_mean,
                "TPR@0.1%FPR_mean": tpr01_mean,
                "TPR@0.0%FPR_mean": tpr0_mean,
            },
            "stderrs": {
                "AUC_stderr": 0.0,
                "TPR@0.1%FPR_stderr": 0.0,
                "TPR@0.0%FPR_stderr": 0.0,
            },
            "final_info_dict": {
                "AUC": auc_list,
                "TPR@0.1%FPR": tpr01_list,
                "TPR@0.0%FPR": tpr0_list,
            },
        }
    }

    with open(output_file, "w") as f:
        json.dump(final_info, f, indent=2)

    print(f"Saved results to {output_file}")
    # Print the metrics
    print("Metrics:")
    print(f"AUC_mean: {auc_mean}")
    print(f"TPR@0.1%FPR_mean: {tpr01_mean}")
    print(f"TPR@0.0%FPR_mean: {tpr0_mean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse log_time_analysis.log and extract final results")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory (e.g., cifar10_benchmark_tmp)")
    args = parser.parse_args()
    main(args.exp_dir)
