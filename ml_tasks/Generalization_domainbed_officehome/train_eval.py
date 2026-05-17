"""
Wrapper script for DomainBed: runs training then extracts val or test metrics.

Val/test split strategy for DomainBed (OfficeHome with test_env=0, Art domain):
  - val:  Out-split accuracy on held-out test domain (env0_out, 30%)
  - test: In-split accuracy on held-out test domain (env0_in, 70%)

Both val and test measure generalization to the same unseen domain (env0/Art).
The holdout_fraction=0.3 splits each domain into 30% out-split and 70% in-split.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def run_training(output_dir="./results_tmp"):
    """Run DomainBed training on OfficeHome. Returns the process exit code."""
    cmd = [
        "python", "-m", "domainbed.scripts.train",
        "--data_dir=../data/",
        "--algorithm", "ERM",
        "--dataset", "OfficeHome",
        "--test_env", "0",
        "--holdout_fraction", "0.3",
        "--output_dir", output_dir,
        "--steps", "5001",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def extract_metrics(output_dir, split):
    """Extract metrics from results.jsonl based on split type."""
    results_file = os.path.join(output_dir, "results.jsonl")

    with open(results_file, 'r') as f:
        lines = f.readlines()

    # Parse the last result line
    last_line = lines[-1].strip()
    if not last_line:
        last_line = lines[-2].strip()
    last_result = json.loads(last_line)

    if split == 'val':
        # Val: out-split accuracy on held-out test domain (env0/Art, 30%)
        accuracy = last_result.get('env0_out_acc', 0.0)
        description = f"Val (held-out domain env0/Art out_split 30%): {accuracy:.4f}"
    else:
        # Test: in-split accuracy on held-out test domain (env0/Art, 70%)
        accuracy = last_result.get('env0_in_acc', 0.0)
        description = f"Test (held-out domain env0/Art in_split 70%): {accuracy:.4f}"

    print(f"\n{description}")
    print(f"avg_acc_mean = {accuracy:.6f}")

    return {
        "OfficeHome_test_env0": {
            "means": {
                "avg_acc_mean": accuracy,
            },
            "stderrs": {
                "avg_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "avg_acc": [accuracy],
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='DomainBed OfficeHome train + eval with val/test split')
    parser.add_argument('--split', choices=['val', 'test'], required=True,
                        help='Which evaluation split to report: val or test')
    args = parser.parse_args()

    output_dir = "./results_tmp"
    checkpoint_dir = "./model_checkpoint"

    if args.split == 'val':
        # Run training (expensive: ~12 min)
        returncode = run_training(output_dir)
        if returncode != 0:
            print(f"Training failed with return code {returncode}", file=sys.stderr)
            sys.exit(returncode)

        # Save results.jsonl for test reuse
        os.makedirs(checkpoint_dir, exist_ok=True)
        src = os.path.join(output_dir, "results.jsonl")
        dst = os.path.join(checkpoint_dir, "results.jsonl")
        shutil.copy2(src, dst)
        print(f"Saved results.jsonl to {checkpoint_dir}/ for test reuse")
    else:
        # Load saved results.jsonl (no training)
        src = os.path.join(checkpoint_dir, "results.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(output_dir, "results.jsonl"))
        print(f"Loaded results.jsonl from {checkpoint_dir}/ (skipping training)")

    # Extract metrics for the requested split
    results = extract_metrics(output_dir, args.split)

    # Save results
    output_path = os.path.join(output_dir, f"{args.split}_info.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
