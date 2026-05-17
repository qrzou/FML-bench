import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], required=True,
                        help='Which evaluation split to use: val or test')
    args = parser.parse_args()

    # Use different random seeds for val vs test to get independent episode sets.
    # Both draw from the same feature pool, so the distribution is identical.
    seed = 42 if args.split == 'val' else 43

    cmd = [
        "python", "-m", "scripts.benchmark_methods",
        "prototypical_networks",
        "data/features/mini_imagenet/test/feat_resnet12_mini_imagenet.parquet.gzip",
        "--config=default",
        "--n-shot=1",
        "--device=cuda",
        "--num-workers=12",
        f"--random-seed={seed}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Combine stdout and stderr in case logs are split
    output_text = result.stdout + "\n" + result.stderr

    # Search for the accuracy line
    match = re.search(r"Average accuracy\s*:\s*([\d\.]+)", output_text)
    if not match:
        print("Could not find accuracy in output! Full output below:\n")
        print(output_text)
        raise SystemExit(1)

    accuracy = float(match.group(1)) / 100.0  # Convert from percentage to [0, 1]

    # Save to JSON
    results = {
        "miniimagenet_test": {
            "means": {
                "accuracy_mean": accuracy
            },
            "stderrs": {
                "accuracy_stderr": 0.0
            },
            "final_info_dict": {
                "accuracy": [accuracy]
            }
        }
    }

    output_path = Path(f"results_tmp/{args.split}_info.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved accuracy {accuracy:.6f} to {output_path}")


if __name__ == '__main__':
    main()
