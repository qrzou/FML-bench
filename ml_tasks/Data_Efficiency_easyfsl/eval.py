import subprocess
import json
import re
from pathlib import Path

# Run the command and capture stdout
cmd = [
    "python", "-m", "scripts.benchmark_methods",
    "prototypical_networks",
    "data/features/mini_imagenet/test/feat_resnet12_mini_imagenet.parquet.gzip",
    "--config=default",
    "--n-shot=1",
    "--device=cuda",
    "--num-workers=12"
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

accuracy = float(match.group(1))

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

output_path = Path("results_tmp/final_info.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved accuracy {accuracy:.4f} to {output_path}")
