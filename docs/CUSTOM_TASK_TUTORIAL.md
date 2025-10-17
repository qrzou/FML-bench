# Custom Task Tutorial

This guide shows how to add a new task to FML-bench. Start with TheAIScientist to set up the common task structure and config; then reuse the same structure for AIDE (Weco) and Claude Code.

## Table of Contents

  - [TheAIScientist](#1-theaiscientist)
  - [AIDE (Weco)](#2-aide-weco)
  - [Claude Code](#3-claude-code)

## 1) TheAIScientist

TheAIScientist is the reference path. Do these steps to define your task once:

### A. Workspace Setup (integrate into `scripts/workspace_and_env_setup.sh` and `scripts/setup_fmlbench.sh`)

Run the specific commands for your new task, and add your repo to the setup script in the same style as existing tasks:

```bash
# =============================================================================
# X. YOUR TASK: YOURREPO
# =============================================================================
echo "Setting up Your Task: yourrepo..."

# Step 1: Git clone and download
echo "  → Cloning yourrepo repository..."
mkdir -p workspace/YourTaskName_yourrepo
cd workspace/YourTaskName_yourrepo
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
git checkout your_commit_hash
cd ../../..

# Step 2: Create environment
echo "  → Creating your_env conda environment..."
conda create -n your_env python=3.10 -y
conda activate your_env
pip install your_dependencies
pip install weco==0.3.0 # If you also need to test weco
conda deactivate

echo "✓ Your Task: yourrepo setup completed"
```

### B. Test Eval Command and Generate Baseline Results

After workspace setup, test your eval command and generate baseline results:

```bash
# Test environment activation and eval command
conda activate your_env
cd workspace/YourTaskName_yourrepo/yourrepo

# Run your eval command to test and generate baseline results
python eval.py  # or whatever your main execution command is

# If you have postprocessing, run it to generate final_info.json
python postprocess.py --directory ./results_tmp/  # if you use postprocess.py

# Copy baseline results to your task directory
mkdir -p ../../../ml_tasks/YourTaskName_yourrepo/baseline_results/
cp ./results_tmp/final_info.json ../../../ml_tasks/YourTaskName_yourrepo/baseline_results/

conda deactivate
cd ../../..
```

This step ensures your task works correctly and provides baseline results that agents can compare against.

### C. Task Directory (under `ml_tasks/`)

Create your task folder and include the core files:

```
ml_tasks/YourTaskName_yourrepo/
├── config.json              # Task execution configuration
├── prompt.json              # Task description for AI agents
├── postprocess.py           # Results processing script (optional)
├── baseline_results/        # Directory for baseline results
│   └──final_info.json
├── original_file_backup/    # Backup of original files
├── theaiscientist/          # TheAIScientist agent integration
│   └── seed_idea.json
├── weco/                    # WECO agent integration
│   └── run_weco.sh
├── claude_code/             # Claude Code agent integration
│   ├── run_claude_code.sh
│   └── prompt.txt
└── (task-specific files)    # optional
```

Minimal required files to get started for TheAIScientist: your repository executes via commands in the YAML config (below). `prompt.json` and `baseline_results/final_info.json` are also required.

### D. Create `config.json` (under your task folder)

This is the core configuration file that defines how your task is executed:

```json
{
    "repo_dir": "workspace/YourTaskName_yourrepo/yourrepo",
    "target_files": [
        "path/to/file1.py",
        "path/to/file2.py"
    ],
    "related_files": [
        "path/to/helper_file.py"
    ],
    "read_only_files": [
        "path/to/readonly_file.py",
        "eval.py"
    ],
    "starter_file": "",
    "backup_excluded_files": [
        ".aider.tags.cache.v4/",
        "results_tmp/"
    ],
    "conda_env": "your_env_name",
    "prepare_setup_files": true,
    "setup_commands": [
        "cp ../../../ml_tasks/YourTaskName_yourrepo/eval.py ./"
    ],
    "do_preprocess": true,
    "preprocess_note": "Copy necessary files to workspace",
    "preprocess_commands": [
        "cp ../../../ml_tasks/YourTaskName_yourrepo/eval.py ./",
        "cp ../../../ml_tasks/YourTaskName_yourrepo/original_file_backup/some_file.py ./path/to/"
    ],
    "execute_commands": [
        "python eval.py"
    ],
    "exp_running_dir": "results_tmp/",
    "do_postprocess": true,
    "postprocess_commands": [
        "cp ../../../ml_tasks/YourTaskName_yourrepo/postprocess.py ./",
        "python postprocess.py --directory ./results_tmp/",
        "rm postprocess.py"
    ],
    "note": "The final_info.json should be UNDER the exp_running_dir"
}
```

Key fields to customize:
- `repo_dir`, `target_files`, `read_only_files`
- `conda_env`
- `execute_commands`, `preprocess_commands`, `postprocess_commands`

### E. Create `prompt.json` (under your task folder)

Define the task description for AI agents:

```json
{
    "system": "You are an AI PhD student focused on [your domain].",
    "task_description": "Your task is to [detailed task description]. The goal is to [specific objective]. You should [guidelines and constraints]."
}
```

### F. Create `postprocess.py` (Optional)

If your task needs custom result processing:

```python
#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path

def process_results(directory):
    """Process experiment results and generate final_info.json"""
    
    # Your custom result processing logic here
    # Parse your experiment results and extract metrics
    
    # Example: Single dataset with multiple metrics
    results = {
        "your_dataset_name": {
            "means": {
                "your_metric_mean": 0.85,
                "another_metric_mean": 0.15
            },
            "stderrs": {
                "your_metric_stderr": 0.0,
                "another_metric_stderr": 0.0
            },
            "final_info_dict": {
                "your_metric": [0.85],
                "another_metric": [0.15]
            }
        }
    }
    
    # Example: Multiple datasets
    # results = {
    #     "dataset1": {
    #         "means": {"metric1_mean": 0.85},
    #         "stderrs": {"metric1_stderr": 0.0},
    #         "final_info_dict": {"metric1": [0.85]}
    #     },
    #     "dataset2": {
    #         "means": {"metric1_mean": 0.90},
    #         "stderrs": {"metric1_stderr": 0.0},
    #         "final_info_dict": {"metric1": [0.90]}
    #     }
    # }
    
    # Save results
    output_file = Path(directory) / "final_info.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, help="Results directory")
    args = parser.parse_args()
    process_results(args.directory)
```

### G. YAML Config (under `configs/`)

Create a YAML config that defines your benchmark and agent settings (TheAIScientist runs via this file). Use this detailed template:

```yaml
# configs/your_task.yaml

# Default configuration for run_agent_benchmark.py

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            AGENT CONFIGURATION                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
agent:
  type: theaiscientist        # ✦ Agent type (options: theaiscientist)
  model: gpt-5-2025-08-07     # ✦ LLM model to use (e.g. gpt-4, gpt-4o, gpt-4o-mini, etc.)
  provider: OpenAI            # ✦ LLM provider (OpenAI, Anthropic, OpenRouter, Google, DeepSeek)

  # ── TheAIScientist-specific settings ──────────────────────────────────────
  theaiscientist:
    max_runs: 5                 # ▶ Maximum experiment runs per idea
    max_retries_per_run: 4      # ▶ Maximum retry attempts for failed experiments (was MAX_ITERS)
    max_iter: 100               # ▶ (Optional) Maximum total iterations (runs + retries). If null, unlimited (bounded by max_runs × max_retries_per_run × num_ideas)
    max_stderr_output: 1500     # ▶ Maximum stderr output to show in retry prompt (characters)
    num_ideas: 25               # ▶ Number of ideas to generate
    num_reflections: 3          # ▶ Number of reflections for idea generation
    skip_novelty_check: true    # ▶ Skip novelty checking for generated ideas
    engine: semanticscholar     # ▶ Scholar engine for novelty check
    use_existing_ideas: false   # ▶ Use existing ideas from file (if true)
    ideas_file: null            # ▶ Path to existing ideas file (used if use_existing_ideas is true)
    execute_timeout: 7200       # ▶ Timeout for experiment execution commands (seconds, default: 2 hours)
    preprocess_timeout: 300     # ▶ Timeout for preprocessing commands (seconds, default: 5 minutes)
    postprocess_timeout: 300    # ▶ Timeout for postprocessing commands (seconds, default: 5 minutes)
    use_early_completion: true  # ▶ Use early completion for experiment iteration

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         BENCHMARK CONFIGURATION                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝
benchmark:
  name: YourTaskName_yourrepo    # ✦ Benchmark name (must match your task directory name exactly)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                          METRICS CONFIGURATION                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
metrics:
  optimization_direction: null   # ✦ Optimization direction: higher, lower, or null (use per_metric_direction if null)

  # ── Dataset and metric filters ────────────────────────────────────────────
  #   If not specified or empty, all datasets/metrics will be used.
  include_datasets: ["your_dataset"]   # ✦ List of dataset names to include
  include_metrics: ["your_metric"]     # ✦ List of metric names to include

  # ── Per-metric optimization direction ─────────────────────────────────────
  per_metric_direction:
    your_metric: higher           # ✦ Set to 'higher' or 'lower' as appropriate
```

### H. Run TheAIScientist to Test

At this point, your task is fully defined for TheAIScientist. Test it:

```bash
conda activate fmlbench
python run_agent_benchmark.py --config configs/your_task.yaml \
  --model gpt-5-2025-08-07 --provider OpenAI
```

The other agents reuse the same task folder and repo.

## 2) AIDE (Weco)

Follow the same task structure defined in TheAIScientist. Add a WECO runner and reuse your repo/configs:

### A. Add `weco/run_weco.sh`

Create `ml_tasks/YourTaskName_yourrepo/weco/run_weco.sh` (copy from an existing task as template) and customize. Align fields with your `config.json` and execution flow:

```bash
# Create timestamped workspace
workspace="../../../benchmark_results/weco/YourTaskName_yourrepo/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy setup and postprocess scripts if your task uses them
cp ../../../ml_tasks/YourTaskName_yourrepo/eval.py ./ 2>/dev/null || true
cp ../../../ml_tasks/YourTaskName_yourrepo/postprocess.py ./ 2>/dev/null || true

# Define the full weco command (edit source/eval-command/metric/goal/model)
WECO_CMD='weco run \
  --source "path/to/target_file.py" \
  --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && rm -rf ./results_tmp/ && python eval.py && python postprocess.py --directory ./results_tmp/ && cp -r ./results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf ./results_tmp" \
  --metric "Your Metric Name" \
  --goal [maximize/minimize] \
  --model '"${MODEL}"' \
  --steps 100 \
  --save-logs \
  --additional-instructions "You are an ambitious AI PhD student focused on [your domain]. [Detailed task description and goals]."'

eval "$WECO_CMD"
```

What to change so it mirrors the general tutorial:
- `YourTaskName_yourrepo` → your task directory name
- Paths in the `cp` lines (only if you use `eval.py`/`postprocess.py`)
- `--source` → a file in your `target_files`
- `--eval-command` → your exact execution steps (clean, run, postprocess, copy final_info.json)
- `--metric`, `--goal`, `--steps`, and `--additional-instructions`

Test AIDE (WECO) Agent:

```bash
# Prepare resources
cp ml_tasks/YourTaskName_yourrepo/weco/run_weco.sh workspace/YourTaskName_yourrepo/run_weco.sh
chmod +x workspace/YourTaskName_yourrepo/run_weco.sh

# Run WECO agent
export CUDA_VISIBLE_DEVICES=0  # or specify which GPU to use
conda activate your_env
cd workspace/YourTaskName_yourrepo
source run_weco.sh --model gpt-5-2025-08-07
```

## 3) Claude Code

Follow the same task structure defined in TheAIScientist. Add the Claude Code starter and reuse your repo/configs:

### A. Add `claude_code/run_claude_code.sh` and `claude_code/prompt.txt`

Copy `claude_code/run_claude_code.sh` from an existing task and customize. Example starter:

```bash
# Setup for benchmarking repo (different for different repo!!!)
REPO_FOLDER_NAME="YourTaskName_yourrepo"
cp ../../../ml_tasks/YourTaskName_yourrepo/eval.py ./ 2>/dev/null || true
cp ../../../ml_tasks/YourTaskName_yourrepo/postprocess.py ./ 2>/dev/null || true
chmod 555 eval.py 2>/dev/null || true
chmod 555 postprocess.py 2>/dev/null || true
```

What to change:
- `REPO_FOLDER_NAME` → your task directory name
- File paths to copy your specific helper files (only if used)
- `chmod` lines for any scripts that you want to protect

For Prompt file `claude_code/prompt.txt`, align with the general tutorial’s “# Begin with:” fields:

```text
# Begin with:
TASK_DESCRIPTION: "You are an ambitious AI PhD student focused on [your domain]. [Detailed task description and goals]."

STARTER_FILE_PATHS: [
    "path/to/file1.py",
    "path/to/file2.py"
]

READONLY_PATHS: [
    "path/to/readonly_file.py",
    "eval.py"
]

ORIGINAL_BASELINE_RESULTS_PATH: "../../../ml_tasks/YourTaskName_yourrepo/baseline_results/final_info.json"

TARGET_METRICS: ["your_metric"]

TARGET_DATASETS:["your_dataset"]

OPTIMIZATION_DIRECTION: ""

PER_METRIC_DIRECTION = {
    "your_metric": "higher"  # or "lower"
}

COMMAND_LIST: [
    "rm -r ./results_tmp/",
    "python eval.py",
    "python postprocess.py --directory ./results_tmp/"
]

MAX_ITERS: 100

RESULT_DIR: "./results_tmp/"
```

What to change so it matches your task exactly:
- `TASK_DESCRIPTION`
- `STARTER_FILE_PATHS` (your `target_files`)
- `READONLY_PATHS` (include `eval.py` if it must be readonly)
- `ORIGINAL_BASELINE_RESULTS_PATH` (if you provide baseline)
- `TARGET_METRICS`, `TARGET_DATASETS`, `OPTIMIZATION_DIRECTION`, `PER_METRIC_DIRECTION` (as in your YAML Config under `configs/`)
- `COMMAND_LIST` steps (your `execute_commands`)
- `RESULT_DIR` (your `exp_running_dir`)

Test Claude Code Agent:

```bash
# Prepare resources
cp ml_tasks/YourTaskName_yourrepo/claude_code/run_claude_code.sh workspace/YourTaskName_yourrepo/run_claude_code.sh
cp ml_tasks/YourTaskName_yourrepo/claude_code/prompt.txt workspace/YourTaskName_yourrepo/prompt.txt
chmod +x workspace/YourTaskName_yourrepo/run_claude_code.sh

# Run Claude Code agent
export CUDA_VISIBLE_DEVICES=0  # or specify which GPU to use
conda activate your_env
cd workspace/YourTaskName_yourrepo
source run_claude_code.sh
```

That's it. Define your task once under TheAIScientist, then run it with AIDE (WECO) and Claude Code by adding their small agent-specific wrappers that follow the TheAIScientist steps.


