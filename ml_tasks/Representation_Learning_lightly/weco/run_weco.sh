#!/bin/bash
cd lightly

# reset to the original one
git restore . && git clean -fd

# remove the results_tmp folder
rm -rf results_tmp

# Parse command-line args
# Supports: --model <name> or --model=<name>; defaults to gpt-5
MODEL="gpt-5"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift 1
            ;;
        *)
            shift 1
            ;;
    esac
done

# Create timestamped workspace
workspace="../../../benchmark_results/weco/Representation_Learning_lightly/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the files to the workspace
cp ../../../ml_tasks/Representation_Learning_lightly/model.py ./
cp ../../../ml_tasks/Representation_Learning_lightly/transform.py ./
cp ../../../ml_tasks/Representation_Learning_lightly/train_eval_baseline.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "model.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python train_eval_baseline.py && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "Final test results" \
    --goal maximize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on improving representation learning on CIFAR-10 using the Lightly self-supervised learning framework. You are working with Lightly’s MoCo baseline on CIFAR-10, evaluated strictly by linear probing Top-1 accuracy. Your goal is to improve representation learning at pretrain stage to improve linear-probe accuracy on the CIFAR-10 test set beyond standard MoCo as much as you can under the same compute and data (no external data). You may modify MoCo or propose new self-supervised methods if they can yield better representations, as long as your modifications are fair compared to the original architecture. You are also allowed to refine the ResNet-18 backbone as long as parameter count and FLOPs remain comparable to the baseline. Pretrain on the CIFAR-10 train split without labels, fit the linear classifier on the same train split, and report Top-1 on the test split with priority on improving representation learning performance."'

# Save the config (exact command) into workspace
echo "$WECO_CMD" > "$workspace/config_used.txt"

# Actually run it
eval "$WECO_CMD"

# Save end time
end_time=$(date +%Y%m%d_%H%M%S)
echo "End time: $end_time" > "$workspace/end_time.txt"

# reset to the original one
git restore . && git clean -fd

cd ..