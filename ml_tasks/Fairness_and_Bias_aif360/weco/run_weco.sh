#!/bin/bash
cd AIF360

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
workspace="../../../benchmark_results/weco/Fairness_and_Bias_aif360/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the setup files  to the workspace
cp ../../../ml_tasks/Fairness_and_Bias_aif360/train_eval_baseline.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "aif360/algorithms/inprocessing/adversarial_debiasing.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python train_eval_baseline.py && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "Test set: Absolute average odds difference =" \
    --goal minimize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on improving fairness-aware learning with AIF360’s Adversarial Debiasing on the Adult dataset. You are working with AIF360’s Adversarial Debiasing (classifier–adversary) as the baseline on the Adult dataset to evaluate the fairness–accuracy trade-off. Your goal is to minimize absolute Average Odds Difference toward parity (=0) while maintaining or improving Balanced Accuracy on held-out test splits and across protected subgroups (e.g., sex/race). You should enhance the baseline Adversarial Debiasing algorithm, but you may also propose entirely new fairness methods if they better support reduced absolute Average Odds Difference without sacrificing Balanced Accuracy. You are allowed to refine the classifier and adversary networks and the training pipeline, provided comparisons remain fair to the original setup (similar capacity, training budget, and data access). The priority is minimizing absolute Average Odds Difference while preserving or improving Balanced Accuracy."'

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