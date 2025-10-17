#!/bin/bash
cd ml_privacy_meter

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
workspace="../../../benchmark_results/weco/Privacy_privacymeter/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the cifar10_benchmark.yaml and postprocess.py to the workspace
cp ../../../ml_tasks/Privacy_privacymeter/cifar10_benchmark.yaml ./configs/
cp ../../../ml_tasks/Privacy_privacymeter/postprocess.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "trainers/default_trainer.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python run_mia.py --cf configs/cifar10_benchmark.yaml && python postprocess.py --exp_dir results_tmp && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "AUC_mean:" \
    --goal minimize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on improving model privacy and security against membership inference attacks. You are working with PrivacyMeter’s MIA (for information leakage through training points) and Robust MIA (RMIA, which refines the Likelihood Ratio Test with a tighter null hypothesis and leverages reference models and population data) to evaluate and reduce the model’s privacy risk. Your goal is to drive the auditor’s AUC toward 0.5 and keep TPR@0.1%FPR and TPR@0.0%FPR near zero while preserving task accuracy. Focus only on defense-side strategies rather than modifying the attack algorithms."'

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