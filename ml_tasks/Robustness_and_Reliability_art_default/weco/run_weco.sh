#!/bin/bash
cd adversarial-robustness-toolbox

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
workspace="../../../benchmark_results/weco/Robustness_and_Reliability_art_default/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the eval.py, model.py, and trainer.py to the workspace
cp ../../../ml_tasks/Robustness_and_Reliability_art_default/eval.py ./
cp ../../../ml_tasks/Robustness_and_Reliability_art_default/model.py ./
cp ../../../ml_tasks/Robustness_and_Reliability_art_default/trainer.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "art/defences/trainer/dp_instahide_trainer.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python eval.py && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "Defense score:" \
    --goal maximize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on improving robust learning under data poisoning and privacy constraints. You are given the Adversarial Robustness Toolbox (ART) codebase with a focus on the dp_instahide defense. dp_instahide mixes inputs with public data and applies differential privacy noise to hinder inversion and poisoning. While designed for privacy-preserving training, its structure offers headroom to harden against both clean-label and trigger/backdoor poisons. Your goal is to improve defense performance against diverse poisoning attacks while maintaining high clean accuracy. You may tune dp_instahide, compose it with other defenses, or propose a new method if it outperforms baselines."'

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