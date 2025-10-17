#!/bin/bash
cd causalml

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
            # Ignore unknown args to stay compatible with callers
            shift 1
            ;;
    esac
done

# Create timestamped workspace
workspace="../../../benchmark_results/weco/Causality_causalml/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the train.py to the workspace
cp ../../../ml_tasks/Causality_causalml/train.py ./

# Copy the postprocess.py to the workspace
cp ../../../ml_tasks/Causality_causalml/postprocess.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "causalml/inference/tf/dragonnet.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python train.py && python postprocess.py && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "Synthetic Abs % Error of ATE" \
    --goal minimize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on advancing machine learning for causal inference, reasoning, and interpretable modeling. You are working with the Dragonnet framework to estimate individual treatment effects (ITEs) in both real (IHDP) and simulated data scenarios. The simulated data follows Setup A from Nie & Wager (2018), featuring difficult nuisance functions (e.g., propensity scores) but simple, easily identifiable treatment effects. Explore modifications to Dragonnet that enhance robustness to nuisance component misspecification and improve counterfactual prediction under covariate shift. Your goal is to improve the precision of treatment effect estimation across both IHDP and simulated benchmarks."'

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