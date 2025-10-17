#!/bin/bash
cd continual-learning

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
workspace="../../../benchmark_results/weco/Continual_Learning_continual_learning/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the main.py and options.py to the workspace
cp ../../../ml_tasks/Continual_Learning_continual_learning/main_benchmark.py ./main.py
cp ../../../ml_tasks/Continual_Learning_continual_learning/options_benchmark.py ./params/options.py

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "models/cl/continual_learner.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python main.py --experiment=splitMNIST --scenario=class --si && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "=> average accuracy over all 5 contexts" \
    --goal maximize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on improving continual learning based on Synaptic Intelligence (SI) on splitMNIST under the class-incremental scenario. You are working with the Continual-Learning repo’s SI baseline to improve accuracy and reduce forgetting on splitMNIST (5 tasks × 2 classes, single-head over 10 classes). Your goal is to improve average accuracy over all 5 contexts on splitMNIST without unfair model size or compute advantages. You should improve SI method, but are also allowed to add lightweight fair components , or propose new methods, as long as your modifications are fair (stay within fairness computation budgets). The priority is to improve the average accuracy."'

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