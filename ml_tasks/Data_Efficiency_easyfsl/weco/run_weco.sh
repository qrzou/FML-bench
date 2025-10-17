#!/bin/bash
cd easy-few-shot-learning

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
workspace="../../../benchmark_results/weco/Data_Efficiency_easyfsl/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the eval.py to the workspace
cp ../../../ml_tasks/Data_Efficiency_easyfsl/eval.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "easyfsl/methods/few_shot_classifier.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python eval.py && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "accuracy" \
    --goal maximize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on data-efficient learning, specializing in few-shot learning and meta-learning. You are working with the EasyFSL framework to enhance the FewShotClassifier on the Mini-ImageNet dataset. The Mini-ImageNet dataset presents a challenging few-shot learning scenario due to its fine-grained inter-class similarities and limited training examples per class. Your goal is to improve the classifier’s ability to generalize to novel classes."'

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