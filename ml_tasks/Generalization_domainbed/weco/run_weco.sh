#!/bin/bash
cd DomainBed

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
workspace="../../../benchmark_results/weco/Generalization_domainbed/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$workspace"

# Copy the postprocess.py to the workspace
cp ../../../ml_tasks/Generalization_domainbed/postprocess.py ./

# Define the full weco command as a variable
WECO_CMD='weco run \
    --source "domainbed/algorithms.py" \
    --eval-command "timestamp=\$(date +%Y%m%d_%H%M%S) && mkdir -p '"${workspace}"'/execution_\$timestamp && cp -r .runs '"${workspace}"' && python -m domainbed.scripts.train --data_dir=../data/ --algorithm ERM --dataset ColoredMNIST --test_env 2 --output_dir ./results_tmp && python postprocess.py --directory ./results_tmp/ && cp -r results_tmp/final_info.json '"${workspace}"'/execution_\$timestamp && rm -rf results_tmp" \
    --metric "ENV2 In-domain Accuracy" \
    --goal maximize \
    --model '"${MODEL}"' \
    --steps 100 \
    --save-logs \
    --additional-instructions "You are an ambitious AI PhD student focused on improving the generalization performance of machine learning methods using the DomainBed benchmark. You are working with DomainBed’s ERM (Empirical Risk Minimization) method as the baseline on ColoredMNIST to evaluate generalization under distribution shifts. Your goal is to enhance test-time domain generalization accuracy beyond standard ERM. You should improve the algorithm based on ERM, but you may also propose entirely new algorithms if they can better support cross-domain generalization. You are also allowed to refine the backbone model, as long as your modifications are fair compared to the original architecture. The priority is to improve the average accuracy on unseen test domains while maintaining accuracy on in-domain tests, along with ensuring efficiency and low complexity."'

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