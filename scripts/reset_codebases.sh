#!/usr/bin/env bash

# Reset all workspace codebases to the exact commits used in workspace_and_env_setup.sh
# Safe to re-run; will hard reset and clean each repo working tree.

set -euo pipefail

error() {
    echo "ERROR: $1" >&2
}

info() {
    echo "[reset] $1"
}

reset_repo() {
    local repo_dir="$1"
    local commit_hash="$2"
    local excludes="${3:-}"

    if [ ! -d "$repo_dir/.git" ]; then
        error "Git repo not found at: $repo_dir (skipping)"
        return 0
    fi

    info "Resetting $repo_dir to $commit_hash"
    pushd "$repo_dir" >/dev/null
    git fetch --all --tags --prune || true
    git reset --hard "$commit_hash"
    # Build clean command with optional excludes (space-separated patterns)
    local clean_cmd=(git clean -fdx)
    if [ -n "$excludes" ]; then
        for pat in $excludes; do
            clean_cmd+=( -e "$pat" )
        done
    fi
    "${clean_cmd[@]}"
    # Keep submodules consistent if any
    if [ -f .gitmodules ]; then
        git submodule sync --recursive || true
        git submodule update --init --recursive || true
    fi
    popd >/dev/null
}

# Verify we are in project root (expect workspace dir present)
if [ ! -d "workspace" ]; then
    error "workspace directory not found. Run from project root."
    exit 1
fi

echo "Starting workspace codebase reset..."

# 1) Generalization: DomainBed (keep downloaded datasets under DomainBed/data/)
reset_repo "workspace/Generalization_domainbed/DomainBed" "b93c22a1cfc3b2428398272c1a116c8de1f4139e" ""

# 2) Data Efficiency: easy-few-shot-learning (keep Mini-ImageNet under data/ and model cache)
reset_repo "workspace/Data_Efficiency_easyfsl/easy-few-shot-learning" "8023ff49a02a68830c10a21b8eb908cb33bdf1b9" "data/"

# 3) Representation Learning: lightly (keep datasets/)
reset_repo "workspace/Representation_Learning_lightly/lightly" "3d371ee3699e2b6d20adc4c79ac2a0fee52009ac" "datasets/"

#    Re-apply setup file copies and commit (as in workspace_and_env_setup.sh)
if [ -d "workspace/Representation_Learning_lightly/lightly/.git" ]; then
    info "Re-applying setup files for lightly"
    pushd workspace/Representation_Learning_lightly/lightly >/dev/null
    if [ -f "../../../ml_tasks/Representation_Learning_lightly/model.py" ] && [ -f "../../../ml_tasks/Representation_Learning_lightly/transform.py" ]; then
        cp ../../../ml_tasks/Representation_Learning_lightly/model.py ./
        cp ../../../ml_tasks/Representation_Learning_lightly/transform.py ./
        git add .
        git commit -m 'setup for agent benchmark' || true
    fi
    popd >/dev/null
fi

# 4) Continual Learning: continual-learning (preserve datasets under store/)
reset_repo "workspace/Continual_Learning_continual_learning/continual-learning" "7cb0ef5a85c928c3cbae3f876f71640251f9dc79" "store/"

# 5) Causality: causalml
reset_repo "workspace/Causality_causalml/causalml" "1a96e01f67496c3846d0e8146e5dd90ae9eb21a6" ""

# 6) Robustness and Reliability: adversarial-robustness-toolbox
reset_repo "workspace/Robustness_and_Reliability_art_default/adversarial-robustness-toolbox" "5ddd8ef204a8352d19d4bd212a4de5d4b7ca6fa9" ""

#    Re-apply setup file copies and commit (as in workspace_and_env_setup.sh)
if [ -d "workspace/Robustness_and_Reliability_art_default/adversarial-robustness-toolbox/.git" ]; then
    info "Re-applying setup files for adversarial-robustness-toolbox"
    pushd workspace/Robustness_and_Reliability_art_default/adversarial-robustness-toolbox >/dev/null
    if [ -f "../../../ml_tasks/Robustness_and_Reliability_art_default/model.py" ] && [ -f "../../../ml_tasks/Robustness_and_Reliability_art_default/trainer.py" ]; then
        cp ../../../ml_tasks/Robustness_and_Reliability_art_default/model.py ./
        cp ../../../ml_tasks/Robustness_and_Reliability_art_default/trainer.py ./
        git add .
        git commit -m 'setup for agent benchmark' || true
    fi
    popd >/dev/null
fi

# 7) Privacy: ml_privacy_meter (preserve data/)
reset_repo "workspace/Privacy_privacymeter/ml_privacy_meter" "e384af8fd9319b8eeb1303aa82474df1441e3c59" "data/"

# 8) Fairness and Bias: AIF360 (preserve downloaded adult dataset under aif360/data/raw/adult)
reset_repo "workspace/Fairness_and_Bias_aif360/AIF360" "cd7e2138b7919e0796db7e7902bf49b20065f4f8" "aif360/data/"

#    Re-apply setup file copies and commit (as in workspace_and_env_setup.sh)
if [ -d "workspace/Fairness_and_Bias_aif360/AIF360/.git" ]; then
    info "Re-applying setup files for AIF360"
    pushd workspace/Fairness_and_Bias_aif360/AIF360 >/dev/null
    if [ -f "../../../ml_tasks/Fairness_and_Bias_aif360/algorithm.py" ]; then
        cp ../../../ml_tasks/Fairness_and_Bias_aif360/algorithm.py ./
        git add .
        git commit -m 'setup for agent benchmark' || true
    fi
    popd >/dev/null
fi

echo "Done. All codebases reset to pinned commits."


