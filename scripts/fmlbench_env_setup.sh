#!/bin/bash

# =============================================================================
#        FML-bench Environment Setup
# =============================================================================

# This script sets up conda environment for FML-bench
# Exit on any error
set -e

# Function to print error and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required commands
echo "Checking required commands..."
for cmd in conda pip; do
    if ! command_exists "$cmd"; then
        error_exit "$cmd is not installed. Please install Anaconda/Miniconda first."
    fi
done

echo "Starting FML-bench environment setup..."

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Set conda to be non-interactive
export CONDA_ALWAYS_YES=true
export CONDA_AUTO_UPDATE_CONDA=false

# Set pip to be non-interactive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# FML-bench environment
echo "Setting up FML-bench environment..."
conda create -n fmlbench python=3.11 -y
conda activate fmlbench

# Install pdflatex (if needed)
# echo "Installing pdflatex..."
# if command_exists apt-get; then
#     sudo apt-get install texlive-full -y
# else
#     echo "Warning: apt-get not found. Please install texlive-full manually."
# fi

# Install PyPI requirements
echo "Installing Python packages..."
pip install aider-chat==0.85.1
pip install anthropic backoff openai google-generativeai matplotlib pypdf pymupdf4llm torch numpy transformers datasets tiktoken wandb tqdm
pip install jupyterlab

# Install Node.js and npm
echo "Installing Node.js and npm..."
if command_exists npm; then
    echo "npm already installed"
else
    pip install npm
fi
conda install -c conda-forge nodejs -y

conda deactivate
echo "✓ FML-bench environment setup completed"








