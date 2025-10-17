#!/bin/bash

## Complete Setup Script
# This script sets up the complete workspace and environments for ML benchmark tasks
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
for cmd in git curl unzip wget conda pip; do
    if ! command_exists "$cmd"; then
        error_exit "$cmd is not installed. Please install it first."
    fi
done

# Check if we're in the right directory
if [ ! -d "workspace" ]; then
    error_exit "workspace directory not found. Please run this script from the project root."
fi

echo "Starting complete setup..."

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Set conda to be non-interactive
export CONDA_ALWAYS_YES=true
export CONDA_AUTO_UPDATE_CONDA=false

# Set pip to be non-interactive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# # =============================================================================
# # 1. GENERALIZATION: DOMAINBED
# # =============================================================================
echo "Setting up Generalization: Domainbed..."

# Step 1: Git clone and download
echo "  → Cloning DomainBed repository..."
mkdir -p workspace/Generalization_domainbed
cd workspace/Generalization_domainbed
git clone https://github.com/facebookresearch/DomainBed.git
cd DomainBed
git checkout b93c22a1cfc3b2428398272c1a116c8de1f4139e
cd ../../..

# Step 2: Create environment
echo "  → Creating domainbed conda environment..."
conda create -n domainbed python=3.10 -y
conda activate domainbed
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install backpack-for-pytorch==1.3.0 numpy==1.22.4 wilds==2.0.0 tqdm==4.66.4 imageio==2.9.0 gdown==3.13.0 parameterized==0.9.0 Pillow==10.3.0 timm==0.9.16
pip install weco==0.3.0
conda deactivate

echo "✓ Generalization: Domainbed setup completed"

# =============================================================================
# 2. DATA EFFICIENCY: EASY-FEW-SHOT-LEARNING
# =============================================================================
echo "Setting up Data Efficiency: easy-few-shot-learning..."

# Step 1: Git clone and download
echo "  → Cloning easy-few-shot-learning repository..."
mkdir -p workspace/Data_Efficiency_easyfsl
cd workspace/Data_Efficiency_easyfsl
git clone https://github.com/sicara/easy-few-shot-learning.git
cd easy-few-shot-learning
git checkout 8023ff49a02a68830c10a21b8eb908cb33bdf1b9

# Download mini imagenet dataset
echo "  → Downloading mini imagenet dataset..."
mkdir -p data/mini_imagenet
cd data/mini_imagenet
curl -L -o ./miniimagenet.zip https://www.kaggle.com/api/v1/datasets/download/arjunashok33/miniimagenet
unzip miniimagenet.zip -d images
cd ../..
cd ../../..

# Step 2: Create environment
echo "  → Creating easyfsl conda environment..."
conda create -n easyfsl python=3.10 -y
conda activate easyfsl
pip install "matplotlib>=3.0.0" "pandas>=1.5.0" "torch>=1.5.0" "torchvision>=0.7.0" "tqdm>=4.1.0" tensorboard
pip install gdown typer loguru pyarrow fastparquet
pip install weco==0.3.0

# Step 3: Run Python setup commands
echo "  → Running Python setup commands..."
# Check if gdown is available
if ! command_exists gdown; then
    echo "Installing gdown..."
    pip install gdown
fi

cd workspace/Data_Efficiency_easyfsl/easy-few-shot-learning
gdown --id 1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ -O data/models/feat_resnet12_mini_imagenet.pth
python -m scripts.predict_embeddings \
          feat_resnet12 \
          data/models/feat_resnet12_mini_imagenet.pth \
          mini_imagenet \
          --device=cuda \
          --num-workers=12 \
          --batch-size=1024
cd ../../..

conda deactivate
echo "✓ Data Efficiency: easy-few-shot-learning setup completed"

# =============================================================================
# 3. REPRESENTATION LEARNING: LIGHTLY
# =============================================================================
echo "Setting up Representation Learning: lightly..."

# Step 1: Git clone and download
echo "  → Cloning lightly repository..."
mkdir -p workspace/Representation_Learning_lightly
cd workspace/Representation_Learning_lightly
git clone https://github.com/lightly-ai/lightly.git
cd lightly
git checkout 3d371ee3699e2b6d20adc4c79ac2a0fee52009ac

# Check if source files exist before copying
if [ ! -f "../../../ml_tasks/Representation_Learning_lightly/model.py" ]; then
    error_exit "Source file ../../../ml_tasks/Representation_Learning_lightly/model.py not found"
fi
if [ ! -f "../../../ml_tasks/Representation_Learning_lightly/transform.py" ]; then
    error_exit "Source file ../../../ml_tasks/Representation_Learning_lightly/transform.py not found"
fi

cp ../../../ml_tasks/Representation_Learning_lightly/model.py ./
cp ../../../ml_tasks/Representation_Learning_lightly/transform.py ./
git add .
git commit -m 'setup for agent benchmark'
cd ../../..

# Step 2: Create environment
echo "  → Creating lightly conda environment..."
conda create -n lightly python=3.10 -y
conda activate lightly
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge torchmetrics==0.11.4 "numpy<2" lightning==2.1.2 scipy einops tqdm timm scikit-learn hydra-core -y
pip install aenum
pip install weco==0.3.0
conda deactivate

echo "✓ Representation Learning: lightly setup completed"

# =============================================================================
# 4. CONTINUAL LEARNING: CONTINUAL-LEARNING
# =============================================================================
echo "Setting up Continual Learning: continual-learning..."

# Step 1: Git clone and download
echo "  → Cloning continual-learning repository..."
mkdir -p workspace/Continual_Learning_continual_learning
cd workspace/Continual_Learning_continual_learning
git clone https://github.com/GMvandeVen/continual-learning.git
cd continual-learning
git checkout 7cb0ef5a85c928c3cbae3f876f71640251f9dc79
cd ../../..

# Step 2: Create environment
echo "  → Creating continual_learning conda environment..."
conda create -n continual_learning python=3.10.4 -y
conda activate continual_learning
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
pip install matplotlib "numpy<2" scipy pandas tqdm scikit-learn
pip install weco==0.3.0
conda deactivate

echo "✓ Continual Learning: continual-learning setup completed"

# =============================================================================
# 5. CAUSALITY: CAUSALML
# =============================================================================
echo "Setting up Causality: causalml..."

# Step 1: Git clone and download
echo "  → Cloning causalml repository..."
mkdir -p workspace/Causality_causalml
cd workspace/Causality_causalml
git clone https://github.com/uber/causalml.git
cd causalml
git checkout 1a96e01f67496c3846d0e8146e5dd90ae9eb21a6
cd ../../..

# Step 2: Create environment
echo "  → Creating causalml conda environment..."
conda create -n causalml python=3.10 -y
conda activate causalml
pip install "cython<3" "scikit-learn>=1.6.0,<1.9.2" "scipy>=1.8,<1.9.2" "packaging>=21.3" "typing-extensions<4.6.0" "platformdirs>=2" "tomli>=1.1.0" "contourpy>=1.0.1" "cycler>=0.10" "fonttools>=4.22.0" "kiwisolver>=1.3.1" "pillow>=8" "python-dateutil>=2.7" "numpy>=1.18.5,<1.25.0" "pytz>=2020.1" "tzdata>=2022.7" "llvmlite>=0.44.0,<0.45"
pip install "optree<0.16.0" pandas matplotlib seaborn xgboost lightgbm sphinx sphinx_rtd_theme "sphinxcontrib-bibtex<2.0.0" nbsphinx statsmodels shap pathos
pip install tensorflow==2.10.0
pip install weco==0.3.0
conda deactivate

echo "✓ Causality: causalml setup completed"

# =============================================================================
# 6. ROBUSTNESS AND RELIABILITY: ADVERSARIAL-ROBUSTNESS-TOOLBOX
# =============================================================================
echo "Setting up Robustness and Reliability: adversarial-robustness-toolbox..."

# Step 1: Git clone and download
echo "  → Cloning adversarial-robustness-toolbox repository..."
mkdir -p workspace/Robustness_and_Reliability_art_default
cd workspace/Robustness_and_Reliability_art_default
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox
git checkout 5ddd8ef204a8352d19d4bd212a4de5d4b7ca6fa9

# Check if source files exist before copying
if [ ! -f "../../../ml_tasks/Robustness_and_Reliability_art_default/model.py" ]; then
    error_exit "Source file ../../../ml_tasks/Robustness_and_Reliability_art_default/model.py not found"
fi
if [ ! -f "../../../ml_tasks/Robustness_and_Reliability_art_default/trainer.py" ]; then
    error_exit "Source file ../../../ml_tasks/Robustness_and_Reliability_art_default/trainer.py not found"
fi

cp ../../../ml_tasks/Robustness_and_Reliability_art_default/model.py ./
cp ../../../ml_tasks/Robustness_and_Reliability_art_default/trainer.py ./
git add .
git commit -m 'setup for agent benchmark'
cd ../../..

# Step 2: Create environment
echo "  → Creating art conda environment..."
conda create -n art python=3.10 -y
conda activate art
pip install tensorflow-gpu==2.10.1
pip install numpy==1.23.5
pip install matplotlib tqdm
pip install "scipy>=1.4.1" "scikit-learn>=0.22.2"
pip install weco==0.3.0
conda deactivate

echo "✓ Robustness and Reliability: adversarial-robustness-toolbox setup completed"

# =============================================================================
# 7. PRIVACY: ML_PRIVACY_METER
# =============================================================================
echo "Setting up Privacy: ml_privacy_meter..."

# Step 1: Git clone and download
echo "  → Cloning ml_privacy_meter repository..."
mkdir -p workspace/Privacy_privacymeter
cd workspace/Privacy_privacymeter
git clone https://github.com/privacytrustlab/ml_privacy_meter.git
cd ml_privacy_meter
git checkout e384af8fd9319b8eeb1303aa82474df1441e3c59
cd ../../..

# Step 2: Create environment
echo "  → Creating privacy_meter conda environment..."
cd workspace/Privacy_privacymeter/ml_privacy_meter
conda create -n privacy_meter python=3.12 -y
conda activate privacy_meter
if [ ! -f "requirements.txt" ]; then
    error_exit "requirements.txt not found in ml_privacy_meter directory"
fi
pip install -r requirements.txt
cd ../../..
pip install weco==0.3.0
conda deactivate

echo "✓ Privacy: ml_privacy_meter setup completed"

# =============================================================================
# 8. FAIRNESS AND BIAS: AIF360
# =============================================================================
echo "Setting up Fairness and Bias: AIF360..."

# Step 1: Git clone and download
echo "  → Cloning AIF360 repository..."
mkdir -p workspace/Fairness_and_Bias_aif360
cd workspace/Fairness_and_Bias_aif360
git clone https://github.com/Trusted-AI/AIF360.git
cd AIF360
git checkout cd7e2138b7919e0796db7e7902bf49b20065f4f8

# Check if source file exists before copying
if [ ! -f "../../../ml_tasks/Fairness_and_Bias_aif360/algorithm.py" ]; then
    error_exit "Source file ../../../ml_tasks/Fairness_and_Bias_aif360/algorithm.py not found"
fi

cp ../../../ml_tasks/Fairness_and_Bias_aif360/algorithm.py ./
git add .
git commit -m 'setup for agent benchmark'

# Download adult dataset
echo "  → Downloading adult dataset..."
mkdir -p aif360/data/raw/adult
cd aif360/data/raw/adult
wget https://archive.ics.uci.edu/static/public/2/adult.zip
unzip adult.zip
cd ../../../../..
cd ../../..

# Step 2: Create environment
echo "  → Creating aif360 conda environment..."
conda create -n aif360 python=3.10 -y
conda activate aif360
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install "numpy<2" scipy pandas scikit-learn matplotlib tqdm seaborn -y
pip install lightgbm==3.1.1 "igraph[plotting]==0.9.8"
pip install lime adversarial-robustness-toolbox==1.13.0 BlackBoxAuditing tensorflow-gpu==2.10.1 "cvxpy>=1.0" "fairlearn>=0.7.0" skorch==0.11.0 "inFairness>=0.2.2" pot==0.9 mlxtend colorama
pip install "pytest>=3.5.0" "pytest-cov>=2.8.1"
pip install weco==0.3.0
conda deactivate

echo "✓ Fairness and Bias: AIF360 setup completed"

echo ""
echo "🎉 All setup completed successfully!"
echo ""
echo "Available conda environments:"
echo "  - domainbed (Generalization)"
echo "  - easyfsl (Data Efficiency)" 
echo "  - lightly (Representation Learning)"
echo "  - continual_learning (Continual Learning)"
echo "  - causalml (Causality)"
echo "  - art (Robustness and Reliability)"
echo "  - privacy_meter (Privacy)"
echo "  - aif360 (Fairness and Bias)"
echo ""
echo "To activate an environment, use: conda activate <environment_name>"