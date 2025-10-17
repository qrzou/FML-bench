# FML-bench Tasks Introduction

This document provides an overview of the 8 machine learning tasks included in the FML-bench benchmark. Each task represents a critical aspect of machine learning research and practice, designed to evaluate AI agents' capabilities across different domains.

## Task Overview

The FML-bench includes 8 comprehensive tasks covering the following areas:

1. **Generalization** - Domain adaptation and generalization
2. **Data Efficiency** - Few-shot learning and data-efficient methods
3. **Representation Learning** - Self-supervised and contrastive learning
4. **Continual Learning** - Learning without catastrophic forgetting
5. **Causality** - Causal inference and treatment effect estimation
6. **Robustness and Reliability** - Adversarial robustness and defense mechanisms
7. **Privacy** - Privacy-preserving machine learning and membership inference
8. **Fairness and Bias** - Fair machine learning and bias mitigation

---

## 1. Generalization (DomainBed)

**Repository**: [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed)  
**Environment**: `domainbed`  
**Dataset**: ColoredMNIST  

### Task Description
This task focuses on domain generalization - the ability of models to perform well on data from domains not seen during training. The challenge is to develop algorithms that can generalize across different data distributions.

### Key Components
- **Target Files**: `domainbed/algorithms.py`, `domainbed/networks.py`
- **Algorithm**: ERM (Empirical Risk Minimization) baseline
- **Evaluation**: Cross-domain performance on ColoredMNIST dataset

### Challenge
Develop domain generalization algorithms that can maintain performance when the test domain differs from training domains, addressing the fundamental challenge of distribution shift in real-world applications.

---

## 2. Data Efficiency (Easy-Few-Shot-Learning)

**Repository**: [sicara/easy-few-shot-learning](https://github.com/sicara/easy-few-shot-learning)  
**Environment**: `easyfsl`  
**Dataset**: Mini-ImageNet  

### Task Description
This task evaluates the ability to learn effectively from limited data, focusing on few-shot learning scenarios where only a few examples per class are available.

### Key Components
- **Target Files**: `easyfsl/methods/few_shot_classifier.py`, `easyfsl/methods/prototypical_networks.py`
- **Methods**: Prototypical Networks, Few-shot classifiers
- **Evaluation**: Few-shot classification performance on Mini-ImageNet

### Challenge
Design methods that can quickly adapt to new classes with minimal examples, addressing the critical need for data-efficient learning in scenarios where labeled data is scarce.

---

## 3. Representation Learning (Lightly)

**Repository**: [lightly-ai/lightly](https://github.com/lightly-ai/lightly)  
**Environment**: `lightly`  
**Dataset**: CIFAR-10  

### Task Description
This task implements self-supervised pretraining using a MoCo-style contrastive learning approach on CIFAR-10, followed by linear probing for evaluation. The goal is to learn strong visual representations without labels and assess them by training a simple classifier on top of the frozen backbone.

### Key Components
- **Target Files**: `model.py`, `transform.py`
- **Methods**: Contrastive self-supervised learning (MoCo with memory bank, momentum encoder), cosine learning rate scheduling, linear probing.
- **Evaluation**: Linear probing top-1 accuracy on CIFAR-10 test set.

### Challenge
Develop self-supervised learning methods that can learn general-purpose and transferable representations from unlabeled data. The key challenge is building robust representations using contrastive learning with limited labels, and evaluating them through linear probing performance on CIFAR-10.

---

## 4. Continual Learning (Continual-Learning)

**Repository**: [GMvandeVen/continual-learning](https://github.com/GMvandeVen/continual-learning)  
**Environment**: `continual_learning`  
**Dataset**: SplitMNIST  

### Task Description
This task addresses the challenge of learning new tasks without forgetting previously learned knowledge, known as catastrophic forgetting in neural networks.

### Key Components
- **Target Files**: `train/train_task_based.py`, `params/param_values.py`, `models/cl/continual_learner.py`
- **Methods**: Synaptic Intelligence (SI), Elastic Weight Consolidation (EWC)
- **Evaluation**: SplitMNIST continual learning benchmark

### Challenge
Implement continual learning algorithms that can sequentially learn multiple tasks while preserving knowledge from previous tasks, avoiding catastrophic forgetting.

---

## 5. Causality (CausalML)

**Repository**: [uber/causalml](https://github.com/uber/causalml)  
**Environment**: `causalml`  
**Dataset**: Synthetic causal inference dataset  

### Task Description
This task focuses on causal inference and treatment effect estimation, requiring understanding of cause-and-effect relationships in data.

### Key Components
- **Target Files**: `causalml/inference/tf/dragonnet.py`, `causalml/inference/tf/utils.py`
- **Methods**: DragonNet, Treatment Effect Estimation
- **Evaluation**: Causal effect estimation accuracy

### Challenge
Develop causal inference methods that can accurately estimate treatment effects and understand causal relationships, going beyond correlation to establish causation.

---

## 6. Robustness and Reliability (Adversarial-Robustness-Toolbox)

**Repository**: [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)  
**Environment**: `art`  
**Dataset**: CIFAR-10 with adversarial attacks  

### Task Description
This task evaluates the robustness of machine learning models against adversarial attacks and the effectiveness of defense mechanisms.

### Key Components
- **Target Files**: `art/defences/trainer/dp_instahide_trainer.py`, `model.py`
- **Methods**: Adversarial training, Defense mechanisms
- **Evaluation**: Robustness against backdoor attacks and adversarial perturbations

### Challenge
Develop robust training methods and defense mechanisms that can protect models against various types of adversarial attacks while maintaining clean accuracy.

---

## 7. Privacy (ML-Privacy-Meter)

**Repository**: [privacytrustlab/ml_privacy_meter](https://github.com/privacytrustlab/ml_privacy_meter)  
**Environment**: `privacy_meter`  
**Dataset**: CIFAR-10  

### Task Description
This task focuses on privacy-preserving machine learning, specifically evaluating membership inference attacks and developing privacy-preserving training methods.

### Key Components
- **Target Files**: `trainers/default_trainer.py`, `models/wide_resnet.py`
- **Methods**: Membership Inference Attacks (MIA), Privacy-preserving training
- **Evaluation**: Privacy leakage assessment on CIFAR-10

### Challenge
Implement privacy-preserving training methods that can protect individual data points from membership inference attacks while maintaining model utility.

---

## 8. Fairness and Bias (AIF360)

**Repository**: [Trusted-AI/AIF360](https://github.com/Trusted-AI/AIF360)  
**Environment**: `aif360`  
**Dataset**: Adult dataset  

### Task Description
This task addresses fairness and bias in machine learning, focusing on developing algorithms that can mitigate unfair treatment across different demographic groups.

### Key Components
- **Target Files**: `aif360/algorithms/inprocessing/adversarial_debiasing.py`
- **Methods**: Adversarial debiasing, Fairness-aware learning
- **Evaluation**: Fairness metrics on Adult dataset

### Challenge
Develop fair machine learning algorithms that can reduce bias and ensure equitable treatment across different demographic groups while maintaining predictive performance.

---

## Task Execution Framework

Each task follows a standardized execution framework:

1. **Environment Setup**: Dedicated conda environment with specific dependencies
2. **Preprocessing**: Copy necessary files and prepare the workspace
3. **Execution**: Run the main experiment with specified parameters
4. **Postprocessing**: Extract results and generate evaluation metrics
5. **Evaluation**: Generate `final_info.json` with performance metrics

## Evaluation Metrics

Each task is evaluated using domain-specific metrics:

- **Generalization**: `in_acc_mean` (in-domain accuracy) on ColoredMNIST test environment 2
- **Data Efficiency**: `accuracy_mean` (few-shot classification accuracy) on Mini-ImageNet test set
- **Representation Learning**: `test_acc_mean` (linear probing accuracy) on CIFAR-10
- **Continual Learning**: `average_acc_mean` (average accuracy across tasks) on SplitMNIST class scenario
- **Causality**: `abs_pct_error_of_ate_mean` (absolute percentage error of Average Treatment Effect) on synthetic test data
- **Robustness**: `defense_score_mean` (defense effectiveness score) on poisoned MNIST dataset
- **Privacy**: `AUC_mean` (Area Under Curve for membership inference attacks) on CIFAR-10
- **Fairness**: `abs_aod_mean` (absolute Average Odds Difference) on Adult dataset

## Getting Started

To run any task:

1. **Setup Environment**: Run the workspace setup script
2. **Activate Environment**: Use the appropriate conda environment
3. **Set GPU**: Export `CUDA_VISIBLE_DEVICES` if using GPU
4. **Run Task**: Execute the task-specific commands

For detailed setup instructions, see the main [README.md](README.md).

---

*This benchmark provides a comprehensive evaluation framework for AI agents across critical machine learning challenges, enabling systematic assessment of capabilities in generalization, efficiency, robustness, privacy, and fairness.*
