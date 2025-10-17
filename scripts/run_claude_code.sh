# =============================================================================
# 1. GENERALIZATION: DOMAINBED
# =============================================================================
cp ml_tasks/Generalization_domainbed/claude_code/run_claude_code.sh workspace/Generalization_domainbed/run_claude_code.sh
cp ml_tasks/Generalization_domainbed/claude_code/prompt.txt workspace/Generalization_domainbed/prompt.txt
chmod +x workspace/Generalization_domainbed/start.sh
conda activate domainbed
cd workspace/Generalization_domainbed
source run_claude_code.sh
cd ../../

# =============================================================================
# 2. DATA EFFICIENCY: EASY-FEW-SHOT-LEARNING
# =============================================================================
cp ml_tasks/Data_Efficiency_easyfsl/claude_code/run_claude_code.sh workspace/Data_Efficiency_easyfsl/run_claude_code.sh
cp ml_tasks/Data_Efficiency_easyfsl/claude_code/prompt.txt workspace/Data_Efficiency_easyfsl/prompt.txt
chmod +x workspace/Data_Efficiency_easyfsl/start.sh
conda activate easyfsl
cd workspace/Data_Efficiency_easyfsl
source run_claude_code.sh
cd ../../

# =============================================================================
# 3. REPRESENTATION LEARNING: LIGHTLY
# =============================================================================
cp ml_tasks/Representation_Learning_lightly/claude_code/run_claude_code.sh workspace/Representation_Learning_lightly/run_claude_code.sh
cp ml_tasks/Representation_Learning_lightly/claude_code/prompt.txt workspace/Representation_Learning_lightly/prompt.txt
chmod +x workspace/Representation_Learning_lightly/start.sh
conda activate lightly
cd workspace/Representation_Learning_lightly
source run_claude_code.sh
cd ../../

# =============================================================================
# 4. CONTINUAL LEARNING: CONTINUAL-LEARNING
# =============================================================================
cp ml_tasks/Continual_Learning_continual_learning/claude_code/run_claude_code.sh workspace/Continual_Learning_continual_learning/run_claude_code.sh
cp ml_tasks/Continual_Learning_continual_learning/claude_code/prompt.txt workspace/Continual_Learning_continual_learning/prompt.txt
chmod +x workspace/Continual_Learning_continual_learning/start.sh
conda activate continual_learning
cd workspace/Continual_Learning_continual_learning
source run_claude_code.sh
cd ../../

# =============================================================================
# 5. CAUSALITY: CAUSALML
# =============================================================================
cp ml_tasks/Causality_causalml/claude_code/run_claude_code.sh workspace/Causality_causalml/run_claude_code.sh
cp ml_tasks/Causality_causalml/claude_code/prompt.txt workspace/Causality_causalml/prompt.txt
chmod +x workspace/Causality_causalml/start.sh
conda activate causalml
cd workspace/Causality_causalml
source run_claude_code.sh
cd ../../

# =============================================================================
# 6. ROBUSTNESS AND RELIABILITY: ADVERSARIAL-ROBUSTNESS-TOOLBOX
# =============================================================================
cp ml_tasks/Robustness_and_Reliability_art_default/claude_code/run_claude_code.sh workspace/Robustness_and_Reliability_art_default/run_claude_code.sh
cp ml_tasks/Robustness_and_Reliability_art_default/claude_code/prompt.txt workspace/Robustness_and_Reliability_art_default/prompt.txt
chmod +x workspace/Robustness_and_Reliability_art_default/start.sh
conda activate art
cd workspace/Robustness_and_Reliability_art_default
source run_claude_code.sh
cd ../../

# =============================================================================
# 7. PRIVACY: ML_PRIVACY_METER
# =============================================================================
cp ml_tasks/Privacy_privacymeter/claude_code/run_claude_code.sh workspace/Privacy_privacymeter/run_claude_code.sh
cp ml_tasks/Privacy_privacymeter/claude_code/prompt.txt workspace/Privacy_privacymeter/prompt.txt
chmod +x workspace/Privacy_privacymeter/start.sh
conda activate privacy_meter
cd workspace/Privacy_privacymeter
source run_claude_code.sh
cd ../../

# =============================================================================
# 8. FAIRNESS AND BIAS: AIF360
# =============================================================================
cp ml_tasks/Fairness_and_Bias_aif360/claude_code/run_claude_code.sh workspace/Fairness_and_Bias_aif360/run_claude_code.sh
cp ml_tasks/Fairness_and_Bias_aif360/claude_code/prompt.txt workspace/Fairness_and_Bias_aif360/prompt.txt
chmod +x workspace/Fairness_and_Bias_aif360/start.sh
conda activate aif360
cd workspace/Fairness_and_Bias_aif360
source run_claude_code.sh
cd ../../


