
# =============================================================================
# 1. GENERALIZATION: DOMAINBED
# =============================================================================
cp ml_tasks/Generalization_domainbed/weco/run_weco.sh workspace/Generalization_domainbed/run_weco.sh
chmod +x workspace/Generalization_domainbed/run_weco.sh
conda activate domainbed
cd workspace/Generalization_domainbed
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../


# =============================================================================
# 2. DATA EFFICIENCY: EASY-FEW-SHOT-LEARNING
# =============================================================================
cp ml_tasks/Data_Efficiency_easyfsl/weco/run_weco.sh workspace/Data_Efficiency_easyfsl/run_weco.sh
chmod +x workspace/Data_Efficiency_easyfsl/run_weco.sh
conda activate easyfsl
cd workspace/Data_Efficiency_easyfsl
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../

# =============================================================================
# 3. REPRESENTATION LEARNING: LIGHTLY
# =============================================================================
cp ml_tasks/Representation_Learning_lightly/weco/run_weco.sh workspace/Representation_Learning_lightly/run_weco.sh
chmod +x workspace/Representation_Learning_lightly/run_weco.sh
conda activate lightly
cd workspace/Representation_Learning_lightly
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../

# =============================================================================
# 4. CONTINUAL LEARNING: CONTINUAL-LEARNING
# =============================================================================
cp ml_tasks/Continual_Learning_continual_learning/weco/run_weco.sh workspace/Continual_Learning_continual_learning/run_weco.sh
chmod +x workspace/Continual_Learning_continual_learning/run_weco.sh
conda activate continual_learning
cd workspace/Continual_Learning_continual_learning
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../

# =============================================================================
# 5. CAUSALITY: CAUSALML
# =============================================================================
cp ml_tasks/Causality_causalml/weco/run_weco.sh workspace/Causality_causalml/run_weco.sh
chmod +x workspace/Causality_causalml/run_weco.sh
conda activate causalml
cd workspace/Causality_causalml
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../

# =============================================================================
# 6. ROBUSTNESS AND RELIABILITY: ADVERSARIAL-ROBUSTNESS-TOOLBOX
# =============================================================================
cp ml_tasks/Robustness_and_Reliability_art_default/weco/run_weco.sh workspace/Robustness_and_Reliability_art_default/run_weco.sh
chmod +x workspace/Robustness_and_Reliability_art_default/run_weco.sh
conda activate art
cd workspace/Robustness_and_Reliability_art_default
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../

# =============================================================================
# 7. PRIVACY: ML_PRIVACY_METER
# =============================================================================
cp ml_tasks/Privacy_privacymeter/weco/run_weco.sh workspace/Privacy_privacymeter/run_weco.sh
chmod +x workspace/Privacy_privacymeter/run_weco.sh
conda activate privacy_meter
cd workspace/Privacy_privacymeter
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../

# =============================================================================
# 8. FAIRNESS AND BIAS: AIF360
# =============================================================================
cp ml_tasks/Fairness_and_Bias_aif360/weco/run_weco.sh workspace/Fairness_and_Bias_aif360/run_weco.sh
chmod +x workspace/Fairness_and_Bias_aif360/run_weco.sh
conda activate aif360
cd workspace/Fairness_and_Bias_aif360
# gpt-5
source run_weco.sh --model gpt-5
# gemini-2.5-pro
source run_weco.sh --model gemini-2.5-pro
cd ../../


