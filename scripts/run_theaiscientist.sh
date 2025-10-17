# ------------------------------
# 1. Generalization (DomainBed)
# ------------------------------
python run_agent_benchmark.py --config configs/generalization.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/generalization.yaml --model gemini-2.5-pro --provider Google

# ------------------------------
# 2. Data Efficiency (EasyFSL)
# ------------------------------
python run_agent_benchmark.py --config configs/data_efficiency.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/data_efficiency.yaml --model gemini-2.5-pro --provider Google

# ----------------------------------------
# 3. Representation Learning (Lightly SSL)
# ----------------------------------------
python run_agent_benchmark.py --config configs/representation_learning.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/representation_learning.yaml --model gemini-2.5-pro --provider Google

# ------------------------------------
# 4. Continual Learning (CL Baselines)
# ------------------------------------
python run_agent_benchmark.py --config configs/continual_learning.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/continual_learning.yaml --model gemini-2.5-pro --provider Google

# ------------------------
# 5. Causality (causalml)
# ------------------------
python run_agent_benchmark.py --config configs/causality.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/causality.yaml --model gemini-2.5-pro --provider Google

# -------------------------------------------------------
# 6. Robustness & Reliability (Adversarial Robustness TB)
# -------------------------------------------------------
python run_agent_benchmark.py --config configs/robustness_and_reliability.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/robustness_and_reliability.yaml --model gemini-2.5-pro --provider Google

# ------------------------------
# 7. Privacy (ML Privacy Meter)
# ------------------------------
python run_agent_benchmark.py --config configs/privacy.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/privacy.yaml --model gemini-2.5-pro --provider Google

# ------------------------------
# 8. Fairness & Bias (AIF360)
# ------------------------------
python run_agent_benchmark.py --config configs/fairness_and_bias.yaml --model gpt-5-2025-08-07 --provider OpenAI
python run_agent_benchmark.py --config configs/fairness_and_bias.yaml --model gemini-2.5-pro --provider Google
