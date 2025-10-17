## Run Agents
### TheAIScientist
Example of running Generalization task using TheAIScientist:

**First, set up your API keys:**
```bash
# LLM API
export OPENAI_API_KEY="your_openai_api_key_here"
# OR
export GEMINI_API_KEY="your_gemini_api_key_here"
# OR
export OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Semantic Scholar API for TheAIScientist idea generation
export S2_API_KEY="your_s2_api_key_here"
```

**Then run TheAIScientist:**
```bash
export CUDA_VISIBLE_DEVICES=0  # or specify which GPU to use
conda activate fmlbench
python run_agent_benchmark.py --config configs/generalization.yaml
```

**Note**: 
- You need to set at least one of the API keys (e.g. OpenAI or Gemini) before running TheAIScientist.

- If you want to change LLM and provider (Geimini and OpenRouter), change the configurations in .yaml file for each task. GEMINI_API_KEY="your_api_key" and OPENROUTER_API_KEY="your_api_key" need to be set correspondingly.



### AIDE (Weco)
Example of running Generalization task using AIDE

- Prepare resources
```bash
cp ml_tasks/Generalization_domainbed/weco/run_weco.sh workspace/Generalization_domainbed/run_weco.sh
chmod +x workspace/Generalization_domainbed/run_weco.sh
```
- Run AIDE agent:
```bash
export CUDA_VISIBLE_DEVICES=0  # or specify which GPU to use
conda activate domainbed
cd workspace/Generalization_domainbed
source run_weco.sh
```

Note: Weco (AIDE's cloud-based commercial variant) needs to be installed. It should already be available in the relevant conda environments if you ran `workspace_and_env_setup.sh`.


### Claude Code
Example of running Generalization task:
- Prepare resources
```bash
cp ml_tasks/Generalization_domainbed/claude_code/run_claude_code.sh workspace/Generalization_domainbed/run_claude_code.sh
cp ml_tasks/Generalization_domainbed/claude_code/prompt.txt workspace/Generalization_domainbed/prompt.txt
chmod +x workspace/Generalization_domainbed/run_claude_code.sh
```
- Run Claude Code agent:
```bash
export CUDA_VISIBLE_DEVICES=0  # or specify which GPU to use
conda activate domainbed
cd workspace/Generalization_domainbed
source run_claude_code.sh
```

Note: Claude Code needs to be installed.