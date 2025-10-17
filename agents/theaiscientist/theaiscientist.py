"""
TheAIScientist agent implementation.
Handles the complete research workflow: task description → idea generation → coding → experimentation.
Uses Aider as a tool for code modification.
"""
import os
import json
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from aider.repo import GitRepo

# from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from .generate_ideas import generate_ideas, check_idea_novelty
from .llm import create_client
from benchmark.executor import BenchmarkExecutor
from benchmark.utils import get_filtered_results_for_prompt

from ..base import BaseAgent, AgentConfig, AgentType


# No hardcoded constants - all values must come from config


class TheAIScientistAgent(BaseAgent):
    """
    TheAIScientist agent - an autonomous AI research agent.
    Executes complete pipeline: generate idea → modify code → run experiment → collect results
    Uses Aider as a code modification tool.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.coder: Optional[Coder] = None
        self.io: Optional[InputOutput] = None
        self.repo: Optional[GitRepo] = None
        self.client = None  # LLM client for idea generation
        self.executor = None  # BenchmarkExecutor for running experiments
        self.task_description: Optional[str] = None
        self.current_idea: Optional[Dict[str, Any]] = None
        self.baseline_results: Dict[str, Any] = {}
        self.target_files: List[str] = []
        self.read_only_files: List[str] = []
        self.notes_path: Optional[str] = None
        self.workspace: Optional[str] = None
        self.experiment_timestamp: Optional[str] = None  # Parent timestamp for all ideas
        self.parent_workspace: Optional[str] = None  # Parent workspace directory
        
        # Token usage tracking
        self.token_usage_log = []
        
    def initialize(self) -> None:
        """Initialize TheAIScientist with necessary components"""
        # Initialize LLM client for idea generation
        provider = self.config.provider
        self.client, _ = create_client(self.config.model, provider)  # _ is same to self.config.model
        
        # Initialize agent state
        # self.state.data["phase"] = "generate_idea"  # Phase tracking not currently used
        self.state.data["ideas"] = []
        self.state.data["current_run"] = 1
        
        # Get max_runs from agent_params, raise error if not set
        if "max_runs" not in self.config.agent_params:
            raise ValueError("max_runs not specified in agent configuration")
        self.state.data["max_runs"] = self.config.agent_params["max_runs"]
        self.state.data["results"] = {}
        self.state.data["filtered_results"] = {}

        # Global iteration cap across all ideas (runs + retries)
        max_retries_per_run = self.config.agent_params.get("max_retries_per_run")
        if max_retries_per_run is None:
            raise ValueError("max_retries_per_run not specified in agent configuration")
        self.state.data["max_retries_per_run"] = max_retries_per_run
        max_iter = self.config.agent_params.get("max_iter")
        self.state.data["max_iter"] = max_iter if max_iter is not None else float("inf")
        self.state.data["total_iter"] = 0
        self.state.data["use_early_completion"] = self.config.agent_params.get("use_early_completion", True)
        if self.config.agent_params.get("prepare_setup_files_before_gen_ideas", False):
            print("Preparing setup files before generating ideas")
            print("Even though this function is supported, but currently we don't use it")
            # self.prepare_setup_files()  # Currently we don't use it
            
    def prepare_setup_files(self):
        """Prepare setup files before generating ideas"""
        benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        repo_dir = benchmark_config.get("repo_dir")
        conda_env = benchmark_config.get("conda_env")
        if benchmark_config.get("prepare_setup_files"):
            print("Start to do preparing setup files...")
            setup_commands = benchmark_config.get("setup_commands", [])

            import os.path as osp
            import subprocess
            from subprocess import TimeoutExpired
            from benchmark.executor import SubprocessResult
            def run_commands(commands: List[str], timeout: int = None) -> SubprocessResult:
                """Execute command list (including conda environment handling)"""
                if not commands:
                    raise ValueError("No commands provided. Please check your benchmark configuration.")
                
                # Use provided timeout (already set by caller with appropriate defaults)
                if timeout is None:
                    timeout = 300  # Default 5 minutes if not specified
                
                cwd = osp.abspath(repo_dir)
                all_stderr = ""
                sum_returncode = 0
                
                # Process commands with conda environment
                for cmd in commands:
                    # Split command and prepend conda run
                    cmd_parts = cmd.split()
                    full_cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd_parts
                        
                    try:
                        print(f"Running command: {' '.join(full_cmd)}")
                        print(f"Current working directory: {cwd}")
                        
                        result = subprocess.run(
                            full_cmd,
                            cwd=cwd,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=timeout
                        )
                        
                        all_stderr += result.stderr
                        sum_returncode += abs(result.returncode)
                        
                        if result.returncode != 0:
                            print(f"Command failed with return code {result.returncode}")
                            print(f"Error: {result.stderr}")
                            break
                            
                    except TimeoutExpired:
                        print(f"Command timed out after {timeout} seconds")
                        return SubprocessResult(1, stderr=f"Timeout after {timeout} seconds")
                    except Exception as e:
                        print(f"Error running command: {e}")
                        return SubprocessResult(1, stderr=str(e))
                        
                return SubprocessResult(sum_returncode, stderr=all_stderr)
            
            if setup_commands:
                print(f"Running setup_commands: {setup_commands}")
                setup_result = run_commands(
                    setup_commands,
                    timeout=300,  # 5 minutes is enough for preparing setup files
                )
                if setup_result.returncode != 0:
                    print("="*60)
                    print("!!! WARNING: prepare setup files failed !!!")
                    print(f"Error details: {setup_result.stderr}")
                    print("="*60)
            else:
                print("No setup commands provided. Skip preparing setup files.")

    
    def log_token_usage(self, usage_info: Dict[str, Any], step_name: str, idea_key: Optional[str] = None):
        """Log token usage information"""
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model,
            "provider": self.config.provider,
            "prompt_tokens": usage_info.get("prompt_tokens", 0),
            "completion_tokens": usage_info.get("completion_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
            "step_name": step_name,
            "idea_key": idea_key,
            "run_number": self.state.data.get("current_run"),
            "iteration": self.state.data.get("total_iter")
        }
        
        # Add cost information if available
        if "message_cost" in usage_info:
            usage_entry["message_cost"] = usage_info["message_cost"]
        if "session_cost" in usage_info:
            usage_entry["session_cost"] = usage_info["session_cost"]
            
        self.token_usage_log.append(usage_entry)
        
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get summary of token usage"""
        if not self.token_usage_log:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "calls_by_step": {},
                "tokens_by_step": {}
            }
        
        summary = {
            "total_calls": len(self.token_usage_log),
            "total_prompt_tokens": sum(entry["prompt_tokens"] for entry in self.token_usage_log),
            "total_completion_tokens": sum(entry["completion_tokens"] for entry in self.token_usage_log),
            "total_tokens": sum(entry["total_tokens"] for entry in self.token_usage_log),
            "calls_by_step": {},
            "tokens_by_step": {},
            "calls_by_idea": {},
            "tokens_by_idea": {}
        }
        
        # Add cost totals if available
        total_message_cost = sum(entry.get("message_cost", 0) for entry in self.token_usage_log)
        total_session_cost = sum(entry.get("session_cost", 0) for entry in self.token_usage_log)
        if total_message_cost > 0 or total_session_cost > 0:
            summary["total_message_cost"] = total_message_cost
            summary["total_session_cost"] = total_session_cost
        
        # Aggregate by step and idea
        for entry in self.token_usage_log:
            step = entry["step_name"]
            idea = entry.get("idea_key", "unknown")
            
            # By step
            summary["calls_by_step"][step] = summary["calls_by_step"].get(step, 0) + 1
            summary["tokens_by_step"][step] = summary["tokens_by_step"].get(step, 0) + entry["total_tokens"]
            
            # By idea
            summary["calls_by_idea"][idea] = summary["calls_by_idea"].get(idea, 0) + 1
            summary["tokens_by_idea"][idea] = summary["tokens_by_idea"].get(idea, 0) + entry["total_tokens"]
        
        return summary
        
    def run(self, 
            task_description: Optional[str] = None,
            target_files: Optional[List[str]] = None,
            baseline_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete research pipeline.
        
        Args:
            task_description: Research task description (used for idea generation)
            target_files: List of files the agent can modify
            baseline_results: Baseline experiment results
            
        Returns:
            Final results dictionary with all ideas' results
        """
        # Store inputs
        self.task_description = task_description
        self.target_files = target_files or []
        self.target_code_files = self.target_files.copy()
        self.baseline_results = baseline_results or {}
        self.baseline_results_filtered = get_filtered_results_for_prompt(self.baseline_results, self.config.runtime_params.get("metrics", {}))
        
        # Get benchmark configuration from runtime params
        self.benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        self.read_only_files = self.benchmark_config.get("read_only_files", [])
        self.agent_name = self.config.runtime_params.get("agent_name", "agent")
        self.benchmark_name = self.config.runtime_params.get("benchmark_name", "benchmark")
        
        # Add notes.txt to target files if repo_dir is available
        repo_dir = self.config.runtime_params.get("repo_dir")
        if repo_dir:
            self.notes_path = os.path.join(repo_dir, "notes.txt")
            if self.notes_path not in self.target_files:
                self.target_files.append(self.notes_path)
        
        # Create parent timestamp for this benchmark run
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.parent_workspace = os.path.join("benchmark_results", self.agent_name, self.benchmark_name, self.experiment_timestamp)
        os.makedirs(self.parent_workspace, exist_ok=True)
        print(f"Parent workspace for this benchmark run: {self.parent_workspace}")
        
        # Generate or load ideas
        # self.state.data["phase"] = "generate_idea"  # Phase tracking not currently used
        self._generate_ideas()
        
        # Save all generated ideas to parent workspace
        self._save_all_ideas_to_parent()
        
        # Execute experiments for all ideas
        all_results = {}
        ideas = self.state.data.get("ideas", [])
        
        for idx, idea_to_test in enumerate(ideas):
            # Check global iteration budget before starting a new idea
            if self.state.data.get("total_iter") >= self.state.data.get("max_iter"):
                print("Global max_iter reached. Stopping further ideas.")
                break
            
            print(f"\n{'='*60}")
            print(f"Testing idea {idx + 1}/{len(ideas)}: {idea_to_test.get('Title', 'Untitled')}")
            print(f"{'='*60}\n")
            
            # Reset state for this idea
            self.current_idea = idea_to_test
            self.state.data["current_run"] = 1
            self.state.data["results"] = {}
            self.state.data["filtered_results"] = {}
            self.state.completed = False
            
            # Create experiment name with timestamp and idea name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            idea_name = self.current_idea.get('Name', f'idea_{idx}')
            # Sanitize idea name for filesystem
            idea_name = "".join(c for c in idea_name if c.isalnum() or c in ('_', '-')).rstrip()
            experiment_name = f"{timestamp}_{idea_name}"
            
            # Create executor for this idea with parent timestamp and agent params
            self.executor = BenchmarkExecutor(
                self.benchmark_config, 
                self.agent_name, 
                self.benchmark_name, 
                experiment_name,
                parent_timestamp=self.experiment_timestamp,
                agent_params=self.config.agent_params,
                metrics_params=self.config.runtime_params.get("metrics", {})
            )
            
            # Setup workspace for this idea
            try:
                self.workspace = self.executor.setup_workspace()
                print(f"Workspace for idea '{idea_name}': {self.workspace}")
                
                # Initialize notes for this idea
                self._initialize_notes()
                
                # Initialize Aider for this idea
                self._initialize_aider()
                
                # Execute experiment loop for this idea respecting global iteration budget
                self._experiment_loop()
                
                # Store results for this idea
                idea_results = {
                    "idea": self.current_idea,
                    "experiments": self.state.data.get("results", {}),
                    "completed": self.state.completed,
                    "workspace": self.workspace,
                    "experiment_name": experiment_name
                }
                all_results[f"idea_{idx}"] = idea_results
                
            except Exception as e:
                print(f"Error during experiments for idea '{idea_name}': {e}")
                idea_results = {
                    "idea": self.current_idea,
                    "experiments": {},
                    "completed": False,
                    "error": str(e),
                    "workspace": getattr(self, 'workspace', None),
                    "experiment_name": experiment_name
                }
                all_results[f"idea_{idx}"] = idea_results
                
            finally:
                # Cleanup executor for this idea
                if self.executor:
                    self.executor.cleanup()
                    self.executor = None
                
                # Cleanup Aider for next idea
                if self.coder:
                    try:
                        if hasattr(self.coder, 'save_chat_history'):
                            self.coder.save_chat_history()
                    except:
                        pass
                    self.coder = None
                    self.io = None
                    self.repo = None
        
        # Collect and return all results
        return self._collect_all_results(all_results)
    
    def _load_ideas_from_file(self, ideas_file: str) -> List[Dict[str, Any]]:
        """Load ideas from a JSON file"""
        try:
            with open(ideas_file, 'r') as f:
                ideas = json.load(f)
            
            # Filter novel ideas if novelty check is enabled
            if not self.config.agent_params.get("skip_novelty_check", True):
                ideas = [idea for idea in ideas if idea.get("novel", True)]
            
            if not ideas:
                raise ValueError(f"No novel ideas found in {ideas_file}")
            
            print(f"Loaded {len(ideas)} ideas from {ideas_file}")
            return ideas
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Ideas file not found: {ideas_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in ideas file: {ideas_file}")
    
    def _generate_ideas(self):
        """Generate ideas from task description or load from file"""
        # Check if we should use existing ideas
        use_existing_ideas = self.config.agent_params.get("use_existing_ideas", False)
        ideas_file = self.config.agent_params.get("ideas_file")
        
        if use_existing_ideas and ideas_file:
            # Load ideas from file
            print(f"Loading existing ideas from {ideas_file}...")
            ideas = self._load_ideas_from_file(ideas_file)
        else:
            # Generate new ideas
            print("Generating ideas from task description...")
            
            try:
                # Get base directory for idea generation
                base_dir = self.config.runtime_params.get("base_dir", ".")
                
                # Set up token usage callback
                from .generate_ideas import set_token_usage_callback
                set_token_usage_callback(self.log_token_usage)
                
                # Generate ideas
                ideas = generate_ideas(
                    base_dir=base_dir,
                    client=self.client,
                    model=self.config.model,
                    target_code_files=self.target_code_files,
                    num_generations=self.config.agent_params.get("num_ideas", 3),
                    num_reflections=self.config.agent_params.get("num_reflections", 3)
                )
                
                # Check novelty if requested
                if not self.config.agent_params.get("skip_novelty_check", True):
                    engine = self.config.agent_params.get("engine", "semanticscholar")
                    ideas = check_idea_novelty(
                        ideas,
                        base_dir=base_dir,
                        client=self.client,
                        model=self.config.model,
                        engine=engine
                    )
                    # Filter novel ideas
                    ideas = [idea for idea in ideas if idea.get("novel", True)]
                
            except Exception as e:
                raise RuntimeError(f"Error during idea generation: {str(e)}")
        
        if not ideas:
            raise ValueError("No ideas available for experimentation")
        
        self.state.data["ideas"] = ideas
        print(f"Total ideas to test: {len(ideas)}")
    
    def _save_all_ideas_to_parent(self):
        """Save all generated ideas to parent workspace directory"""
        if not self.parent_workspace:
            return
            
        ideas = self.state.data.get("ideas", [])
        if ideas:
            ideas_file = os.path.join(self.parent_workspace, "generated_ideas.json")
            with open(ideas_file, 'w') as f:
                json.dump(ideas, f, indent=2)
            print(f"\nSaved {len(ideas)} generated ideas to: {ideas_file}")
    
    def _initialize_aider(self):
        """Initialize Aider components for code modification"""
        # Setup IO for non-interactive mode
        if self.workspace is None:
            raise ValueError("Workspace is not set")
        folder_name = self.workspace
        idea_name = self.current_idea.get('Name', 'experiment') if self.current_idea else 'experiment'
        
        chat_history_file = os.path.join(folder_name, f"{idea_name}_aider.txt")
        
        self.io = InputOutput(
            yes=True,
            chat_history_file=chat_history_file
        )
        
        # Handle model name based on provider
        model_name = self.config.model
        provider = self.config.provider
        
        # Special model name mappings (from original code)
        if model_name == "deepseek-coder-v2-0724":
            model_name = "deepseek/deepseek-coder"
        elif model_name == "deepseek-reasoner":
            model_name = "deepseek/deepseek-reasoner"
        elif model_name == "llama3.1-405b":
            model_name = "openrouter/meta-llama/llama-3.1-405b-instruct"
        elif provider == "OpenRouter":
            model_name = f"openrouter/{model_name}"
            
        main_model = Model(model_name)
        
        # Create GitRepo if repo_dir is provided
        repo_dir = self.config.runtime_params.get("repo_dir")
        if repo_dir and self.target_files:
            self.repo = GitRepo(
                io=self.io,
                fnames=self.target_files,
                git_dname=repo_dir,
                models=[]
            )
        else:
            raise ValueError("Repo_dir or target_files is not set")
        
        if len(self.target_files) == 0:
            raise ValueError("target_files is empty")
        
        # Note:
        # Even though we set read_only_files to Coder, it still have chance to write to the files.
        # so we need to copy the original read_only_files to the workspace everytime before running the experiment
        # by setting the copy command in the config.json, we can ensure the eval/train files are not changed
        
        # Print to show the read_only_files
        print(f"\n\nread_only_files: {self.read_only_files}\n\n")

        # Create Aider coder
        self.coder = Coder.create(
            main_model=main_model,
            fnames=self.target_files,
            read_only_fnames=self.read_only_files,
            io=self.io,
            repo=self.repo,
            auto_commits=False,
            dirty_commits=False,
            stream=False,
            use_git=False,
            edit_format="diff"
        )
        
        # Print to show the edit/read_only status
        self.coder.run("/ls")  # check the file write/read_only status

        # Verify coder root is correct
        if repo_dir and hasattr(self.coder, 'root'):
            print(f"Coder root: {self.coder.root}")
            if os.path.abspath(self.coder.root) != os.path.abspath(repo_dir):
                print(f"Warning: Coder root ({self.coder.root}) != repo_dir ({repo_dir})")
    
    def _initialize_notes(self):
        """Initialize notes.txt file"""
        if not self.notes_path:
            return
            
        with open(self.notes_path, 'w') as f:
            f.write(f"# Experiment Notes\n")
            if self.current_idea:
                f.write(f"# Idea Title: {self.current_idea.get('Title', 'Untitled')}\n")
                f.write(f"# Experiment Description: {self.current_idea.get('Experiment', '')}\n")
            f.write(f"\n## Baseline Results\n")
            f.write(f"{json.dumps(self.baseline_results_filtered, indent=2)}\n\n")
            f.write(f"## Experiment Progress\n")
            f.write(f"(Use this space to track your experiments and findings)\n\n")
    
    def _experiment_loop(self):
        """Main experiment loop: code editing → run experiment → analyze results"""
        current_run = 1
        max_runs = self.state.data["max_runs"]

        # Global budget
        if "total_iter" not in self.state.data:
            raise ValueError("total_iter not found in state data")
        total_iter = self.state.data["total_iter"]
        max_iter = self.state.data.get("max_iter")
        max_retries = self.state.data["max_retries_per_run"]
        
        # while current_run <= max_runs and (not self.state.data["use_early_completion"] or not self.state.completed) and total_iter < max_iter:
        while current_run <= max_runs and (not (self.state.data["use_early_completion"] and self.state.completed)) and total_iter < max_iter:
            print(f"\n=== Run {current_run}/{max_runs} === (global_iter {total_iter}/{max_iter})")
            
            # Phase 1: Code editing
            prompt = self._prepare_experiment_prompt(current_run)
            
            # Allow multiple iterations for code editing if errors occur
            edit_iter = 0
            success = False
            
            while edit_iter < max_retries and not success and total_iter < max_iter:
                print(f"Code editing iteration {edit_iter + 1}/{max_retries} (global_iter {total_iter + 1}/{max_iter})")
                
                # Use Aider to modify code
                response = self.coder.run(prompt)
                print(f"Agent response: {response[:200]}...")
                
                # Extract token usage from Aider response
                from benchmark.utils import extract_token_usage_from_aider
                aider_usage = extract_token_usage_from_aider(self.coder.usage_report)

                if aider_usage["total_tokens"] > 0:
                    idea_key = self.current_idea.get('Name', 'unknown') if self.current_idea else 'unknown'
                    self.log_token_usage(aider_usage, "code_editing", idea_key)
                    cost_info = ""
                    if "message_cost" in aider_usage:
                        cost_info = f", ${aider_usage['message_cost']:.4f} message, ${aider_usage['session_cost']:.4f} session"
                    print(f"Aider usage: {aider_usage['total_tokens']} tokens{cost_info}")
                
                # Check for completion signal
                if "ALL_COMPLETED" in response:
                    self.state.completed = True
                
                # Phase 2: Run experiment
                print(f"Running experiment for run {current_run}...")
                result = self.executor.run_experiment(current_run)
                
                # Count this attempt towards global iterations
                total_iter += 1
                self.state.data["total_iter"] = total_iter
                
                if result.get("success"):
                    # Store original results for analysis
                    self.state.data["results"][f"run_{current_run}"] = result.get("results", {})
                    # Store filtered results for prompts
                    self.state.data["filtered_results"][f"run_{current_run}"] = result.get("filtered_results", {})
                    current_run += 1
                    success = True
                else:
                    # Prepare error prompt for retry
                    error_msg = result.get('error', 'Unknown error')
                    max_stderr = self.config.agent_params.get("max_stderr_output")
                    if max_stderr is None:
                        raise ValueError("max_stderr_output not specified in agent configuration")
                    if len(error_msg) > max_stderr:
                        error_msg = "..." + error_msg[-max_stderr:]
                    prompt = f"Run failed with the following error: {error_msg}"
                    edit_iter += 1
            
            # if not success and (not self.state.data["use_early_completion"] or not self.state.completed):
            if not success and (not (self.state.data["use_early_completion"] and self.state.completed)):
                print(f"Failed to complete run {current_run} after {edit_iter} iterations or max_iter reached")
                break
        
        # Copy notes.txt to workspace
        if self.notes_path and os.path.exists(self.notes_path) and self.workspace:
            dst = os.path.join(self.workspace, "notes.txt")
            shutil.copy(self.notes_path, dst)
    
    def _prepare_experiment_prompt(self, current_run: int) -> str:
        """Prepare prompt for experiment based on current run"""
        if current_run == 1:
            # Initial prompt
            if not self.current_idea:
                raise ValueError("No idea available for experimentation")
                
            prompt = f"""Your goal is to implement the following idea: {self.current_idea.get('Title', 'Untitled')}.
The proposed experiment is as follows: {self.current_idea.get('Experiment', 'No experiment description')}.
You are given a total of up to {self.state.data['max_runs']} runs to complete the necessary experiments.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{json.dumps(self.baseline_results_filtered, indent=2)}

After you complete each change, we will run the experiment with command `{self.benchmark_config['execute_commands'][0]}' and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
YOUR PROPOSED CHANGE MUST NOT CHANGE THE READ-ONLY FILES, IF YOU CHANGE THE READ-ONLY FILES, YOU WILL BE FIRED.
These read-only files are: {self.read_only_files}.
EVEN IF YOU CHANGE THE READ-ONLY FILES, IT WILL BE REVERTED TO THE ORIGINAL VERSION IMMEDIATELY.
You can then implement the next thing on your list.
"""
        else:
            # Continuation prompt based on last results
            last_run = current_run - 1
            last_results = self.state.data["filtered_results"].get(f"run_{last_run}", {})
            
            prompt = f"""Run {last_run} completed. Here are the results:
{json.dumps(last_results, indent=2)}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {last_run}, including an experiment description and the run number. Be as verbose as necessary.

Then, implement the next thing on your list for Run {current_run}.
we will run the experiment with command `{self.benchmark_config['execute_commands'][0]}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
YOUR PROPOSED CHANGE MUST NOT CHANGE THE READ-ONLY FILES, IF YOU CHANGE THE READ-ONLY FILES, YOU WILL BE FIRED.
These read-only files are: {self.read_only_files}.
EVEN IF YOU CHANGE THE READ-ONLY FILES, IT WILL BE REVERTED TO THE ORIGINAL VERSION IMMEDIATELY.
If you are finished with experiments, respond with 'ALL_COMPLETED'."""
        
        return prompt
    
    def _collect_all_results(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Collect and return results for all ideas"""
        # Get optimization direction from config
        metrics_config = self.config.runtime_params.get("metrics", {})
        default_direction = metrics_config.get("optimization_direction", "higher")
        per_metric_direction = metrics_config.get("per_metric_direction", {})
        
        # Find best idea based on improvements
        best_idea = None
        best_improvement = -float('inf')
        best_idea_key = None
        
        # First, store the complete original all_results
        idea_summaries = all_results.copy()
        
        # Analyze each idea's results and add summary
        for idea_key, idea_data in all_results.items():
            idea = idea_data.get("idea", {})
            experiments = idea_data.get("experiments", {})
            
            # Calculate summary for this idea
            summary = {
                "total_runs": len(experiments),
                "successful_runs": sum(1 for r in experiments.values() if r),
                "completed": idea_data.get("completed", False)
            }
            
            # Add error info if present
            if "error" in idea_data:
                summary["error"] = idea_data["error"]
            
            # Compare with baseline handling nested dataset-metric structure
            if self.baseline_results and experiments:
                improvements = {}
                overall_improvement = 0
                
                # Iterate through baseline datasets and metrics
                for dataset_name, dataset_data in self.baseline_results.items():
                    if isinstance(dataset_data, dict):
                        # Handle nested structure with metrics
                        for metric_name, baseline_val in dataset_data.items():
                            if isinstance(baseline_val, (int, float)):
                                run_values = []
                                for run_results in experiments.values():
                                    if isinstance(run_results, dict):
                                        # Check if dataset exists in run results
                                        if dataset_name in run_results:
                                            dataset_results = run_results[dataset_name]
                                            if isinstance(dataset_results, dict) and metric_name in dataset_results:
                                                run_values.append(dataset_results[metric_name])
                                
                                if run_values:
                                    # Determine optimization direction for this metric
                                    direction = per_metric_direction.get(metric_name, default_direction)
                                    
                                    if direction == "higher":
                                        best_val = max(run_values)
                                        improvement = ((best_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                                    elif direction == "lower":
                                        best_val = min(run_values)
                                        improvement = ((baseline_val - best_val) / baseline_val * 100) if baseline_val != 0 else 0
                                    else:
                                        best_val = None
                                        improvement = None
                                    
                                    if improvement is not None:   # <- only add if it's numeric
                                        metric_key = f"{dataset_name}_{metric_name}"
                                        improvements[metric_key] = {
                                            "baseline": baseline_val,
                                            "best": best_val,
                                            "improvement": improvement,
                                            "direction": direction,
                                            "all_values": run_values  # Store all run values
                                        }
                                        overall_improvement += improvement
                
                summary["improvements"] = improvements
                summary["overall_improvement"] = overall_improvement / len(improvements) if improvements else 0
                
                # Track best idea
                if summary["overall_improvement"] > best_improvement:
                    best_improvement = summary["overall_improvement"]
                    best_idea = idea
                    best_idea_key = idea_key
            
            # Add summary to the idea data (this modifies idea_summaries which contains all original data)
            idea_summaries[idea_key]["summary"] = summary
        
        # Save all results to parent workspace (not individual idea workspace)
        if self.parent_workspace:
            all_results_file = os.path.join(self.parent_workspace, "all_ideas_results.json")
            with open(all_results_file, 'w') as f:
                json.dump(idea_summaries, f, indent=2)
            print(f"\nAll ideas results saved to: {all_results_file}")
        
        # Get token usage summary
        token_summary = self.get_token_usage_summary()
        
        # Save token usage log to parent workspace
        if self.parent_workspace:
            token_usage_file = os.path.join(self.parent_workspace, "token_usage.json")
            with open(token_usage_file, 'w') as f:
                json.dump({
                    "usage_log": self.token_usage_log,
                    "summary": token_summary
                }, f, indent=2)
            print(f"\nToken usage saved to: {token_usage_file}")
        
        # Return summary with best idea highlighted
        return {
            "all_ideas": idea_summaries,
            "best_idea": best_idea,
            "best_idea_key": best_idea_key,
            "baseline": self.baseline_results,
            "total_ideas_tested": len(all_results),
            "parent_workspace": self.parent_workspace,
            "experiment_timestamp": self.experiment_timestamp,
            "final_run_iter": self.state.data.get("total_iter", 0),
            "token_usage": token_summary
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.coder:
            try:
                if hasattr(self.coder, 'save_chat_history'):
                    self.coder.save_chat_history()
            except:
                pass
        
        # Clear references
        self.coder = None
        self.io = None
        self.repo = None
        self.client = None
        self.executor = None
        
    def get_agent_type(self) -> AgentType:
        """Return the agent type"""
        return AgentType.THEAISCIENTIST
    
    def get_current_idea(self) -> Optional[Dict[str, Any]]:
        """Get the current idea being implemented"""
        return self.current_idea
    
    def get_all_ideas(self) -> List[Dict[str, Any]]:
        """Get all generated ideas"""
        return self.state.data.get("ideas", [])
