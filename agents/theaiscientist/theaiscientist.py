"""
TheAIScientist agent implementation.
Handles the complete research workflow: task description -> idea generation -> coding -> experimentation.
Uses CodeEditor for code modification (replacing Aider).
"""
import os
import json
import shutil
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from .generate_ideas import generate_ideas
from ..llm import create_client
from benchmark.executor_factory import make_executor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt

from ..base import BaseAgent, AgentConfig, AgentType, AgentResult, StepResult
from ..code_editor import CodeEditor


# No hardcoded constants - all values must come from config


class TheAIScientistAgent(BaseAgent):
    """
    TheAIScientist agent - an autonomous AI research agent.
    Executes complete pipeline: generate idea -> modify code -> run experiment -> collect results
    Uses CodeEditor for code modification.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None  # LLM client for idea generation
        self.task_description: Optional[str] = None
        self.task_description_json: Optional[Dict[str, Any]] = None
        self.current_idea: Optional[Dict[str, Any]] = None
        self.baseline_results: Dict[str, Any] = {}
        self.baseline_results_filtered: Dict[str, Any] = {}
        self.read_only_files: List[str] = []
        self.notes_path: Optional[str] = None
        self.workspace: Optional[str] = None
        self.experiment_timestamp: Optional[str] = None  # Parent timestamp for all ideas
        self.parent_workspace: Optional[str] = None  # Parent workspace directory

        # State tracking (replaces self.state.data)
        self.ideas: List[Dict[str, Any]] = []
        self.current_run: int = 1
        self.max_runs: int = 0
        self.max_retries_per_run: int = 0
        self.max_iter: int = 0
        self.total_iter: int = 0
        self.use_early_completion: bool = True
        self.completed: bool = False
        self.results: Dict[str, Any] = {}
        self.filtered_results: Dict[str, Any] = {}
        self._run_primary_metrics: Dict[str, Optional[float]] = {}  # run_N -> primary_metric

        # All steps tracking for AgentResult
        self.all_steps: List[StepResult] = []

    def initialize(self) -> None:
        """Initialize TheAIScientist with necessary components"""
        # Initialize LLM client for idea generation
        provider = self.config.provider
        self.client, _ = create_client(self.config.model, provider)

        # Get max_runs from agent_params, raise error if not set
        if "max_runs" not in self.config.agent_params:
            raise ValueError("max_runs not specified in agent configuration")
        self.max_runs = self.config.agent_params["max_runs"]

        # Global iteration cap across all ideas (runs + retries)
        max_retries_per_run = self.config.agent_params.get("max_retries_per_run")
        if max_retries_per_run is None:
            raise ValueError("max_retries_per_run not specified in agent configuration")
        self.max_retries_per_run = max_retries_per_run
        max_iter = self.config.agent_params.get("max_iter")
        self.max_iter = max_iter if max_iter is not None else float("inf")
        self.total_iter = 0
        self.use_early_completion = self.config.agent_params.get("use_early_completion", True)

        if self.config.agent_params.get("prepare_setup_files_before_gen_ideas", False):
            print("Preparing setup files before generating ideas")
            print("Even though this function is supported, but currently we don't use it")

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

                if timeout is None:
                    timeout = 300

                cwd = osp.abspath(repo_dir)
                all_stderr = ""
                sum_returncode = 0

                for cmd in commands:
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
                    timeout=300,
                )
                if setup_result.returncode != 0:
                    print("="*60)
                    print("!!! WARNING: prepare setup files failed !!!")
                    print(f"Error details: {setup_result.stderr}")
                    print("="*60)
            else:
                print("No setup commands provided. Skip preparing setup files.")

    def log_token_usage(self, usage_info: Dict[str, Any], step_name: str, idea_key: Optional[str] = None):
        """Log token usage information from idea generation callback"""
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model,
            "provider": self.config.provider,
            "prompt_tokens": usage_info.get("prompt_tokens", 0),
            "completion_tokens": usage_info.get("completion_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
            "step_name": step_name,
            "idea_key": idea_key,
            "run_number": self.current_run,
            "iteration": self.total_iter
        }

        if "message_cost" in usage_info:
            usage_entry["message_cost"] = usage_info["message_cost"]
        if "session_cost" in usage_info:
            usage_entry["session_cost"] = usage_info["session_cost"]

        self.token_usage_log.append(usage_entry)

    def run(self,
            task_description: Optional[str] = None,
            target_files: Optional[List[str]] = None,
            baseline_results: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute the complete research pipeline.

        Args:
            task_description: Research task description (used for idea generation)
            target_files: List of files the agent can modify
            baseline_results: Baseline experiment results

        Returns:
            AgentResult with all steps, best step, and test result
        """
        # Store inputs
        self.task_description = task_description
        self.target_files = target_files or []
        self.target_code_files = self.target_files.copy()
        self.baseline_results = baseline_results or {}
        self.baseline_results_filtered = get_filtered_results_for_prompt(self.baseline_results, self.config.runtime_params.get("metrics", {}))

        # Get benchmark configuration from runtime params
        self.benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        self.metric_name = self.benchmark_config.get("metric", "")
        self.metric_direction = self.benchmark_config.get("metric_direction", "higher")
        metrics_cfg = self.config.runtime_params.get("metrics", {})
        include_datasets = metrics_cfg.get("include_datasets")
        self.baseline_primary_metric = extract_primary_metric(
            self.baseline_results, self.metric_name, include_datasets,
        )
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
        self.experiment_timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._output_dir = self.config.runtime_params.get("output_dir", "benchmark_results")
        self.parent_workspace = os.path.join(self._output_dir, self.agent_name, self.benchmark_name, self.experiment_timestamp)
        os.makedirs(self.parent_workspace, exist_ok=False)
        print(f"Parent workspace for this benchmark run: {self.parent_workspace}")

        # Generate or load ideas
        self._generate_ideas()

        # Save all generated ideas to parent workspace
        self._save_all_ideas_to_parent()

        # Execute experiments for all ideas
        all_results = {}

        for idx, idea_to_test in enumerate(self.ideas):
            # Check global iteration budget before starting a new idea
            if self.total_iter >= self.max_iter:
                print("Global max_iter reached. Stopping further ideas.")
                break

            print(f"\n{'='*60}")
            print(f"Testing idea {idx + 1}/{len(self.ideas)}: {idea_to_test.get('Title', 'Untitled')}")
            print(f"{'='*60}\n")

            # Reset state for this idea
            self.current_idea = idea_to_test
            self.current_run = 1
            self.results = {}
            self.filtered_results = {}
            self._run_primary_metrics = {}
            self.completed = False

            # Create experiment name with timestamp and idea name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            idea_name = self.current_idea.get('Name', f'idea_{idx}')
            idea_name = "".join(c for c in idea_name if c.isalnum() or c in ('_', '-')).rstrip()
            experiment_name = f"{timestamp}_{idea_name}"

            # Create executor for this idea
            timeout = self.config.agent_params.get("execute_timeout", 2400)
            eval_backend = self.config.runtime_params.get("eval_backend", "local")
            self.executor = make_executor(
                self.benchmark_config,
                self.agent_name,
                self.benchmark_name,
                experiment_name,
                parent_timestamp=self.experiment_timestamp,
                timeout=timeout,
                output_dir=self._output_dir,
                eval_backend=eval_backend,
            )

            # Setup workspace for this idea
            try:
                self.workspace = self.executor.setup_workspace()
                print(f"Workspace for idea '{idea_name}': {self.workspace}")

                # Initialize notes for this idea
                self._initialize_notes()

                # Initialize CodeEditor for this idea
                self._initialize_editor()

                # Execute experiment loop for this idea respecting global iteration budget
                self._experiment_loop()

                # Store results for this idea
                idea_results = {
                    "idea": self.current_idea,
                    "experiments": self.results.copy(),
                    "completed": self.completed,
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

                # Clear editor for next idea
                self.editor = None

        # Run test with best code
        test_result = self._run_final_test()

        # Build and return AgentResult
        return self._build_agent_result(all_results, test_result)

    def _run_final_test(self) -> Optional[dict]:
        """Run final test using the best code snapshot."""
        if self.best_code_snapshot is None:
            print("No best code snapshot found, skipping test execution.")
            return None

        # We need an executor to run the test
        try:
            timeout = self.config.agent_params.get("execute_timeout", 2400)
            eval_backend = self.config.runtime_params.get("eval_backend", "local")
            self.executor = make_executor(
                self.benchmark_config,
                self.agent_name,
                self.benchmark_name,
                "final_test",
                parent_timestamp=self.experiment_timestamp,
                timeout=timeout,
                output_dir=self._output_dir,
                eval_backend=eval_backend,
            )
            self.executor.setup_workspace()

            # Restore best code and run test
            test_result = self._execute_test()
            return test_result
        except Exception as e:
            print(f"Error during final test execution: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if self.executor:
                self.executor.cleanup()
                self.executor = None

    def _build_agent_result(self, all_results: Dict[str, Dict], test_result: Optional[dict]) -> AgentResult:
        """Build AgentResult from collected data."""
        # Find best step
        best_step = None
        for step in self.all_steps:
            if step.primary_metric is not None:
                if best_step is None or self._is_better_metric(step.primary_metric, best_step.primary_metric):
                    best_step = step

        # Save all results to parent workspace
        self._save_results_to_workspace(all_results)

        return AgentResult(
            all_steps=self.all_steps,
            best_step=best_step,
            test_result=test_result,
            total_steps=self.step_count,
            total_ideas=len(self.ideas),
            token_usage=self.get_token_usage_summary(),
            parent_workspace=self.parent_workspace or "",
            metadata={
                "all_ideas_results": all_results,
                "experiment_timestamp": self.experiment_timestamp,
                "final_run_iter": self.total_iter,
            }
        )

    def _is_better_metric(self, new_metric: float, old_metric: float) -> bool:
        """Check if new metric is better than old based on direction."""
        metric_direction = self.benchmark_config.get("metric_direction", "higher")
        if metric_direction == "lower":
            return new_metric < old_metric
        return new_metric > old_metric

    def _save_results_to_workspace(self, all_results: Dict[str, Dict]):
        """Save results and token usage to parent workspace."""
        if not self.parent_workspace:
            return

        # Save all ideas results
        all_results_file = os.path.join(self.parent_workspace, "all_ideas_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll ideas results saved to: {all_results_file}")

        # Save token usage
        token_summary = self.get_token_usage_summary()
        token_usage_file = os.path.join(self.parent_workspace, "token_usage.json")
        with open(token_usage_file, 'w') as f:
            json.dump({
                "usage_log": self.token_usage_log,
                "summary": token_summary
            }, f, indent=2)
        print(f"\nToken usage saved to: {token_usage_file}")

    def _load_ideas_from_file(self, ideas_file: str) -> List[Dict[str, Any]]:
        """Load ideas from a JSON file"""
        try:
            with open(ideas_file, 'r') as f:
                ideas = json.load(f)

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
        use_existing_ideas = self.config.agent_params.get("use_existing_ideas", False)
        ideas_file = self.config.agent_params.get("ideas_file")

        if use_existing_ideas and ideas_file:
            print(f"Loading existing ideas from {ideas_file}...")
            ideas = self._load_ideas_from_file(ideas_file)
        else:
            print("Generating ideas from task description...")

            # Hard-block: novelty check uses external paper search (unfair advantage)
            if not self.config.agent_params.get("skip_novelty_check", True):
                raise RuntimeError(
                    "Novelty check (paper search) is disabled in FML-bench. "
                    "It provides an unfair advantage via external literature "
                    "access. Set skip_novelty_check=True or remove the flag."
                )

            try:
                base_dir = self.config.runtime_params.get("base_dir", ".")

                from .generate_ideas import set_token_usage_callback
                set_token_usage_callback(self.log_token_usage)

                ideas = generate_ideas(
                    base_dir=base_dir,
                    client=self.client,
                    model=self.config.model,
                    target_code_files=self.target_code_files,
                    num_generations=self.config.agent_params.get("num_ideas", 3),
                    num_reflections=self.config.agent_params.get("num_reflections", 3)
                )

            except Exception as e:
                raise RuntimeError(f"Error during idea generation: {str(e)}")

        if not ideas:
            raise ValueError("No ideas available for experimentation")

        self.ideas = ideas
        print(f"Total ideas to test: {len(ideas)}")

    def _save_all_ideas_to_parent(self):
        """Save all generated ideas to parent workspace directory"""
        if not self.parent_workspace:
            return

        if self.ideas:
            ideas_file = os.path.join(self.parent_workspace, "generated_ideas.json")
            with open(ideas_file, 'w') as f:
                json.dump(self.ideas, f, indent=2)
            print(f"\nSaved {len(self.ideas)} generated ideas to: {ideas_file}")

    def _initialize_editor(self):
        """Initialize CodeEditor for code modification"""
        self.editor = CodeEditor(
            model=self.config.model,
            provider=self.config.provider,
            target_files=self.target_files,
            task_description=self.task_description or "",
            log_dir=self.workspace,
            metric_name=self.benchmark_config.get("metric", ""),
            metric_direction=self.benchmark_config.get("metric_direction", "higher"),
        )

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
            f.write(f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}\n\n")
            f.write(f"## Experiment Progress\n")
            f.write(f"(Use this space to track your experiments and findings)\n\n")

    def _experiment_loop(self):
        """Main experiment loop: code editing -> run experiment -> analyze results"""
        current_run = 1
        max_runs = self.max_runs
        max_retries = self.max_retries_per_run
        last_error = None

        while current_run <= max_runs and (not (self.use_early_completion and self.completed)) and self.total_iter < self.max_iter and self.budget_remaining():
            print(f"\n=== Run {current_run}/{max_runs} === (global_iter {self.total_iter}/{self.max_iter})")

            # Phase 1: Code editing
            prompt = self._prepare_experiment_prompt(current_run)

            # Allow multiple iterations for code editing if errors occur
            edit_iter = 0
            success = False

            while edit_iter < max_retries and not success and self.total_iter < self.max_iter and self.budget_remaining():
                print(f"Code editing iteration {edit_iter + 1}/{max_retries} (global_iter {self.total_iter + 1}/{self.max_iter})")

                # Token tracking and step timing
                _token_start = len(self.token_usage_log)
                _step_t0 = time.monotonic()

                # Use CodeEditor to modify code
                error_context = last_error if edit_iter > 0 else None
                edit_success = self._edit_code(prompt, error_context=error_context)
                print(f"Edit success: {edit_success}")

                # Check for completion signal in raw LLM response
                if (self.use_early_completion
                        and self._last_edit_result
                        and self._last_edit_result.raw_response
                        and "ALL_COMPLETED" in self._last_edit_result.raw_response):
                    self.completed = True
                    print("Agent signaled ALL_COMPLETED")

                # Phase 2: Run experiment
                print(f"Running experiment for run {current_run}...")
                result = self._execute_val(current_run)

                # Count this attempt towards global iterations
                self.total_iter += 1

                # Build StepResult
                idea_name = self.current_idea.get('Name', 'unknown') if self.current_idea else 'unknown'
                action = "debug" if edit_iter > 0 else ("draft" if current_run == 1 else "improve")
                snap_path = self._save_step_code_snapshot(self.step_count, self.parent_workspace)
                step_result = StepResult(
                    step_id=self.step_count,
                    idea_id=idea_name,
                    idea_description=self.current_idea.get('Title', '') if self.current_idea else '',
                    action=action,
                    edit_success=edit_success,
                    val_result=result,
                    primary_metric=result.get("primary_metric"),
                    token_usage=self._collect_step_tokens(_token_start),
                    step_duration_seconds=time.monotonic() - _step_t0,
                    metadata={
                        "code_snapshot_path": snap_path,
                        "idea": self.current_idea if self.current_idea else None,
                        "instruction": self._last_edit_instruction,
                        "editor_log_path": getattr(self._last_edit_result, "log_path", None),
                    },
                )
                self.all_steps.append(step_result)

                if result.get("success"):
                    # Store results
                    self.results[f"run_{current_run}"] = result.get("results", {})
                    self.filtered_results[f"run_{current_run}"] = result.get("filtered_results", result.get("results", {}))
                    self._run_primary_metrics[f"run_{current_run}"] = result.get("primary_metric")
                    current_run += 1
                    success = True
                    last_error = None
                else:
                    # Prepare error prompt for retry
                    error_msg = result.get('error', 'Unknown error')
                    max_stderr = self.config.agent_params.get("max_stderr_output")
                    if max_stderr is None:
                        raise ValueError("max_stderr_output not specified in agent configuration")
                    if len(error_msg) > max_stderr:
                        error_msg = "..." + error_msg[-max_stderr:]
                    prompt = "The previous run failed. Please fix the issue based on the error output provided."
                    last_error = error_msg
                    edit_iter += 1

            if not success and (not (self.use_early_completion and self.completed)):
                print(f"Failed to complete run {current_run} after {edit_iter} iterations or max_iter reached")
                break

        # Copy notes.txt to workspace
        if self.notes_path and os.path.exists(self.notes_path) and self.workspace:
            dst = os.path.join(self.workspace, "notes.txt")
            shutil.copy(self.notes_path, dst)

    def _prepare_experiment_prompt(self, current_run: int) -> str:
        """Prepare prompt for experiment based on current run"""
        if current_run == 1:
            if not self.current_idea:
                raise ValueError("No idea available for experimentation")

            prompt = f"""Your goal is to implement the following idea: {self.current_idea.get('Title', 'Untitled')}.
The proposed experiment is as follows: {self.current_idea.get('Experiment', 'No experiment description')}.
You are given a total of up to {self.max_runs} runs to complete the necessary experiments.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:
{self._format_metric_line(self.baseline_primary_metric, "Baseline")}

After you complete each change, we will run the experiment and evaluate the results.
YOUR PROPOSED CHANGE MUST NOT CHANGE THE READ-ONLY FILES, IF YOU CHANGE THE READ-ONLY FILES, YOU WILL BE FIRED.
These read-only files are: {self.read_only_files}.
EVEN IF YOU CHANGE THE READ-ONLY FILES, IT WILL BE REVERTED TO THE ORIGINAL VERSION IMMEDIATELY.
You can then implement the next thing on your list.
"""
        else:
            last_run = current_run - 1
            last_metric = self._run_primary_metrics.get(f"run_{last_run}")

            prompt = f"""Run {last_run} completed. Here are the results:
{self._format_metric_line(last_metric, "Last run result")}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {last_run}, including an experiment description and the run number. Be as verbose as necessary.

Then, implement the next thing on your list for Run {current_run}.
YOUR PROPOSED CHANGE MUST NOT CHANGE THE READ-ONLY FILES, IF YOU CHANGE THE READ-ONLY FILES, YOU WILL BE FIRED.
These read-only files are: {self.read_only_files}.
EVEN IF YOU CHANGE THE READ-ONLY FILES, IT WILL BE REVERTED TO THE ORIGINAL VERSION IMMEDIATELY.
If you are finished with experiments, respond with 'ALL_COMPLETED'."""

        return prompt

    def cleanup(self) -> None:
        """Clean up resources"""
        self.editor = None
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
        return self.ideas
