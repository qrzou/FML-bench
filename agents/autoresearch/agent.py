"""
Autoresearch agent for FML-bench.

Faithfully ported from: https://github.com/karpathy/autoresearch
(Karpathy, March 2026)

Core algorithm: Greedy hill-climbing loop directed by LLM.
  - Modify code with an experimental idea
  - Run experiment
  - If metric improved: KEEP (advance)
  - If metric equal or worse: DISCARD (revert)
  - If crash: debug if simple, skip if fundamental
  - Repeat until budget exhausted

Original design philosophy (preserved):
  - Greedy acceptance: strictly better -> keep, else -> discard
  - Crash recovery: debug simple bugs, skip fundamentally broken ideas
  - Experiment history tracking: in-memory ExperimentRecord list (replaces results.tsv)
  - Continuous operation: loop until budget exhausted ("NEVER STOP")
  - Simplicity: all else equal, simpler is better

Adapted for FML-bench:
  - No built-in LLM in original -> idea generation via agents/llm.py,
    code editing via CodeEditor (shared, fair)
  - Git-based versioning -> _snapshot_target_files / _restore_snapshot
  - results.tsv -> in-memory ExperimentRecord list
  - Fixed 5-min time budget -> step budget + execute_timeout
  - Single metric (val_bpb, lower=better) -> configurable metric/direction
  - program.md instructions -> prompt construction in code
  - Execution via BenchmarkExecutor (standardized)

Reference: upstream autoresearch — program.md.
"""

import json
import logging
import os
import os.path as osp
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from benchmark.executor import BenchmarkExecutor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt

from ..base import AgentConfig, AgentResult, BaseAgent, StepResult
from ..code_editor import CodeEditor
from ..llm import create_client, get_response_from_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """One row of the in-memory results log (replaces results.tsv).

    Original columns: commit, val_bpb, memory_gb, status, description.
    Mapped to FML-bench: step_id, primary_metric, status, description.
    """

    step_id: int
    primary_metric: Optional[float]  # None for crashes
    status: str  # "keep" | "discard" | "crash"
    description: str
    error_context: Optional[str] = None


# ---------------------------------------------------------------------------
# AutoresearchAgent
# ---------------------------------------------------------------------------

class AutoresearchAgent(BaseAgent):
    """
    Autoresearch: greedy hill-climbing ML research agent.

    Search policy (from program.md):
      1. Propose experimental idea (LLM call)
      2. Edit code (CodeEditor)
      3. Run experiment (_execute_val)
      4. If improved -> keep. If not -> discard. If crash -> debug or skip.
      5. Repeat.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None

        # Agent params (set in initialize)
        self.max_debug_retries: int = 3
        self.max_stderr_output: int = 1500

        # Runtime state
        self.experiment_log: List[ExperimentRecord] = []
        self.current_snapshot: Dict[str, str] = {}
        self.current_best_metric: Optional[float] = None
        self.consecutive_crashes: int = 0
        self.metric_direction: str = "higher"
        self.metric_name: str = ""
        self.task_description: str = ""
        self.baseline_results_filtered: Dict[str, Any] = {}
        self.all_steps: List[StepResult] = []

        # Debug state (for crash recovery)
        self._pending_debug: bool = False
        self._last_idea: str = ""
        self._last_error: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if self.client is not None:
            return
        self.client, _ = create_client(self.config.model, self.config.provider)
        p = self.config.agent_params
        self.max_debug_retries = int(p.get("max_debug_retries", 3))
        self.max_stderr_output = int(p.get("max_stderr_output", 1500))
        self.step_budget = int(p.get("max_steps", p.get("max_iter", 50)))

    def run(
        self,
        task_description=None,
        target_files=None,
        baseline_results=None,
    ) -> AgentResult:
        # -- unpack inputs --
        if isinstance(task_description, tuple):
            self.task_description, _ = task_description
        else:
            self.task_description = task_description or ""

        self.target_files = list(target_files or [])
        metrics_cfg = self.config.runtime_params.get("metrics", {})
        self.baseline_results_filtered = get_filtered_results_for_prompt(
            baseline_results or {}, metrics_cfg
        )

        benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        self.metric_direction = benchmark_config.get("metric_direction", "higher")
        self.metric_name = benchmark_config.get("metric", "")
        include_datasets = metrics_cfg.get("include_datasets")
        self.baseline_primary_metric = extract_primary_metric(
            baseline_results or {}, self.metric_name, include_datasets,
        )
        # Initialize comparison threshold from baseline (original: first run establishes baseline)
        self.current_best_metric = self.baseline_primary_metric
        agent_name = self.config.runtime_params.get("agent_name", "autoresearch")
        benchmark_name = self.config.runtime_params.get("benchmark_name", "benchmark")

        ts = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._output_dir = self.config.runtime_params.get("output_dir", "benchmark_results")
        parent_workspace = osp.join(
            self._output_dir, agent_name, benchmark_name, ts
        )
        os.makedirs(parent_workspace, exist_ok=True)
        self._parent_workspace = parent_workspace

        # -- create executor --
        timeout = self.config.agent_params.get("execute_timeout", 2400)
        self.executor = BenchmarkExecutor(
            benchmark_config, agent_name, benchmark_name,
            f"{ts}_autoresearch", parent_timestamp=ts, timeout=timeout,
            output_dir=self._output_dir,
        )
        workspace = self.executor.setup_workspace()
        print(f"Autoresearch workspace: {workspace}")

        # -- create editor --
        self.editor = CodeEditor(
            model=self.config.model,
            provider=self.config.provider,
            target_files=self.target_files,
            task_description=self.task_description,
            log_dir=workspace,
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
        )

        # -- initialize --
        self.initialize()
        self.experiment_log = []
        self.all_steps = []
        self.consecutive_crashes = 0
        self._pending_debug = False

        # Snapshot baseline code (the "branch head")
        self.current_snapshot = self._snapshot_target_files()

        # -- main loop --
        try:
            self._main_loop()
        except Exception as e:
            print(f"[Autoresearch] Loop error: {e}")
            import traceback
            traceback.print_exc()

        # -- final test --
        test_result = self._run_final_test(
            benchmark_config, agent_name, benchmark_name, ts, timeout
        )

        # -- build result --
        best_step = self._find_best_step()
        return AgentResult(
            all_steps=self.all_steps,
            best_step=best_step,
            test_result=test_result,
            total_steps=self.step_count,
            total_ideas=len([r for r in self.experiment_log if r.status == "keep"]),
            token_usage=self.get_token_usage_summary(),
            parent_workspace=parent_workspace,
            metadata={
                "experiment_log": [
                    {
                        "step_id": r.step_id,
                        "primary_metric": r.primary_metric,
                        "status": r.status,
                        "description": r.description[:200],
                    }
                    for r in self.experiment_log
                ]
            },
        )

    # ------------------------------------------------------------------
    # Main loop (from program.md lines 94-104)
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        print(
            f"\n{'='*60}\n"
            f"Autoresearch loop  |  budget={self.step_budget}  "
            f"debug_retries={self.max_debug_retries}\n"
            f"{'='*60}\n"
        )

        while self.budget_remaining():
            _token_start = len(self.token_usage_log)
            _step_t0 = time.monotonic()

            if self._pending_debug:
                # Debug retry: restore snapshot, edit with error context
                self._pending_debug = False
                self._restore_snapshot(self.current_snapshot)
                instruction = self._build_debug_instruction(
                    self._last_error, self._last_idea
                )
                edit_ok = self._edit_code(instruction, error_context=self._last_error)
                idea = f"[debug retry] {self._last_idea}"
                action = "debug"
            else:
                # Generate new idea via LLM
                idea = self._generate_idea()
                self._last_idea = idea
                # Restore to current best before editing
                self._restore_snapshot(self.current_snapshot)
                # Edit code via CodeEditor
                edit_ok = self._edit_code(idea)
                action = "draft" if self.step_count == 0 else "improve"

            # Run experiment (1 step)
            run_id = self.step_count
            val_result = self._execute_val(run_id)

            # Record StepResult
            snap_path = self._save_step_code_snapshot(self.step_count, self._parent_workspace)
            step_result = StepResult(
                step_id=self.step_count,
                idea_id=f"exp_{self.step_count}",
                idea_description=idea[:200],
                action=action,
                edit_success=edit_ok,
                val_result=val_result,
                primary_metric=val_result.get("primary_metric"),
                token_usage=self._collect_step_tokens(_token_start),
                step_duration_seconds=time.monotonic() - _step_t0,
                metadata={
                    "code_snapshot_path": snap_path,
                    "idea": idea,
                    "instruction": self._last_edit_instruction,
                    "editor_log_path": getattr(self._last_edit_result, "log_path", None),
                },
            )
            self.all_steps.append(step_result)

            # Evaluate: keep / discard / crash
            if (
                not val_result.get("success")
                or val_result.get("primary_metric") is None
            ):
                self._handle_crash(val_result, idea)
            else:
                metric = val_result["primary_metric"]
                if self._is_strict_improvement(metric):
                    self._keep(metric, idea)
                else:
                    self._discard(metric, idea)

    # ------------------------------------------------------------------
    # Keep / Discard / Crash (from program.md lines 101-110)
    # ------------------------------------------------------------------

    def _is_strict_improvement(self, metric: float) -> bool:
        """Strictly better only (program.md: 'If val_bpb improved')."""
        if self.current_best_metric is None:
            return True
        if self.metric_direction == "lower":
            return metric < self.current_best_metric
        return metric > self.current_best_metric

    def _keep(self, metric: float, idea: str) -> None:
        """Keep: advance the branch (program.md line 103)."""
        self.current_snapshot = self._snapshot_target_files()
        self.current_best_metric = metric
        self.consecutive_crashes = 0
        self.experiment_log.append(
            ExperimentRecord(
                step_id=self.step_count,
                primary_metric=metric,
                status="keep",
                description=idea[:200],
            )
        )
        print(
            f"  KEEP  | step={self.step_count} | "
            f"{self.metric_name}={metric} | {idea[:80]}"
        )

    def _discard(self, metric: float, idea: str) -> None:
        """Discard: git reset to previous commit (program.md line 104)."""
        self._restore_snapshot(self.current_snapshot)
        self.consecutive_crashes = 0
        self.experiment_log.append(
            ExperimentRecord(
                step_id=self.step_count,
                primary_metric=metric,
                status="discard",
                description=idea[:200],
            )
        )
        print(
            f"  DISCARD | step={self.step_count} | "
            f"{self.metric_name}={metric} | {idea[:80]}"
        )

    def _handle_crash(self, val_result: dict, idea: str) -> None:
        """Handle crash (program.md lines 110): debug if simple, skip if fundamental."""
        error_msg = val_result.get("error", "Unknown error")
        truncated_error = self._truncate_error(error_msg)

        self.experiment_log.append(
            ExperimentRecord(
                step_id=self.step_count,
                primary_metric=None,
                status="crash",
                description=idea[:200],
                error_context=truncated_error[:300],
            )
        )

        self.consecutive_crashes += 1
        print(
            f"  CRASH | step={self.step_count} | "
            f"consecutive={self.consecutive_crashes} | {idea[:80]}"
        )

        if self.consecutive_crashes < self.max_debug_retries:
            # "something dumb and easy to fix" -> debug retry
            self._pending_debug = True
            self._last_error = truncated_error
        else:
            # "the idea itself is fundamentally broken, just skip it"
            self._restore_snapshot(self.current_snapshot)
            self.consecutive_crashes = 0
            self._pending_debug = False
            print("  Giving up on this idea, moving on.")

    # ------------------------------------------------------------------
    # LLM: Idea generation (program.md: the agent proposes ideas)
    # ------------------------------------------------------------------

    def _generate_idea(self) -> str:
        """Generate next experimental idea via LLM.

        In the original, the external LLM reads train.py + results.tsv + program.md
        and decides what to change. Here we replicate that context.
        """
        prompt = self._build_idea_prompt()
        system_msg = (
            "You are an autonomous ML researcher running experiments to "
            "improve an ML algorithm. You propose concise, specific "
            "experimental ideas. Be creative but practical. All else being "
            "equal, simpler is better."
        )
        try:
            text, _, usage = get_response_from_llm(
                prompt,
                client=self.client,
                model=self.config.model,
                system_message=system_msg,
            )
            if usage:
                self.token_usage_log.append(usage)
            return text.strip()
        except Exception as e:
            logger.error("Idea generation failed: %s", e)
            return "Try a small hyperparameter change to improve the metric."

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_idea_prompt(self) -> str:
        """Build the idea-generation prompt.

        Replicates the context the original Autoresearch LLM has:
        - README.md -> task_description
        - train.py -> current code (via _format_current_code)
        - results.tsv -> experiment_log formatted as table
        - Baseline results
        """
        parts = []

        # Task description (equivalent to README.md in original)
        parts.append(
            "You are an autonomous ML researcher. Your goal is to propose "
            "the next experimental idea to improve the ML algorithm.\n"
        )
        parts.append(f"## Task Description\n{self.task_description}")

        # Baseline results
        if self.baseline_primary_metric is not None:
            parts.append(
                f"\n## Baseline Results\n"
                f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}"
            )

        # Current best metric
        if self.current_best_metric is not None:
            parts.append(
                f"\n## Current Best Metric\n"
                f"{self._format_metric_line(self.current_best_metric, 'Current Best')}"
            )

        # Experiment history (equivalent to results.tsv)
        parts.append(
            f"\n## Experiment History\n{self._format_experiment_log()}"
        )

        # Current code (equivalent to reading train.py)
        parts.append(f"\n## Current Code\n{self._format_current_code()}")

        # Instructions (from program.md philosophy)
        parts.append(
            "\n## Instructions\n"
            "Propose ONE specific experimental idea to try next. Consider:\n"
            "- What has worked and what hasn't (see Experiment History)\n"
            "- The current code state and architecture\n"
            "- Architecture changes, optimizer changes, hyperparameter tuning, "
            "training loop modifications are all fair game\n"
            "- Keep changes focused and atomic so we can evaluate the effect\n"
            "- If recent experiments have been discarded or crashed, try a "
            "different direction\n"
            "- All else being equal, simpler is better\n\n"
            "Respond with a concise description (3-5 sentences) of what to "
            "change and why. Be specific enough that the change can be "
            "implemented directly."
        )

        return "\n".join(parts)

    def _build_debug_instruction(self, error_context: str, original_idea: str) -> str:
        """Build instruction for debugging a crashed experiment."""
        return (
            f"The previous experiment crashed. The intended change was:\n"
            f"{original_idea}\n\n"
            f"## Error\n```\n{error_context}\n```\n\n"
            f"If this is a simple fix (typo, missing import, wrong variable name), "
            f"fix it while preserving the intended change. "
            f"If the approach is fundamentally broken, simplify it to something "
            f"that will run correctly."
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_experiment_log(self) -> str:
        """Format experiment_log as a table (equivalent to results.tsv)."""
        if not self.experiment_log:
            return "(No experiments run yet.)"

        lines = [f"step | {self.metric_name:>10s} | status  | description"]
        lines.append("-" * 65)
        for rec in self.experiment_log:
            if rec.primary_metric is not None:
                metric_str = f"{rec.primary_metric:.6f}"
            else:
                metric_str = "   CRASH  "
            lines.append(
                f"{rec.step_id:4d} | {metric_str:>10s} | "
                f"{rec.status:<7s} | {rec.description[:60]}"
            )
        return "\n".join(lines)

    def _format_current_code(self) -> str:
        """Read current target files for the LLM context."""
        parts: List[str] = []
        for filepath in self.target_files:
            try:
                with open(filepath, "r") as f:
                    content = f.read()
                basename = os.path.basename(filepath)
                parts.append(f"### {basename}\n```python\n{content}\n```")
            except FileNotFoundError:
                parts.append(f"### {os.path.basename(filepath)}\n(file not found)")
        return "\n\n".join(parts)

    def _truncate_error(self, error: str) -> str:
        if not error:
            return ""
        if len(error) > self.max_stderr_output:
            return "..." + error[-self.max_stderr_output :]
        return error

    # ------------------------------------------------------------------
    # Final test & result helpers
    # ------------------------------------------------------------------

    def _run_final_test(
        self, benchmark_config, agent_name, benchmark_name, parent_ts, timeout
    ) -> Optional[dict]:
        if self.best_code_snapshot is None:
            print("No best code snapshot found, skipping test.")
            return None
        try:
            # Clean up search executor
            if self.executor:
                self.executor.cleanup()
            test_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.executor = BenchmarkExecutor(
                benchmark_config,
                agent_name,
                benchmark_name,
                f"{test_ts}_final_test",
                parent_timestamp=parent_ts,
                timeout=timeout,
                output_dir=self._output_dir,
            )
            self.executor.setup_workspace()
            test_result = self._execute_test()
            return test_result
        except Exception as e:
            print(f"Error during final test: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if self.executor:
                self.executor.cleanup()
                self.executor = None

    def _find_best_step(self) -> Optional[StepResult]:
        valid_steps = [s for s in self.all_steps if s.primary_metric is not None]
        if not valid_steps:
            return None
        if self.metric_direction == "higher":
            return max(valid_steps, key=lambda s: s.primary_metric)
        return min(valid_steps, key=lambda s: s.primary_metric)
