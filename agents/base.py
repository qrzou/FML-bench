"""
Base agent abstraction for AI research agents.

Provides shared infrastructure: step tracking, code snapshot/restore,
token usage aggregation, and abstract interfaces for agent subclasses.
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from benchmark.executor import BenchmarkExecutor


class AgentType(Enum):
    """Supported agent types."""
    THEAISCIENTIST = "theaiscientist"
    AI_SCIENTIST_V2 = "ai_scientist_v2"
    AIDE = "aide"
    AIRA_MCTS = "aira_mcts"
    OPENEVOLVE = "openevolve"
    AUTORESEARCH = "autoresearch"
    ADAPTIVESEARCH = "adaptivesearch"


@dataclass
class AgentConfig:
    """Configuration for agent initialization."""
    agent_type: AgentType
    model: str = "gpt-5"
    provider: str = "OpenAI"
    agent_params: Dict[str, Any] = field(default_factory=dict)
    runtime_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a single agent step (one val execution)."""
    step_id: int
    idea_id: str
    idea_description: str
    action: str               # "draft" | "improve" | "debug" | "switch"
    edit_success: bool
    val_result: Optional[dict]
    primary_metric: Optional[float]
    token_usage: Optional[dict] = None
    step_duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Aggregate result returned by an agent after its full run."""
    all_steps: List[StepResult]
    best_step: Optional[StepResult]
    test_result: Optional[dict]
    total_steps: int
    total_ideas: int
    token_usage: dict
    parent_workspace: str
    total_duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all AI research agents.

    Provides shared infrastructure for step counting, best-code tracking,
    code snapshot/restore, and token usage aggregation. Subclasses implement
    ``initialize()`` and ``run()`` with their own search/decision strategy.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.editor = None          # CodeEditor instance, set during initialize()
        self.executor: Optional["BenchmarkExecutor"] = None  # set during run()
        self.step_count: int = 0
        self.step_budget: int = config.agent_params.get("max_steps", 50)
        self.token_usage_log: list = []
        self.best_metric: Optional[float] = None
        self.best_code_snapshot: Optional[Dict[str, str]] = None  # {filepath: content}
        self.target_files: List[str] = []
        self._last_edit_result = None  # Full EditResult from last _edit_code() call
        self._last_val_duration: Optional[float] = None  # Wall-clock seconds for last _execute_val
        # Metric info — set by agents during run() from benchmark_config
        self.metric_name: str = ""
        self.metric_direction: str = "higher"
        self.baseline_primary_metric: Optional[float] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Initialize agent. Subclass must set up self.editor and agent-specific state."""
        pass

    @abstractmethod
    def run(
        self,
        task_description: str,
        target_files: List[str],
        baseline_results: dict,
    ) -> AgentResult:
        """Main execution loop. Subclass implements search/decision strategy."""
        pass

    # ------------------------------------------------------------------
    # Shared code-editing helper
    # ------------------------------------------------------------------

    def _edit_code(self, instruction: str, error_context: Optional[str] = None) -> bool:
        """Shared code editing via CodeEditor. Returns True if edit succeeded.

        Raises:
            RuntimeError: If ``self.editor`` has not been set (i.e., ``initialize()``
                was not called or did not assign an editor).
        """
        if self.editor is None:
            raise RuntimeError(
                "self.editor is None. Call initialize() before _edit_code()."
            )
        self._last_edit_instruction = instruction  # Full instruction for per-step logging
        result = self.editor.edit(instruction, error_context)
        self._last_edit_result = result
        if result.token_usage:
            self.token_usage_log.append(result.token_usage)
        return result.success

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_val(self, run_id: int) -> dict:
        """Run a validation experiment. Increments step_count. Auto-tracks best code.

        Raises:
            RuntimeError: If ``self.executor`` has not been set.
        """
        if self.executor is None:
            raise RuntimeError(
                "self.executor is None. Assign it before calling _execute_val()."
            )
        self.step_count += 1
        t0 = time.monotonic()
        try:
            result = self.executor.run_val(run_id)
        finally:
            self._last_val_duration = time.monotonic() - t0

        # Auto-track best code
        if result.get("success") and result.get("primary_metric") is not None:
            if self._should_update_best(result["primary_metric"], self.best_metric):
                self.best_metric = result["primary_metric"]
                self.best_code_snapshot = self._snapshot_target_files()

        return result

    def _execute_test(self) -> dict:
        """Run the final test once. Restores best code first. Agent never sees result.

        Flow: restore best target files → re-run val (regenerates checkpoint
        artifacts deleted by setup_workspace's git clean) → run test.

        Raises:
            RuntimeError: If ``self.executor`` has not been set.
        """
        if self.executor is None:
            raise RuntimeError(
                "self.executor is None. Assign it before calling _execute_test()."
            )
        if self.best_code_snapshot:
            self._restore_snapshot(self.best_code_snapshot)
        # Re-run val to regenerate checkpoint artifacts (model weights, etc.)
        # that were deleted by setup_workspace's git clean.
        # Uses executor.run_val directly (not _execute_val) to avoid
        # incrementing step_count or overwriting best_code_snapshot.
        # The per-step timeout guards the search loop; the final test phase
        # (pre_test_val regen + final_test eval) must run unbounded.
        saved_timeout = self.executor.timeout
        self.executor.timeout = None
        try:
            self.executor.run_val(run_id="pre_test_val")
            return self.executor.run_test(run_id="final_test")
        finally:
            self.executor.timeout = saved_timeout

    # ------------------------------------------------------------------
    # Best-code selection strategy
    # ------------------------------------------------------------------

    def _should_update_best(self, new_metric: float, current_best: Optional[float]) -> bool:
        """Decide whether new_metric is better than current_best.

        Subclasses can override for smarter selection strategies
        (e.g., prefer stable improvements, Pareto front).
        Default: simple comparison based on metric_direction.
        """
        direction = self.executor.config.get("metric_direction", "higher")
        if current_best is None:
            return True
        if direction == "higher":
            return new_metric > current_best
        return new_metric < current_best

    # ------------------------------------------------------------------
    # Code snapshot / restore
    # ------------------------------------------------------------------

    def _snapshot_target_files(self) -> Dict[str, str]:
        """Save current state of all target files.

        Returns:
            Mapping from filepath to file content.
        """
        snapshot: Dict[str, str] = {}
        for filepath in self.target_files:
            p = Path(filepath)
            if p.exists():
                snapshot[filepath] = p.read_text()
        return snapshot

    def _restore_snapshot(self, snapshot: Dict[str, str]) -> None:
        """Restore target files from a previously saved snapshot."""
        for filepath, content in snapshot.items():
            Path(filepath).write_text(content)

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def budget_remaining(self) -> bool:
        """Return True if the step budget has not been exhausted."""
        return self.step_count < self.step_budget

    # ------------------------------------------------------------------
    # Metric formatting (unified across all agents for fairness)
    # ------------------------------------------------------------------

    def _format_metric_line(self, metric_value: Optional[float],
                            label: str = "Metric") -> str:
        """Format metric info as a single standardised line for LLM prompts.

        All agents MUST use this helper to present metric information to the
        LLM so that no agent sees more detail than another.
        """
        if metric_value is not None:
            return (f"{label}: {self.metric_name} = {metric_value} "
                    f"({self.metric_direction} is better)")
        return (f"{label}: {self.metric_name} = N/A "
                f"({self.metric_direction} is better)")

    # ------------------------------------------------------------------
    # Token usage
    # ------------------------------------------------------------------

    def _collect_step_tokens(self, start_idx: int) -> Optional[dict]:
        """Aggregate token usage entries added since *start_idx*.

        Call ``start_idx = len(self.token_usage_log)`` **before** the step's
        edit + val + analysis cycle, then call this method afterwards to get
        only the tokens consumed during that step.
        """
        summary: Dict[str, Any] = {}
        for entry in self.token_usage_log[start_idx:]:
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    summary[key] = summary.get(key, 0) + value
        return summary or None  # None keeps StepResult lightweight when empty

    def _save_step_code_snapshot(
        self,
        step_id: int,
        workspace_dir: str,
        snapshot: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Save code snapshot to a per-step JSON file.

        Args:
            step_id: The step number (used in the filename).
            workspace_dir: Directory under which ``step_snapshots/`` is created.
            snapshot: Explicit snapshot dict to write.  When *None*, the
                current on-disk target files are read via
                ``_snapshot_target_files()``.  Pass an explicit dict when the
                disk state may no longer match the step's code (e.g. when
                saving retroactively from tree nodes after the search loop).

        Returns the file path on success, or None if saving fails.
        """
        if snapshot is None:
            snapshot = self._snapshot_target_files()
        if not snapshot or not workspace_dir:
            return None
        try:
            snap_dir = os.path.join(workspace_dir, "step_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            path = os.path.join(snap_dir, f"step_{step_id:04d}_code.json")
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2)
            return path
        except Exception:
            return None

    def get_token_usage_summary(self) -> dict:
        """Aggregate token usage from all logged entries.

        Each entry in ``self.token_usage_log`` is expected to be a dict with
        numeric values (e.g., ``{"prompt_tokens": 100, "completion_tokens": 50}``).
        Keys are unioned across all entries and values are summed.

        Returns:
            A single dict with summed token counts.
        """
        summary: Dict[str, Any] = {}
        for entry in self.token_usage_log:
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    summary[key] = summary.get(key, 0) + value
        return summary

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up agent resources. Override if needed."""
        pass

    def kill_running_process(self) -> None:
        """Kill the currently running experiment subprocess.

        Delegates to executor.kill_running_process() which kills the entire
        process group (conda → bash → python train_eval.py).
        Called by signal handlers when the parent process is killed externally.
        """
        if self.executor is not None:
            self.executor.kill_running_process()
