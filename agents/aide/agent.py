"""
AIDE agent implementation for FML-bench.

Ports the AIDE (Weco) tree-search algorithm into the BaseAgent framework.
Uses a solution tree with draft/improve/debug stages and a greedy+debug
search policy. All code editing goes through the shared CodeEditor;
all execution goes through BenchmarkExecutor.

Reference: upstream AIDE — agent.py and journal.py.
"""

import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from agents.base import AgentConfig, AgentResult, BaseAgent, StepResult
from agents.code_editor import CodeEditor
from benchmark.executor import BenchmarkExecutor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt

logger = logging.getLogger(__name__)


# ======================================================================
# Data Structures
# ======================================================================

@dataclass(eq=False)
class TreeNode:
    """A single node in the AIDE solution tree.

    Each node represents one attempt (draft, improvement, or debug fix).
    The node stores a code snapshot so the tree can be traversed freely.
    """

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    step_id: Optional[int] = None
    plan: str = ""
    action: str = ""  # "draft" | "improve" | "debug"
    code_snapshot: Dict[str, str] = field(default_factory=dict)
    parent: Optional["TreeNode"] = field(default=None, repr=False)
    children: set = field(default_factory=set, repr=False)
    val_result: Optional[dict] = None
    primary_metric: Optional[float] = None
    is_buggy: bool = True
    error_context: str = ""
    analysis: str = ""

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """Determine the stage based on the parent relationship."""
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    @property
    def debug_depth(self) -> int:
        """Length of the current consecutive debug path.

        0 if this node is not a debug node, otherwise 1 + parent's debug_depth.
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore[union-attr]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TreeNode) and self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)


class Journal:
    """Collection of TreeNodes forming the solution tree.

    Mirrors AIDE's Journal: tracks draft/buggy/good partitions and
    provides best-node selection and summary generation.
    """

    def __init__(self, metric_direction: str = "higher") -> None:
        self.nodes: List[TreeNode] = []
        self.metric_direction = metric_direction

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: TreeNode) -> None:
        node.step_id = len(self.nodes)
        self.nodes.append(node)

    # ---- filtered views ----

    @property
    def draft_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if not n.is_buggy]

    # ---- best node ----

    def get_best_node(self) -> Optional[TreeNode]:
        """Return the good node with the best primary_metric."""
        good = self.good_nodes
        if not good:
            return None
        if self.metric_direction == "higher":
            return max(good, key=lambda n: n.primary_metric if n.primary_metric is not None else float("-inf"))
        return min(good, key=lambda n: n.primary_metric if n.primary_metric is not None else float("inf"))

    # ---- memory / summary ----

    def generate_summary(self, metric_name: str = "metric") -> str:
        """Summarize all good nodes' plans and metrics as context for the LLM.

        Matches original AIDE's ``generate_summary()`` default behaviour
        (include_code=False).  Code is NOT included in search memory —
        original only includes code in final report generation.
        """
        if not self.good_nodes:
            return "(No successful attempts yet.)"
        parts: List[str] = []
        for n in self.good_nodes:
            part = f"Plan: {n.plan}\nAnalysis: {n.analysis}\n{metric_name}: {n.primary_metric}"
            parts.append(part)
        return "\n-------------------------------\n".join(parts)


# ======================================================================
# AIDE Agent
# ======================================================================

class AIDEAgent(BaseAgent):
    """AIDE tree-search agent adapted for FML-bench.

    Search policy (from AIDE agent.py:61-92):
        1. If fewer than ``num_drafts`` drafts exist -> draft a new solution.
        2. With probability ``debug_prob``, pick a random debuggable buggy leaf.
        3. Otherwise, greedily improve the best good node.
        4. If no good nodes exist, fall back to drafting.

    Node lifecycle:
        - Draft: restore baseline, build instruction, edit, eval.
        - Improve: restore parent's snapshot, build instruction, edit, eval.
        - Debug: restore parent's snapshot, build instruction with error, edit, eval.
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.journal: Optional[Journal] = None
        self.baseline_snapshot: Dict[str, str] = {}
        self.task_description: str = ""
        self.baseline_results_dict: dict = {}

        # Agent params with defaults
        self.num_drafts: int = config.agent_params.get("num_drafts", 5)
        self.debug_prob: float = config.agent_params.get("debug_prob", 0.5)
        self.max_debug_depth: int = config.agent_params.get("max_debug_depth", 3)
        self.max_stderr_output: int = config.agent_params.get("max_stderr_output", 1500)

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize agent. Editor and executor are created in run()."""
        pass

    def run(self,
            task_description: str,
            target_files: List[str],
            baseline_results: dict) -> AgentResult:
        """Main AIDE tree-search loop.

        Returns:
            AgentResult with all steps, best step, and test result.
        """
        self.task_description = task_description
        self.target_files = target_files
        metrics_cfg = self.config.runtime_params.get("metrics", {})
        self.baseline_results_dict = get_filtered_results_for_prompt(baseline_results, metrics_cfg)
        benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        self.metric_name = benchmark_config.get("metric", "")
        self.metric_direction = benchmark_config.get("metric_direction", "higher")
        include_datasets = metrics_cfg.get("include_datasets")
        self.baseline_primary_metric = extract_primary_metric(
            baseline_results, self.metric_name, include_datasets,
        )

        # --- workspace & executor setup ---
        benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        agent_name = self.config.runtime_params.get("agent_name", "aide")
        benchmark_name = self.config.runtime_params.get("benchmark_name", "benchmark")
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = self.config.runtime_params.get("output_dir", "benchmark_results")
        parent_workspace = os.path.join(
            self._output_dir, agent_name, benchmark_name, experiment_timestamp
        )
        os.makedirs(parent_workspace, exist_ok=True)
        self._parent_workspace = parent_workspace

        # Create main executor
        timeout = self.config.agent_params.get("execute_timeout", 2400)
        experiment_name = f"{experiment_timestamp}_aide_search"
        self.executor = BenchmarkExecutor(
            benchmark_config,
            agent_name,
            benchmark_name,
            experiment_name,
            parent_timestamp=experiment_timestamp,
            timeout=timeout,
            output_dir=self._output_dir,
        )
        workspace = self.executor.setup_workspace()
        logger.info("AIDE workspace: %s", workspace)

        # Create CodeEditor
        log_dir = os.path.join(workspace, "code_editor_logs")
        metric_direction = benchmark_config.get("metric_direction", "higher")
        metric_name = benchmark_config.get("metric", "")
        self.editor = CodeEditor(
            model=self.config.model,
            provider=self.config.provider,
            target_files=self.target_files,
            task_description=self.task_description,
            log_dir=log_dir,
            metric_name=metric_name,
            metric_direction=metric_direction,
        )

        # LLM client for analysis calls
        from agents.llm import create_client, get_response_from_llm
        self.client, _ = create_client(self.config.model, self.config.provider)

        # Initialize journal
        self.journal = Journal(metric_direction=metric_direction)

        # Snapshot baseline code
        self.baseline_snapshot = self._snapshot_target_files()

        # --- main search loop ---
        all_steps: List[StepResult] = []

        while self.budget_remaining():
            parent_node = self._search_policy()

            if parent_node is None:
                node = self._draft()
            elif parent_node.is_buggy:
                node = self._debug(parent_node)
            else:
                node = self._improve(parent_node)

            step_result = self._make_step_result(node)
            all_steps.append(step_result)

        # --- final test ---
        test_result = self._run_final_test(
            benchmark_config, agent_name, benchmark_name, experiment_timestamp
        )

        # --- build result ---
        best_step = None
        if self.best_code_snapshot is not None:
            for s in all_steps:
                if s.primary_metric == self.best_metric:
                    best_step = s
                    break

        return AgentResult(
            all_steps=all_steps,
            best_step=best_step,
            test_result=test_result,
            total_steps=self.step_count,
            total_ideas=len(self.journal.draft_nodes) if self.journal else 0,
            token_usage=self.get_token_usage_summary(),
            parent_workspace=parent_workspace,
        )

    def _run_final_test(self, benchmark_config: dict, agent_name: str,
                        benchmark_name: str, experiment_timestamp: str) -> Optional[dict]:
        """Run final test using the best code snapshot."""
        if self.best_code_snapshot is None:
            logger.info("No best code snapshot found, skipping test execution.")
            return None

        try:
            timeout = self.config.agent_params.get("execute_timeout", 2400)
            self.executor = BenchmarkExecutor(
                benchmark_config,
                agent_name,
                benchmark_name,
                f"{experiment_timestamp}_final_test",
                parent_timestamp=experiment_timestamp,
                timeout=timeout,
                output_dir=self._output_dir,
            )
            self.executor.setup_workspace()

            # _execute_test restores best code and runs test
            test_result = self._execute_test()
            return test_result
        except Exception as e:
            logger.error("Error during final test execution: %s", e)
            return {"success": False, "error": str(e)}
        finally:
            if self.executor:
                self.executor.cleanup()
                self.executor = None

    # ------------------------------------------------------------------
    # Search policy (from aide/agent.py:61-92)
    # ------------------------------------------------------------------

    def _search_policy(self) -> Optional[TreeNode]:
        """Select a parent node to work on, or None to draft a new solution."""
        assert self.journal is not None

        # Initial drafting phase
        if len(self.journal.draft_nodes) < self.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # Debugging with probability debug_prob
        if random.random() < self.debug_prob:
            debuggable = [
                n for n in self.journal.buggy_nodes
                if n.is_leaf and n.debug_depth <= self.max_debug_depth
            ]
            if debuggable:
                logger.debug("[search policy] debugging a buggy node")
                return random.choice(debuggable)

        # Greedy improvement of best good node
        good = self.journal.good_nodes
        if not good:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        best = self.journal.get_best_node()
        logger.debug("[search policy] improving best node")
        return best

    # ------------------------------------------------------------------
    # Node lifecycle: draft / improve / debug
    # ------------------------------------------------------------------

    def _draft(self) -> TreeNode:
        """Create a new draft solution from baseline."""
        assert self.journal is not None
        _step_t0 = time.monotonic()
        _token_start = len(self.token_usage_log)

        # Restore baseline
        self._restore_snapshot(self.baseline_snapshot)

        # Build instruction
        memory = self.journal.generate_summary(self.metric_name)

        instruction = (
            f"You are an expert ML researcher. Your goal is to improve the ML algorithm "
            f"to achieve better performance.\n\n"
            f"## Baseline results\n"
            f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}\n\n"
            f"## Memory of previous attempts\n{memory}\n\n"
            f"## Instructions\n"
            f"Propose a novel approach that differs from previous attempts (listed in Memory). "
            f"Make targeted changes to improve the metric. "
            f"Keep the changes minimal and focused on a single idea."
        )

        node = TreeNode(action="draft", plan=instruction[:300])
        edit_success = self._edit_code(instruction)
        node._edit_success = edit_success
        node._editor_log_path = getattr(self._last_edit_result, "log_path", None)
        node._instruction = instruction

        # Snapshot after edit
        node.code_snapshot = self._snapshot_target_files()

        # Execute validation
        run_id = self.step_count  # step_count will be incremented inside _execute_val
        val_result = self._execute_val(run_id)
        node.val_duration = self._last_val_duration  # store per-node for StepResult

        # Update node with results
        node.val_result = val_result
        node.is_buggy = (not val_result["success"]) or (val_result.get("primary_metric") is None)
        if not node.is_buggy:
            node.primary_metric = val_result["primary_metric"]
        if val_result.get("error"):
            node.error_context = str(val_result["error"])[:self.max_stderr_output]

        self._analyze_execution(node, val_result)
        node._token_start = _token_start
        node._step_duration = time.monotonic() - _step_t0
        self.journal.append(node)
        return node

    def _improve(self, parent_node: TreeNode) -> TreeNode:
        """Improve a good parent node."""
        assert self.journal is not None
        _step_t0 = time.monotonic()
        _token_start = len(self.token_usage_log)

        # Restore parent's code
        self._restore_snapshot(parent_node.code_snapshot)

        memory = self.journal.generate_summary(self.metric_name)

        parent_metric_str = self._format_metric_line(
            parent_node.primary_metric, "Last validation result"
        )
        analysis_str = (
            f"\nAnalysis: {parent_node.analysis}\n"
            if parent_node.analysis else ""
        )

        instruction = (
            f"You are an expert ML researcher. You have a working solution and need to "
            f"improve it further.\n\n"
            f"## Last validation result\n{parent_metric_str}\n"
            f"{analysis_str}\n"
            f"## Memory of previous attempts\n{memory}\n\n"
            f"## Instructions\n"
            f"Propose a single, specific improvement to the current solution. "
            f"The change should be atomic so we can evaluate its effect. "
            f"Do not repeat approaches that already appear in Memory."
        )

        node = TreeNode(action="improve", parent=parent_node, plan=instruction[:300])
        edit_success = self._edit_code(instruction)
        node._edit_success = edit_success
        node._editor_log_path = getattr(self._last_edit_result, "log_path", None)
        node._instruction = instruction

        # Snapshot after edit
        node.code_snapshot = self._snapshot_target_files()

        # Execute validation
        run_id = self.step_count
        val_result = self._execute_val(run_id)
        node.val_duration = self._last_val_duration  # store per-node for StepResult

        node.val_result = val_result
        node.is_buggy = (not val_result["success"]) or (val_result.get("primary_metric") is None)
        if not node.is_buggy:
            node.primary_metric = val_result["primary_metric"]
        if val_result.get("error"):
            node.error_context = str(val_result["error"])[:self.max_stderr_output]

        self._analyze_execution(node, val_result)
        node._token_start = _token_start
        node._step_duration = time.monotonic() - _step_t0
        self.journal.append(node)
        return node

    def _debug(self, parent_node: TreeNode) -> TreeNode:
        """Debug a buggy parent node."""
        assert self.journal is not None
        _step_t0 = time.monotonic()
        _token_start = len(self.token_usage_log)

        # Restore parent's code (the buggy version)
        self._restore_snapshot(parent_node.code_snapshot)

        # Build debug ancestor chain (A3)
        ancestors = []
        node_iter = parent_node
        while node_iter and node_iter.stage_name == "debug":
            ancestors.append(
                f"- Attempt: {node_iter.plan[:150]}\n"
                f"  Error: {(node_iter.error_context or '')[:100]}"
            )
            node_iter = node_iter.parent
        ancestor_section = ""
        if ancestors:
            ancestor_section = (
                "\n\n## Previous debug attempts\n"
                + "\n".join(reversed(ancestors))
            )

        error_output = parent_node.error_context or "(No error output captured.)"

        bug_analysis_section = ""
        if parent_node.analysis:
            bug_analysis_section = f"\n\n## Bug analysis\n{parent_node.analysis}"

        instruction = (
            f"You are an expert ML researcher. Your previous solution had a bug. "
            f"Fix the issue based on the error context provided."
            f"{bug_analysis_section}\n\n"
            f"## Instructions\n"
            f"Fix the bug while preserving the original intent of the code. "
            f"Make minimal changes to resolve the error."
            f"{ancestor_section}"
        )

        node = TreeNode(action="debug", parent=parent_node, plan=instruction[:300])
        edit_success = self._edit_code(instruction, error_context=error_output)
        node._edit_success = edit_success
        node._editor_log_path = getattr(self._last_edit_result, "log_path", None)
        node._instruction = instruction

        # Snapshot after edit
        node.code_snapshot = self._snapshot_target_files()

        # Execute validation
        run_id = self.step_count
        val_result = self._execute_val(run_id)
        node.val_duration = self._last_val_duration  # store per-node for StepResult

        node.val_result = val_result
        node.is_buggy = (not val_result["success"]) or (val_result.get("primary_metric") is None)
        if not node.is_buggy:
            node.primary_metric = val_result["primary_metric"]
        if val_result.get("error"):
            node.error_context = str(val_result["error"])[:self.max_stderr_output]

        self._analyze_execution(node, val_result)
        node._token_start = _token_start
        node._step_duration = time.monotonic() - _step_t0
        self.journal.append(node)
        return node

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze_execution(self, node: TreeNode, val_result: dict) -> None:
        """LLM analysis of execution results. NOT counted as a step."""
        from agents.llm import get_response_from_llm

        success = val_result.get("success", False)
        metric = val_result.get("primary_metric")
        error = val_result.get("error", "")
        if error and len(error) > self.max_stderr_output:
            error = "..." + error[-self.max_stderr_output:]

        prompt = (
            "You have written code for your research experiment and now need to "
            "evaluate the output of the code execution. "
            "Analyze the execution output, determine if there were any bugs, "
            "and provide a summary of the findings.\n\n"
            f"## Task\n{self.task_description}\n\n"
            f"## Plan\n{node.plan[:500]}\n\n"
            f"## Execution Result\n"
            f"Success: {success}\n"
            f"Primary Metric ({self.metric_name}): {metric}\n"
        )
        if error:
            prompt += f"Error output:\n```\n{error}\n```\n\n"
        prompt += (
            "If there is a bug, summarize the bug and propose a fix. "
            "Otherwise, provide a 2-3 sentence summary of findings."
        )
        try:
            text, _, usage = get_response_from_llm(
                prompt, client=self.client, model=self.config.model,
                system_message=(
                    "You are an experienced AI researcher evaluating "
                    "experiment results."
                ),
            )
            if usage:
                self.token_usage_log.append(usage)
            node.analysis = text.strip()[:500]
        except Exception as e:
            node.analysis = f"Analysis failed: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_step_result(self, node: TreeNode) -> StepResult:
        """Convert a TreeNode into a StepResult for the AgentResult."""
        idea_id = node.node_id[:8]
        # Determine the root draft node for idea_description
        root = node
        while root.parent is not None:
            root = root.parent
        idea_description = root.plan or f"Draft {idea_id}"

        # Use step_count (1-based, from _execute_val) for consistency with other agents
        step_id = self.step_count
        token_start = getattr(node, "_token_start", None)
        step_tokens = self._collect_step_tokens(token_start) if token_start is not None else None
        snap_path = self._save_step_code_snapshot(
            step_id, self._parent_workspace, snapshot=node.code_snapshot
        )

        meta = {
            "node_id": node.node_id,
            "parent_id": node.parent.node_id if node.parent else "",
            "is_buggy": str(node.is_buggy),
            "debug_depth": str(node.debug_depth),
            "code_snapshot_path": snap_path,
            "instruction": getattr(node, "_instruction", None),
            "analysis": node.analysis if node.analysis else None,
            "editor_log_path": getattr(node, "_editor_log_path", None),
        }

        return StepResult(
            step_id=step_id,
            idea_id=idea_id,
            idea_description=idea_description,
            action=node.stage_name,
            edit_success=getattr(node, "_edit_success", False),
            val_result=node.val_result,
            primary_metric=node.primary_metric,
            token_usage=step_tokens,
            step_duration_seconds=getattr(node, "_step_duration", None),
            metadata=meta,
        )
