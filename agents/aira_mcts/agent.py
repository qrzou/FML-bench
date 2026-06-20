"""
AIRA-MCTS agent: Monte Carlo Tree Search solver for FML-bench.

Ported from the AIRA-dojo MCTS solver (https://github.com/facebookresearch/aira-dojo, src/dojo/solvers/mcts/mcts.py).
Uses UCT selection, draft/improve/debug operators via CodeEditor, and backpropagation.
"""
import math
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ..base import BaseAgent, AgentConfig, AgentResult, StepResult
from ..code_editor import CodeEditor
from benchmark.executor_factory import make_executor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt


# ---------------------------------------------------------------------------
# UCT helpers (ported from mcts.py:44-65)
# ---------------------------------------------------------------------------

def normalise_q_value(q_value: float, global_max_q: float, global_min_q: float) -> float:
    """Normalise q-value to [0, 1] range."""
    if global_max_q == global_min_q:
        return 0.5
    return (q_value - global_min_q) / (global_max_q - global_min_q)


def uct_value(
    node: "MCTSNode",
    parent: "MCTSNode",
    uct_c: float,
    global_max_q: float,
    global_min_q: float,
    direction: str,
) -> float:
    """Compute UCT value for a child node."""
    if node.explore_count == 0:
        return -1e8
    norm_q = normalise_q_value(node.q_value(direction), global_max_q, global_min_q)
    exploration = math.sqrt(math.log(parent.explore_count) / node.explore_count)
    return norm_q + uct_c * exploration


def search_policy(
    root: "MCTSNode",
    uct_c: float,
    global_max_q: float,
    global_min_q: float,
    direction: str,
) -> List["MCTSNode"]:
    """Traverse from root to leaf using UCT selection.  Returns path list."""
    path: List[MCTSNode] = []
    node = root
    while True:
        path.append(node)
        if not node.children:
            return path
        node = max(
            node.children,
            key=lambda c: uct_value(c, node, uct_c, global_max_q, global_min_q, direction),
        )


# ---------------------------------------------------------------------------
# MCTSNode  (ported from mcts.py:68-105)
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the MCTS tree."""

    _id_counter = 0

    def __init__(
        self,
        plan: str = "",
        action: str = "",
        code_snapshot: Optional[Dict[str, str]] = None,
        parent: Optional["MCTSNode"] = None,
        val_result: Optional[Dict[str, Any]] = None,
        primary_metric: Optional[float] = None,
        is_buggy: bool = False,
        error_context: Optional[str] = None,
        step_id: Optional[int] = None,
    ):
        MCTSNode._id_counter += 1
        self.node_id: int = MCTSNode._id_counter
        self.step_id: Optional[int] = step_id
        self.plan: str = plan
        self.action: str = action  # "draft" | "improve" | "debug"
        self.code_snapshot: Optional[Dict[str, str]] = code_snapshot
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.val_result: Optional[Dict[str, Any]] = val_result
        self.primary_metric: Optional[float] = primary_metric
        self.is_buggy: bool = is_buggy
        self.error_context: Optional[str] = error_context
        self.analysis: str = ""

        # MCTS statistics
        self.explore_count: int = 0
        self.cumulative_value: float = 0.0

    # -- MCTS value methods -------------------------------------------------

    def q_value(self, direction: str) -> float:
        """Average value.  *direction* is 'higher' or 'lower'."""
        if self.explore_count == 0:
            return float("-inf")
        q = self.cumulative_value / self.explore_count
        if direction == "lower":
            q = -q
        return q

    def add_value(self, v: float) -> None:
        if v is not None:
            self.cumulative_value += v

    def increment_explore_count(self) -> None:
        self.explore_count += 1

    # -- Tree properties ----------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def stage_name(self) -> str:
        return self.action or "root"

    @property
    def debug_depth(self) -> int:
        """Count consecutive debug ancestors (including self)."""
        depth = 0
        node = self
        while node is not None and node.action == "debug":
            depth += 1
            node = node.parent
        return depth

    def __repr__(self) -> str:
        return (
            f"MCTSNode(id={self.node_id}, action={self.action}, "
            f"metric={self.primary_metric}, buggy={self.is_buggy}, "
            f"visits={self.explore_count})"
        )


# ---------------------------------------------------------------------------
# Journal  (simplified from journal.py)
# ---------------------------------------------------------------------------

class Journal:
    """Append-only log of all MCTSNodes."""

    def __init__(self):
        self.nodes: List[MCTSNode] = []

    def append(self, node: MCTSNode) -> None:
        self.nodes.append(node)

    # -- Filtered views -----------------------------------------------------

    @property
    def draft_nodes(self) -> List[MCTSNode]:
        return [n for n in self.nodes if n.action == "draft"]

    @property
    def buggy_nodes(self) -> List[MCTSNode]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[MCTSNode]:
        return [n for n in self.nodes if not n.is_buggy and n.primary_metric is not None]

    def get_best_node(self, direction: str) -> Optional[MCTSNode]:
        """Return the node with the best primary_metric."""
        good = self.good_nodes
        if not good:
            return None
        if direction == "higher":
            return max(good, key=lambda n: n.primary_metric)
        else:
            return min(good, key=lambda n: n.primary_metric)

    # -- Memory context for LLM prompts ------------------------------------

    def build_memory_context(self, direction: str,
                             metric_name: str = "metric") -> str:
        """Summarise good nodes for LLM context (memory operator)."""
        good = self.good_nodes
        if not good:
            return "(No successful attempts yet.)"
        # Sort best first
        good_sorted = sorted(
            good,
            key=lambda n: n.primary_metric,
            reverse=(direction == "higher"),
        )
        lines = []
        for n in good_sorted[:10]:  # top-10 summaries
            lines.append(
                f"- [Node {n.node_id}] Action={n.action}, "
                f"Plan: {n.plan[:200]}, Analysis: {n.analysis[:200]}, "
                f"{metric_name}={n.primary_metric}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AIRAMCTSAgent
# ---------------------------------------------------------------------------

class AIRAMCTSAgent(BaseAgent):
    """
    AIRA-MCTS agent -- Monte Carlo Tree Search for iterative ML code improvement.

    Uses UCT selection to balance exploration / exploitation, with
    draft (new approach), improve (refine working code), and debug
    (fix broken code) operators implemented via LLM calls + CodeEditor pattern.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # MCTS hyper-parameters (from agent_params)
        self.uct_c: float = 1.41
        self.num_children: int = 2
        self.max_debug_depth: int = 3
        self.max_stderr_output: int = 1500

        # Search state
        self.journal: Optional[Journal] = None
        self.root_node: Optional[MCTSNode] = None
        self.global_max_q: float = float("-inf")
        self.global_min_q: float = float("inf")

        # Task / benchmark references
        self.benchmark_config: Dict[str, Any] = {}
        self.repo_dir: str = ""
        self.metric_direction: str = "higher"
        self.task_description: str = ""
        self.baseline_results: Dict[str, Any] = {}
        self.experiment_timestamp: str = ""
        self.parent_workspace: str = ""
        self.agent_name: str = ""
        self.benchmark_name: str = ""

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def initialize(self) -> None:
        """Initialise LLM client and load agent_params."""
        params = self.config.agent_params
        self.uct_c = float(params.get("uct_c", 1.41))
        self.num_children = int(params.get("num_children", 2))
        self.max_debug_depth = int(params.get("max_debug_depth", 3))
        self.max_stderr_output = int(params.get("max_stderr_output", 1500))
        self.step_budget = int(params.get("max_steps", params.get("max_iter", 50)))

    def run(
        self,
        task_description: Optional[Any] = None,
        target_files: Optional[List[str]] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Main entry-point -- run the full MCTS search loop."""
        # ---- unpack inputs ------------------------------------------------
        if isinstance(task_description, tuple):
            self.task_description, _ = task_description
        else:
            self.task_description = task_description or ""

        self.target_files = list(target_files or [])
        metrics_cfg = self.config.runtime_params.get("metrics", {})
        self.baseline_results = get_filtered_results_for_prompt(baseline_results or {}, metrics_cfg)

        self.benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        self.repo_dir = self.config.runtime_params.get("repo_dir", self.benchmark_config.get("repo_dir", ""))
        self.agent_name = self.config.runtime_params.get("agent_name", "aira_mcts")
        self.benchmark_name = self.config.runtime_params.get("benchmark_name", "benchmark")

        self.metric_direction = self.benchmark_config.get("metric_direction", "higher")
        self.metric_name = self.benchmark_config.get("metric", "")
        include_datasets = metrics_cfg.get("include_datasets")
        self.baseline_primary_metric = extract_primary_metric(
            baseline_results or {}, self.metric_name, include_datasets
        )

        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = self.config.runtime_params.get("output_dir", "benchmark_results")
        self.parent_workspace = os.path.join(
            self._output_dir, self.agent_name, self.benchmark_name, self.experiment_timestamp
        )
        os.makedirs(self.parent_workspace, exist_ok=True)

        # ---- create executor (for inherited _execute_val / _execute_test) --
        timeout = self.config.agent_params.get("execute_timeout", 2400)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{timestamp}_mcts_search"
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
        self.executor.setup_workspace()

        # ---- create editor (for inherited _edit_code) ----------------------
        self.editor = CodeEditor(
            model=self.config.model,
            provider=self.config.provider,
            target_files=self.target_files,
            task_description=self.task_description,
            log_dir=self.parent_workspace,
            metric_name=self.benchmark_config.get("metric", ""),
            metric_direction=self.metric_direction,
        )

        # ---- LLM client for analysis calls --------------------------------
        from agents.llm import create_client, get_response_from_llm
        self.client, _ = create_client(self.config.model, self.config.provider)

        # ---- initialise MCTS structures -----------------------------------
        self.journal = Journal()
        self.step_count = 0
        MCTSNode._id_counter = 0

        baseline_snapshot = self._snapshot_target_files()
        self.root_node = MCTSNode(
            plan="baseline (root)",
            action="root",
            code_snapshot=baseline_snapshot,
            is_buggy=True,
            primary_metric=None,
        )
        self.journal.append(self.root_node)

        # ---- main MCTS loop -----------------------------------------------
        print(f"\n{'='*60}")
        print(f"AIRA-MCTS search  |  budget={self.step_budget}  uct_c={self.uct_c}  "
              f"children={self.num_children}  debug_depth={self.max_debug_depth}")
        print(f"{'='*60}\n")

        while self.budget_remaining():
            path = search_policy(
                self.root_node,
                self.uct_c,
                self.global_max_q,
                self.global_min_q,
                self.metric_direction,
            )
            self._expand_leaf_and_backprop(path)

        # ---- final test run ------------------------------------------------
        best_node = self.journal.get_best_node(self.metric_direction)
        test_result = None
        if best_node is not None:
            print(f"\nBest node: {best_node}")
            self._restore_snapshot(best_node.code_snapshot)
            # Store as best_code_snapshot so inherited _execute_test restores it
            self.best_code_snapshot = best_node.code_snapshot
            # Clean up search executor before creating test executor
            if self.executor:
                self.executor.cleanup()
            # Create a fresh executor for the test phase
            test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_backend = self.config.runtime_params.get("eval_backend", "local")
            self.executor = make_executor(
                self.benchmark_config,
                self.agent_name,
                self.benchmark_name,
                f"{test_timestamp}_final_test",
                parent_timestamp=self.experiment_timestamp,
                timeout=timeout,
                output_dir=self._output_dir,
                eval_backend=eval_backend,
            )
            self.executor.setup_workspace()
            test_result = self._execute_test()
        else:
            print("\nNo successful node found during search.")

        # ---- cleanup executor ----------------------------------------------
        if self.executor:
            self.executor.cleanup()
            self.executor = None

        # ---- collect results -----------------------------------------------
        return self._collect_results(best_node, test_result)

    # =====================================================================
    # LLM operators: draft / improve / debug
    # =====================================================================

    def _draft(self, parent: MCTSNode) -> MCTSNode:
        """Generate a brand-new approach (child of root)."""
        memory = self.journal.build_memory_context(self.metric_direction, self.metric_name)
        instruction = (
            f"You are starting a new approach to improve the ML code for this task.\n\n"
            f"## Previous attempts summary\n{memory}\n\n"
            f"## Baseline\n{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}\n\n"
            "## Instructions\n"
            "Propose AND implement a novel algorithmic improvement.\n"
            "- Describe your plan in 3-5 sentences before implementing."
        )

        # Restore parent snapshot before editing
        self._restore_snapshot(parent.code_snapshot)

        edit_ok = self._edit_code(instruction)
        _editor_log = getattr(self._last_edit_result, "log_path", None)
        snapshot = self._snapshot_target_files()

        node = MCTSNode(
            plan=f"Draft #{len([c for c in parent.children if c.action == 'draft']) + 1}",
            action="draft",
            code_snapshot=snapshot,
            parent=parent,
        )
        node._edit_success = edit_ok
        node._editor_log_path = _editor_log
        node._instruction = instruction
        parent.children.append(node)

        if not edit_ok:
            node.is_buggy = True
            node.error_context = "Code editing failed (no file changes)."
        return node

    def _improve(self, parent: MCTSNode) -> MCTSNode:
        """Improve an existing working solution."""
        memory = self.journal.build_memory_context(self.metric_direction, self.metric_name)

        # Summarise parent's result
        parent_summary = (
            f"Parent node {parent.node_id}: "
            f"{self._format_metric_line(parent.primary_metric, 'Metric')}, "
            f"action={parent.action}"
        )
        if parent.analysis:
            parent_summary += f"\nAnalysis: {parent.analysis}"

        instruction = (
            f"You are improving an existing ML solution.\n\n"
            f"## Current solution info\n{parent_summary}\n"
            f"The current code in the target files reflects this solution's modifications.\n\n"
            f"## Previous attempts summary\n{memory}\n\n"
            "## Instructions\n"
            "Make a single, targeted improvement to boost the metric further.\n"
            "- Focus on one specific change, not multiple simultaneous modifications.\n"
            "- Do not repeat approaches that have already been tried (see previous attempts).\n"
            "- Describe your improvement plan in 3-5 sentences before implementing."
        )

        # Restore parent snapshot before editing
        self._restore_snapshot(parent.code_snapshot)

        edit_ok = self._edit_code(instruction)
        _editor_log = getattr(self._last_edit_result, "log_path", None)
        snapshot = self._snapshot_target_files()

        node = MCTSNode(
            plan=f"Improve from node {parent.node_id}",
            action="improve",
            code_snapshot=snapshot,
            parent=parent,
        )
        node._edit_success = edit_ok
        node._editor_log_path = _editor_log
        node._instruction = instruction
        parent.children.append(node)

        if not edit_ok:
            node.is_buggy = True
            node.error_context = "Code editing failed (no file changes)."
        return node

    def _debug(self, buggy_node: MCTSNode) -> MCTSNode:
        """Attempt to fix a buggy node."""
        error_ctx = buggy_node.error_context or ""

        # Build debug ancestor chain (A3)
        ancestors = []
        node_iter = buggy_node
        while node_iter and node_iter.action == "debug":
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

        # Include analysis from the buggy node if available
        analysis_section = ""
        if buggy_node.analysis:
            analysis_section = f"\n\n## Analysis of buggy attempt\n{buggy_node.analysis}"

        # Include error context inline so the LLM sees it in both instruction and _edit_code param
        error_section = ""
        if error_ctx:
            error_section = f"\n\n## Error output\n{error_ctx[:self.max_stderr_output]}"

        instruction = (
            f"The previous code change caused an error. Please fix it.\n\n"
            "## Instructions\n"
            "Fix the bug while preserving the intended improvement.\n"
            "- Read the error message carefully and identify the root cause.\n"
            "- Make minimal changes to fix the issue."
            f"{analysis_section}"
            f"{error_section}"
            f"{ancestor_section}"
        )

        # Restore buggy node's snapshot, then attempt fix
        self._restore_snapshot(buggy_node.code_snapshot)

        edit_ok = self._edit_code(instruction, error_context=error_ctx)
        _editor_log = getattr(self._last_edit_result, "log_path", None)
        snapshot = self._snapshot_target_files()

        node = MCTSNode(
            plan=f"Debug fix for node {buggy_node.node_id}",
            action="debug",
            code_snapshot=snapshot,
            parent=buggy_node,
        )
        node._edit_success = edit_ok
        node._editor_log_path = _editor_log
        node._instruction = instruction
        buggy_node.children.append(node)

        if not edit_ok:
            node.is_buggy = True
            node.error_context = "Debug code editing failed (no file changes)."
        return node

    # =====================================================================
    # Node evaluation
    # =====================================================================

    def _evaluate_node(self, node: MCTSNode) -> None:
        """Run _execute_val and populate the node's result fields.

        Always runs validation even if the edit failed, matching the
        original AIRA-MCTS behaviour where every generated child is
        evaluated (and consumes a step budget unit).
        """
        # Restore node's code before executing
        self._restore_snapshot(node.code_snapshot)

        run_id = f"step_{self.step_count}"
        result = self._execute_val(run_id)
        node.val_duration = self._last_val_duration  # store per-node for StepResult

        node.step_id = self.step_count
        node.val_result = result
        node.is_buggy = (not result.get("success", False)) or (result.get("primary_metric") is None)
        node.primary_metric = result.get("primary_metric")
        node.error_context = result.get("error", "")

        # Take a snapshot after successful evaluation so we capture the exact code state
        if not node.is_buggy:
            node.code_snapshot = self._snapshot_target_files()

        if node.is_buggy:
            print(f"  Node {node.node_id} BUGGY  |  error: {str(node.error_context)[:200]}")
        else:
            print(f"  Node {node.node_id} OK     |  metric={node.primary_metric}")

    # =====================================================================
    # Execution analysis (non-step LLM call)
    # =====================================================================

    def _analyze_execution(self, node: MCTSNode, val_result: Dict[str, Any]) -> None:
        """Ask LLM to analyze execution results.  Does NOT count as a step."""
        try:
            from agents.llm import get_response_from_llm

            metric_val = node.primary_metric
            prompt = (
                f"Analyze these ML experiment results briefly.\n\n"
                f"## Task\n{self.task_description}\n\n"
                f"Action: {node.action}\n"
                f"Plan: {node.plan[:300]}\n"
                f"{self._format_metric_line(metric_val, 'Execution result')}\n"
                f"Success: {not node.is_buggy}\n"
                f"Error: {(node.error_context or '')[:300]}\n\n"
                "Provide a concise analysis (2-3 sentences): what worked, what didn't, "
                "and one concrete suggestion for the next iteration."
            )
            text, _, usage = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.config.model,
                system_message="You are an ML experiment analyst. Be concise and actionable.",
                print_debug=False,
            )
            if usage:
                self.token_usage_log.append(usage)
            node.analysis = text.strip()[:500]
        except Exception as e:
            node.analysis = f"(analysis failed: {e})"

    # =====================================================================
    # Backpropagation  (from mcts.py:447-454)
    # =====================================================================

    def _backprop(self, path: List[MCTSNode], value: float) -> None:
        """Backpropagate value along the path."""
        for node in path:
            node.increment_explore_count()
            node.add_value(value)

    def _update_global_q(self, value: Optional[float]) -> None:
        if value is not None:
            self.global_max_q = max(self.global_max_q, value)
            self.global_min_q = min(self.global_min_q, value)

    # =====================================================================
    # Expand + backprop  (from mcts.py:472-554)
    # =====================================================================

    def _expand_leaf_and_backprop(self, path: List[MCTSNode]) -> None:
        """Expand a leaf node and backpropagate results."""
        leaf = path[-1]
        children_to_create = min(self.num_children, self.step_budget - self.step_count)

        for _ in range(children_to_create):
            if not self.budget_remaining():
                break

            _token_start = len(self.token_usage_log)
            _step_t0 = time.monotonic()

            # Draft from root, improve from non-root
            if leaf is self.root_node:
                child = self._draft(leaf)
            else:
                child = self._improve(leaf)

            # Evaluate
            self._evaluate_node(child)
            self._analyze_execution(child, child.val_result or {})
            child._token_start = _token_start
            child._step_duration = time.monotonic() - _step_t0
            self.journal.append(child)

            if not child.is_buggy:
                # Successful -- backprop metric value
                value = child.primary_metric if child.primary_metric is not None else 0.0
                self._backprop(path + [child], value)
                self._update_global_q(child.primary_metric)
            else:
                # Enter debug cycle
                debug_path, fixed_metric = self._debug_cycle(child)
                if fixed_metric is not None:
                    self._backprop(path + debug_path, fixed_metric)
                    self._update_global_q(fixed_metric)
                # else: no backprop (complete failure)

            if not self.budget_remaining():
                break

    # =====================================================================
    # Debug cycle  (from mcts.py:522-554)
    # =====================================================================

    def _debug_cycle(self, buggy_node: MCTSNode) -> Tuple[List[MCTSNode], Optional[float]]:
        """
        Attempt to fix a buggy node up to max_debug_depth times.

        Returns:
            (debug_path, fixed_metric)  -- fixed_metric is None if never fixed.
        """
        debug_path: List[MCTSNode] = [buggy_node]
        fixed_metric: Optional[float] = None
        current = buggy_node

        for _ in range(self.max_debug_depth):
            if not self.budget_remaining():
                break
            if current.debug_depth >= self.max_debug_depth:
                break

            _dbg_token_start = len(self.token_usage_log)
            _dbg_t0 = time.monotonic()
            child = self._debug(current)
            self._evaluate_node(child)
            self._analyze_execution(child, child.val_result or {})
            child._token_start = _dbg_token_start
            child._step_duration = time.monotonic() - _dbg_t0
            self.journal.append(child)
            debug_path.append(child)

            if not child.is_buggy and child.primary_metric is not None:
                fixed_metric = child.primary_metric
                break

            current = child

        return debug_path, fixed_metric

    # =====================================================================
    # Result collection
    # =====================================================================

    def _collect_results(
        self, best_node: Optional[MCTSNode], test_result: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """Package final results as an AgentResult."""
        all_step_results: List[StepResult] = []
        for n in self.journal.nodes:
            if n.action == "root":
                continue  # skip root sentinel node
            sid = n.step_id if n.step_id is not None else 0
            token_start = getattr(n, "_token_start", None)
            step_tokens = self._collect_step_tokens(token_start) if token_start is not None else None
            # Pass node's own code_snapshot (disk state may differ after loop)
            snap_path = self._save_step_code_snapshot(
                sid, self.parent_workspace, snapshot=n.code_snapshot
            )
            meta = {
                "node_id": n.node_id,
                "is_buggy": n.is_buggy,
                "explore_count": n.explore_count,
                "cumulative_value": n.cumulative_value,
                "code_snapshot_path": snap_path,
                "instruction": getattr(n, "_instruction", None),
                "analysis": n.analysis if n.analysis else None,
                "editor_log_path": getattr(n, "_editor_log_path", None),
            }
            all_step_results.append(StepResult(
                step_id=sid,
                idea_id=f"node_{n.node_id}",
                idea_description=n.plan[:200] if n.plan else "",
                action=n.action,
                edit_success=getattr(n, "_edit_success", False),
                val_result=n.val_result,
                primary_metric=n.primary_metric,
                token_usage=step_tokens,
                step_duration_seconds=getattr(n, "_step_duration", None),
                metadata=meta,
            ))

        # Find best step
        best_step = None
        if best_node is not None:
            for s in all_step_results:
                if s.primary_metric == best_node.primary_metric:
                    best_step = s
                    break

        # Persist summary
        summary: Dict[str, Any] = {
            "total_steps": self.step_count,
            "total_nodes": len(self.journal.nodes),
            "best_node": {
                "node_id": best_node.node_id,
                "action": best_node.action,
                "primary_metric": best_node.primary_metric,
            } if best_node else None,
        }
        summary_path = os.path.join(self.parent_workspace, "summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Results saved to {summary_path}")
        except Exception as e:
            print(f"Warning: could not save summary: {e}")

        return AgentResult(
            all_steps=all_step_results,
            best_step=best_step,
            test_result=test_result,
            total_steps=self.step_count,
            total_ideas=len(self.journal.draft_nodes),
            token_usage=self.get_token_usage_summary(),
            parent_workspace=self.parent_workspace,
            metadata=summary,
        )
