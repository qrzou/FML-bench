"""
AI Scientist v2 agent for FML-bench.

Faithfully ported from: https://github.com/sakanaai/ai-scientist-v2 (arXiv:2504.08066)

Core algorithm: BFTS (Best-First Tree Search) with 4-stage pipeline.

Original v2 key features (preserved):
  - BFTS tree search with draft/improve/debug operations
  - Simulated batch selection (num_parallel nodes per round, tree-level dedup)
  - 4-stage pipeline: basic → tuning → creative → ablation
  - Per-stage independent Journal ("intentional forgetting")
  - Inter-stage best-node carryover
  - Dynamic sub-stages with LLM goal generation
  - LLM-generated journal memory (successful + failed experiments analysis)
  - Post-execution LLM analysis -> node.analysis (not counted as step)
  - Structured prompts (Introduction + Research idea + Memory + Stage guidance)

Adapted for FML-bench:
  - Serial execution with batch selection (simulates parallel workers)
  - All ideas share one 4-stage run (original: per-idea independent 4-stage)
  - No VLM plot analysis / multi-seed evaluation / writeup
  - Code editing via CodeEditor (not internal plan_and_code_query)
  - Execution via BenchmarkExecutor (not Interpreter subprocess)
"""
import json
import os
import os.path as osp
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from benchmark.executor import BenchmarkExecutor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt

from ..base import AgentConfig, AgentResult, BaseAgent, StepResult
from ..code_editor import CodeEditor
from ..llm import create_client, get_response_from_llm


# ---------------------------------------------------------------------------
# 4-stage pipeline (from original agent_manager.py:143-167)
# ---------------------------------------------------------------------------

V2_STAGES = [
    {
        "name": "basic_implementation",
        "goals": (
            "Focus on getting a basic working implementation. "
            "Keep changes simple and incremental. Verify correctness first. "
            "Use a straightforward approach before any sophisticated improvements."
        ),
    },
    {
        "name": "hyperparameter_tuning",
        "goals": (
            "Focus on tuning hyperparameters such as learning rate, batch size, "
            "number of epochs, regularization, etc. to improve performance. "
            "DO NOT change the core algorithm architecture from the previous stage."
        ),
    },
    {
        "name": "creative_research",
        "goals": (
            "Explore novel and creative improvements. Try architectural changes, "
            "new techniques, or unconventional approaches. "
            "Be creative and think outside the box."
        ),
    },
    {
        "name": "ablation_refinement",
        "goals": (
            "Conduct systematic tests of your best solution's components. "
            "Identify which changes contribute most to performance. "
            "Refine the solution by removing unnecessary complexity."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data structures (from original journal.py)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class TreeNode:
    """A node in the BFTS solution tree."""

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    step_id: Optional[int] = None
    plan: str = ""
    action: str = ""  # "draft" | "improve" | "debug"
    code_snapshot: Dict[str, str] = field(default_factory=dict)
    parent: Optional["TreeNode"] = field(default=None, repr=False)
    children: list = field(default_factory=list, repr=False)
    val_result: Optional[dict] = None
    primary_metric: Optional[float] = None
    is_buggy: bool = True
    error_context: str = ""
    analysis: str = ""  # LLM analysis of execution results
    idea: Optional[Dict[str, Any]] = None  # Bound research idea (set on draft nodes)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.append(self)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def stage_name(self) -> str:
        if self.parent is None:
            return "draft"
        if self.parent.is_buggy:
            return "debug"
        return "improve"

    @property
    def debug_depth(self) -> int:
        depth = 0
        node = self
        while node is not None and node.parent is not None and node.parent.is_buggy:
            depth += 1
            node = node.parent
        return depth

    @property
    def root_idea(self) -> Optional[Dict[str, Any]]:
        """Trace parent chain to root and return its bound idea."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node.idea

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self):
        return hash(self.node_id)


class Journal:
    """Container for all BFTS nodes with memory generation."""

    def __init__(self, metric_direction: str = "higher"):
        self.nodes: List[TreeNode] = []
        self.metric_direction = metric_direction
        self._cached_summary: str = ""
        self._summary_version: int = -1

    def append(self, node: TreeNode) -> None:
        node.step_id = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self) -> Optional[TreeNode]:
        good = self.good_nodes
        if not good:
            return None
        if self.metric_direction == "higher":
            return max(good, key=lambda n: n.primary_metric if n.primary_metric is not None else float("-inf"))
        return min(good, key=lambda n: n.primary_metric if n.primary_metric is not None else float("inf"))

    def build_summary_input(self, metric_name: str = "metric") -> str:
        """Build raw input for LLM summary generation (successful + failed experiments)."""
        parts = []

        # Successful experiments
        good = self.good_nodes
        if good:
            parts.append("## Successful Experiments")
            for i, n in enumerate(good):
                parts.append(
                    f"\n### Experiment {i+1}\n"
                    f"Design: {n.plan[:300]}\n"
                    f"Results: {n.analysis}\n"
                    f"{metric_name}: {n.primary_metric}"
                )

        # Failed experiments
        buggy = self.buggy_nodes
        if buggy:
            parts.append("\n## Failed Experiments")
            for i, n in enumerate(buggy[:10]):  # cap at 10
                err_preview = (n.error_context[:200] if n.error_context else "unknown error")
                parts.append(
                    f"\n### Failed Experiment {i+1}\n"
                    f"Design: {n.plan[:200]}\n"
                    f"Error: {err_preview}\n"
                    f"Debug Depth: {n.debug_depth}"
                )

        if not parts:
            return ""
        return "\n".join(parts)

    def generate_summary(self, client, model: str, token_log: list,
                         metric_name: str = "metric") -> str:
        """LLM-generated memory summary. NOT counted as a step."""
        if len(self.nodes) == self._summary_version:
            return self._cached_summary

        raw_input = self.build_summary_input(metric_name)
        if not raw_input:
            self._cached_summary = "(No experiments completed yet.)"
            self._summary_version = len(self.nodes)
            return self._cached_summary

        prompt = (
            "You are an AI researcher summarizing experimental progress. "
            "Please analyze both successful and failed experiments to provide "
            "insights for future improvements.\n\n"
            f"{raw_input}\n\n"
            "Please provide a comprehensive summary that includes:\n"
            "1. Key patterns of success across working experiments\n"
            "2. Common failure patterns and pitfalls to avoid\n"
            "3. Specific recommendations for future experiments based on "
            "both successes and failures\n\n"
            "Keep the summary concise (10-15 sentences)."
        )
        try:
            text, _, usage = get_response_from_llm(
                prompt, client=client, model=model,
                system_message=(
                    "You are an experienced AI researcher summarizing "
                    "experimental progress."
                ),
            )
            if usage:
                token_log.append(usage)
            self._cached_summary = text.strip()[:1500]
        except Exception as e:
            # Fallback: structured summary without LLM
            lines = []
            for n in self.good_nodes[-5:]:
                lines.append(f"- [success] {n.plan[:150]} | {metric_name}={n.primary_metric}")
            for n in self.buggy_nodes[-3:]:
                lines.append(f"- [failed] {n.plan[:150]}")
            self._cached_summary = "\n".join(lines) if lines else "(No experiments yet.)"

        self._summary_version = len(self.nodes)
        return self._cached_summary


# ---------------------------------------------------------------------------
# AIScientistV2Agent
# ---------------------------------------------------------------------------

class AIScientistV2Agent(BaseAgent):
    """
    AI Scientist v2: BFTS tree search for automated ML research.
    Faithfully implements the search strategy from arXiv:2504.08066.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        self.all_steps: List[StepResult] = []

        # BFTS params (defaults from original bfts_config.yaml)
        self.num_drafts: int = 0  # Derived from len(ideas) in _generate_ideas()
        self.debug_prob: float = 0.5
        self.max_debug_depth: int = 3
        self.max_stderr_output: int = 1500

        # Idea generation state
        self.ideas: List[Dict[str, Any]] = []
        self._draft_count: int = 0

    def initialize(self) -> None:
        if self.client is not None:
            return
        self.client, _ = create_client(self.config.model, self.config.provider)
        p = self.config.agent_params
        # num_drafts is derived from len(ideas) after idea generation
        self.debug_prob = float(p.get("debug_prob", 0.5))
        self.max_debug_depth = int(p.get("max_debug_depth", 3))
        self.max_stderr_output = int(p.get("max_stderr_output", 1500))
        self.step_budget = int(p.get("max_steps", p.get("max_iter", 50)))
        self.num_parallel = int(p.get("num_parallel", 4))
        self._stage_budgets_ratios = p.get("stage_budgets", [0.10, 0.20, 0.50, 0.20])

    def run(
        self, task_description=None, target_files=None,
        baseline_results=None,
    ) -> AgentResult:
        # Store inputs
        if isinstance(task_description, tuple):
            self.task_description, _ = task_description
        else:
            self.task_description = task_description
        self.target_files = target_files or []
        self.baseline_results = baseline_results or {}
        self.baseline_results_filtered = get_filtered_results_for_prompt(
            self.baseline_results,
            self.config.runtime_params.get("metrics", {}),
        )

        benchmark_config = self.config.runtime_params.get("benchmark_config", {})
        self.direction = benchmark_config.get("metric_direction", "higher")
        self.metric_direction = self.direction
        self.metric_name = benchmark_config.get("metric", "")
        metrics_cfg = self.config.runtime_params.get("metrics", {})
        include_datasets = metrics_cfg.get("include_datasets")
        self.baseline_primary_metric = extract_primary_metric(
            self.baseline_results, self.metric_name, include_datasets,
        )
        agent_name = self.config.runtime_params.get("agent_name", "ai_scientist_v2")
        benchmark_name = self.config.runtime_params.get("benchmark_name", "benchmark")

        ts = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._output_dir = self.config.runtime_params.get("output_dir", "benchmark_results")
        parent_workspace = osp.join(self._output_dir, agent_name, benchmark_name, ts)
        os.makedirs(parent_workspace, exist_ok=True)
        self._parent_workspace = parent_workspace

        timeout = self.config.agent_params.get("execute_timeout", 2400)
        self.executor = BenchmarkExecutor(
            benchmark_config, agent_name, benchmark_name,
            f"{ts}_bfts", parent_timestamp=ts, timeout=timeout,
            output_dir=self._output_dir,
        )
        workspace = self.executor.setup_workspace()
        print(f"AI-Scientist-v2 workspace: {workspace}")

        self.editor = CodeEditor(
            model=self.config.model, provider=self.config.provider,
            target_files=self.target_files,
            task_description=self.task_description or "",
            log_dir=workspace,
            metric_name=self.metric_name,
            metric_direction=self.direction,
        )

        # Initialize
        self.initialize()
        self.journal = Journal(metric_direction=self.direction)
        self.baseline_snapshot = self._snapshot_target_files()
        self.all_steps = []

        # Idea generation (reuse TASv1's generate_ideas)
        self._generate_ideas(parent_workspace)

        # BFTS main loop
        try:
            self._bfts_loop()
        except Exception as e:
            print(f"[AI-Scientist-v2] BFTS loop error: {e}")
            import traceback; traceback.print_exc()

        # Final test
        test_result = self._run_final_test()

        # Build result
        best_step = None
        valid_steps = [s for s in self.all_steps if s.primary_metric is not None]
        if valid_steps:
            best_step = (max if self.direction == "higher" else min)(
                valid_steps, key=lambda s: s.primary_metric
            )

        # total_ideas counts draft nodes across all stages (from all_steps)
        total_ideas = sum(1 for s in self.all_steps if s.action == "draft")

        return AgentResult(
            all_steps=self.all_steps, best_step=best_step,
            test_result=test_result, total_steps=self.step_count,
            total_ideas=total_ideas,
            token_usage=self.get_token_usage_summary(),
            parent_workspace=parent_workspace,
        )

    # ------------------------------------------------------------------
    # Idea generation (reuses TASv1's generate_ideas)
    # ------------------------------------------------------------------

    def _generate_ideas(self, parent_workspace: str) -> None:
        """Generate research ideas and set num_drafts = len(ideas).

        If idea generation fails (missing seed_ideas.json, LLM errors, etc.),
        falls back to pure BFTS without idea constraints.
        """
        p = self.config.agent_params
        num_ideas_cfg = int(p.get("num_ideas", 3))

        base_dir = self.config.runtime_params.get("base_dir", ".")
        seed_file = osp.join(base_dir, "theaiscientist", "seed_ideas.json")

        if not osp.isfile(seed_file):
            print(f"[AI-Scientist-v2] seed_ideas.json not found at {seed_file}, "
                  f"falling back to pure BFTS without idea constraints "
                  f"(num_drafts={num_ideas_cfg})")
            self.ideas = []
            self.num_drafts = num_ideas_cfg
            self._draft_count = 0
            return

        try:
            from agents.theaiscientist.generate_ideas import (
                generate_ideas, set_token_usage_callback,
            )
            set_token_usage_callback(self._log_idea_token_usage)

            # Filter to files that exist (some target files like split_config.json
            # are created by val_command and don't exist yet at this point)
            readable_files = [f for f in self.target_files if osp.isfile(f)]

            self.ideas = generate_ideas(
                base_dir=base_dir,
                client=self.client,
                model=self.config.model,
                target_code_files=readable_files,
                num_generations=num_ideas_cfg,
                num_reflections=int(p.get("num_reflections", 3)),
            )
        except Exception as e:
            print(f"[AI-Scientist-v2] Idea generation failed: {e}")
            print(f"Falling back to pure BFTS without idea constraints "
                  f"(num_drafts={num_ideas_cfg})")
            self.ideas = []

        self.num_drafts = len(self.ideas) if self.ideas else num_ideas_cfg
        self._draft_count = 0
        print(f"Ideas: {len(self.ideas)} generated, num_drafts={self.num_drafts}")

        # Save ideas to workspace (even if empty)
        if self.ideas:
            ideas_file = osp.join(parent_workspace, "generated_ideas.json")
            with open(ideas_file, "w") as f:
                json.dump(self.ideas, f, indent=2)
            print(f"Saved ideas to: {ideas_file}")

    def _log_idea_token_usage(self, usage_info, step_name):
        """Token usage callback for idea generation. Not counted as step."""
        self.token_usage_log.append(usage_info)

    # ------------------------------------------------------------------
    # BFTS main loop with 4-stage pipeline + batch selection
    # ------------------------------------------------------------------

    def _bfts_loop(self) -> None:
        """BFTS with 4-stage pipeline, per-stage Journal isolation,
        simulated batch selection, and dynamic sub-stages.

        Faithfully reproduces:
          - agent_manager.py: 4-stage pipeline with independent Journals
          - parallel_agent.py: batch node selection with tree-level dedup
          - agent_manager.py:552-637: dynamic sub-stage goal generation
        """
        stage_budgets = self._compute_stage_budgets()
        best_prev_snapshot = None  # best code from previous stage
        best_prev_node = None      # best TreeNode from previous stage

        for stage_idx, stage_def in enumerate(V2_STAGES):
            stage_budget = stage_budgets[stage_idx]
            if stage_budget <= 0:
                continue
            stage_step_start = self.step_count

            # 1) Fresh Journal per stage ("intentional forgetting")
            self.journal = Journal(metric_direction=self.metric_direction)
            self._current_stage = stage_def
            self._current_goals = stage_def["goals"]
            self._substage_num = 1
            self._substage_step_start = self.step_count

            # 2) Restore previous stage's best code as starting point
            if best_prev_snapshot is not None:
                self._restore_snapshot(best_prev_snapshot)

            # 3) Stage-specific behavior
            #    Stage 1/3: normal BFTS with batch selection
            #    Stage 2/4: force improve from previous best (original behavior)
            stage_num_drafts = self.num_drafts if stage_idx == 0 else 1
            force_improve = stage_idx in (1, 3) and best_prev_node is not None

            mode_str = "force-improve" if force_improve else f"BFTS(drafts={stage_num_drafts})"
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx+1}/4: {stage_def['name']} "
                  f"(budget={stage_budget}, mode={mode_str})")
            print(f"{'='*60}")

            # 4) Run stage
            if force_improve:
                # Stage 2/4: all steps are forced improve from previous best
                # (original parallel_agent.py:2009-2015)
                while (self.step_count - stage_step_start < stage_budget
                       and self.budget_remaining()):
                    # Sub-stage check
                    if self.step_count > self._substage_step_start:
                        completed, reason = self._check_substage_completion()
                        if completed:
                            self._transition_substage(stage_def)
                    self._improve(best_prev_node)
            else:
                # Stage 1/3: normal BFTS batch selection
                while (self.step_count - stage_step_start < stage_budget
                       and self.budget_remaining()):
                    # Sub-stage check
                    if self.step_count > self._substage_step_start:
                        completed, reason = self._check_substage_completion()
                        if completed:
                            self._transition_substage(stage_def)

                    # Batch selection + execution
                    remaining_in_stage = stage_budget - (self.step_count - stage_step_start)
                    remaining_global = self.step_budget - self.step_count
                    batch_size = min(
                        self.num_parallel, remaining_in_stage, remaining_global
                    )
                    if batch_size <= 0:
                        break
                    batch = self._select_batch(batch_size, stage_num_drafts)
                    for action, target in batch:
                        if self.step_count - stage_step_start >= stage_budget:
                            break
                        if not self.budget_remaining():
                            break
                        if action == "draft":
                            self._draft()
                        elif action == "debug":
                            self._debug(target)
                        else:
                            self._improve(target)

            # 5) Save stage's best for next stage
            stage_best = self.journal.get_best_node()
            if stage_best is not None and stage_best.code_snapshot:
                best_prev_snapshot = stage_best.code_snapshot
                best_prev_node = stage_best
                # Also update BaseAgent's best tracking for _execute_test()
                if self._should_update_best(
                    stage_best.primary_metric, self.best_metric
                ):
                    self.best_metric = stage_best.primary_metric
                    self.best_code_snapshot = dict(stage_best.code_snapshot)

            steps_used = self.step_count - stage_step_start
            best_m = stage_best.primary_metric if stage_best else None
            print(f"Stage {stage_idx+1} ({stage_def['name']}): "
                  f"{steps_used} steps, best={best_m}")

    # ------------------------------------------------------------------
    # Stage budget computation
    # ------------------------------------------------------------------

    def _compute_stage_budgets(self) -> list:
        """Allocate step budget across 4 stages based on configured ratios."""
        ratios = self._stage_budgets_ratios
        budgets = [int(self.step_budget * r) for r in ratios]
        budgets[-1] += self.step_budget - sum(budgets)  # remainder to last
        return budgets

    # ------------------------------------------------------------------
    # Dynamic sub-stage management
    # ------------------------------------------------------------------

    def _check_substage_completion(self) -> tuple:
        """Check if current sub-stage should transition.
        Simplified from original VLM-based check: uses metric convergence."""
        # Max iterations per sub-stage: half the stage budget
        substage_max = max(3, len(self.journal.nodes) // 2)
        steps_in_substage = self.step_count - self._substage_step_start
        if steps_in_substage >= substage_max:
            return True, "substage max iterations reached"
        # Convergence: 3 consecutive good nodes with same metric
        good = self.journal.good_nodes
        if len(good) >= 3:
            recent = good[-3:]
            if all(n.primary_metric == recent[0].primary_metric for n in recent):
                return True, "metric converged"
        return False, "still improving"

    def _transition_substage(self, stage_def: dict) -> None:
        """Generate new sub-stage goals via LLM and update guidance."""
        good_count = len(self.journal.good_nodes)
        buggy_count = len(self.journal.buggy_nodes)
        best = self.journal.get_best_node()
        best_metric = best.primary_metric if best else "N/A"

        prompt = (
            "Based on experimental progress, generate focused goals for the "
            "next sub-stage of research.\n\n"
            f"Main Stage Goals:\n{stage_def['goals']}\n\n"
            f"Current Progress:\n"
            f"- Successful experiments: {good_count}\n"
            f"- Failed experiments: {buggy_count}\n"
            f"- Best {self.metric_name}: {best_metric} "
            f"({self.metric_direction} is better)\n\n"
            "Generate specific, actionable goals (3-5 sentences) that:\n"
            "1. Address current issues and limitations\n"
            "2. Build on recent progress\n"
            "3. Move towards the main stage goals"
        )
        try:
            text, _, usage = get_response_from_llm(
                prompt, client=self.client, model=self.config.model,
                system_message="You are an AI research advisor.",
            )
            if usage:
                self.token_usage_log.append(usage)
            sub_goals = text.strip()[:500]
        except Exception as e:
            sub_goals = "Continue improving based on current progress."
            print(f"[v2] Sub-stage goal generation failed: {e}")

        self._substage_num += 1
        self._substage_step_start = self.step_count
        self._current_goals = (
            f"{stage_def['goals']}\n\n"
            f"Sub-stage {self._substage_num} focus:\n{sub_goals}"
        )
        print(f"  Sub-stage {self._substage_num}: {sub_goals[:100]}...")

    # ------------------------------------------------------------------
    # Batch selection (simulates original _select_parallel_nodes)
    # ------------------------------------------------------------------

    def _select_batch(self, batch_size: int, num_drafts_for_stage: int
                      ) -> list:
        """Select batch_size nodes based on current Journal state.

        Simulates original parallel_agent.py:_select_parallel_nodes() with
        tree-level deduplication via processed_trees set.
        """
        batch = []
        processed_trees: set = set()
        pending_drafts = 0

        while len(batch) < batch_size:
            # Phase 1: initial drafting
            total_drafts = len(self.journal.draft_nodes) + pending_drafts
            if total_drafts < num_drafts_for_stage:
                batch.append(("draft", None))
                pending_drafts += 1
                continue

            # Viable trees (at least one non-buggy leaf)
            viable_trees = [
                root for root in self.journal.draft_nodes
                if not all(
                    leaf.is_buggy for leaf in self._get_leaves(root)
                )
            ]

            # Phase 2: probabilistic debugging (with tree dedup)
            if random.random() < self.debug_prob:
                debuggable = [
                    n for n in self.journal.buggy_nodes
                    if n.is_leaf
                    and n.debug_depth <= self.max_debug_depth
                    and self._get_tree_root_id(n) not in processed_trees
                ]
                if debuggable:
                    node = random.choice(debuggable)
                    processed_trees.add(self._get_tree_root_id(node))
                    batch.append(("debug", node))
                    continue

            # Phase 3: improve best good node (with tree dedup)
            good = self.journal.good_nodes
            if not good:
                batch.append(("draft", None))
                pending_drafts += 1
                continue

            sorted_good = sorted(
                good,
                key=lambda n: (
                    n.primary_metric
                    if n.primary_metric is not None
                    else float("-inf")
                ),
                reverse=(self.metric_direction == "higher"),
            )
            selected = False
            for node in sorted_good:
                tree_id = self._get_tree_root_id(node)
                if (tree_id not in processed_trees
                        or len(processed_trees) >= len(viable_trees)):
                    processed_trees.add(tree_id)
                    batch.append(("improve", node))
                    selected = True
                    break

            if not selected:
                batch.append(("draft", None))
                pending_drafts += 1

        return batch

    # ------------------------------------------------------------------
    # Tree traversal helpers
    # ------------------------------------------------------------------

    def _get_tree_root_id(self, node: TreeNode) -> str:
        """Trace to tree root, return its node_id."""
        current = node
        while current.parent is not None:
            current = current.parent
        return current.node_id

    def _get_leaves(self, root: TreeNode) -> list:
        """Get all leaf nodes of the subtree rooted at root."""
        leaves = []
        stack = [root]
        while stack:
            n = stack.pop()
            if not n.children:
                leaves.append(n)
            else:
                stack.extend(n.children)
        return leaves

    # ------------------------------------------------------------------
    # Draft operation (from original parallel_agent.py:453-492)
    # ------------------------------------------------------------------

    def _draft(self) -> None:
        _step_t0 = time.monotonic()
        self._restore_snapshot(self.baseline_snapshot)

        # Bind next idea (if available)
        idea = None
        if self._draft_count < len(self.ideas):
            idea = self.ideas[self._draft_count]
            self._draft_count += 1

        _token_start = len(self.token_usage_log)
        memory = self.journal.generate_summary(
            self.client, self.config.model, self.token_usage_log, self.metric_name
        )
        instruction = self._build_draft_instruction(memory, idea)
        edit_ok = self._edit_code(instruction)
        snap = self._snapshot_target_files()
        val = self._execute_val(run_id=self.step_count)

        node = TreeNode(
            plan=instruction[:500], action="draft",
            code_snapshot=snap, parent=None,
            val_result=val,
            primary_metric=val.get("primary_metric"),
            is_buggy=not val.get("success", False),
            error_context=self._truncate_error(val.get("error", "")),
            idea=idea,
        )
        self.journal.append(node)

        # Post-execution analysis (not a step)
        self._analyze_execution(node, val)

        idea_name = idea.get("Name", node.node_id) if idea else node.node_id
        snap_path = self._save_step_code_snapshot(self.step_count, self._parent_workspace, snapshot=snap)
        self.all_steps.append(StepResult(
            step_id=self.step_count,
            idea_id=f"draft_{idea_name}",
            idea_description=idea.get("Title", node.plan[:200]) if idea else node.plan[:200],
            action="draft", edit_success=edit_ok,
            val_result=val, primary_metric=val.get("primary_metric"),
            token_usage=self._collect_step_tokens(_token_start),
            step_duration_seconds=time.monotonic() - _step_t0,
            metadata={
                "code_snapshot_path": snap_path,
                "idea": idea,
                "instruction": self._last_edit_instruction,
                "analysis": node.analysis,
                "editor_log_path": getattr(self._last_edit_result, "log_path", None),
                "stage": getattr(self, '_current_stage', {}).get("name", ""),
                "substage": getattr(self, '_substage_num', 1),
            },
        ))

    # ------------------------------------------------------------------
    # Improve operation (from original parallel_agent.py:523-547)
    # ------------------------------------------------------------------

    def _improve(self, parent: TreeNode) -> None:
        _step_t0 = time.monotonic()
        self._restore_snapshot(parent.code_snapshot)

        idea = parent.root_idea  # Inherit from root of this tree
        _token_start = len(self.token_usage_log)
        memory = self.journal.generate_summary(
            self.client, self.config.model, self.token_usage_log, self.metric_name
        )
        instruction = self._build_improve_instruction(parent, memory, idea)
        edit_ok = self._edit_code(instruction)
        snap = self._snapshot_target_files()
        val = self._execute_val(run_id=self.step_count)

        node = TreeNode(
            plan=instruction[:500], action="improve",
            code_snapshot=snap, parent=parent,
            val_result=val,
            primary_metric=val.get("primary_metric"),
            is_buggy=not val.get("success", False),
            error_context=self._truncate_error(val.get("error", "")),
        )
        self.journal.append(node)
        self._analyze_execution(node, val)

        idea_name = idea.get("Name", node.node_id) if idea else node.node_id
        snap_path = self._save_step_code_snapshot(self.step_count, self._parent_workspace, snapshot=snap)
        self.all_steps.append(StepResult(
            step_id=self.step_count,
            idea_id=f"improve_{idea_name}",
            idea_description=idea.get("Title", node.plan[:200]) if idea else node.plan[:200],
            action="improve", edit_success=edit_ok,
            val_result=val, primary_metric=val.get("primary_metric"),
            token_usage=self._collect_step_tokens(_token_start),
            step_duration_seconds=time.monotonic() - _step_t0,
            metadata={
                "code_snapshot_path": snap_path,
                "idea": idea,
                "instruction": self._last_edit_instruction,
                "analysis": node.analysis,
                "editor_log_path": getattr(self._last_edit_result, "log_path", None),
                "stage": getattr(self, '_current_stage', {}).get("name", ""),
                "substage": getattr(self, '_substage_num', 1),
            },
        ))

    # ------------------------------------------------------------------
    # Debug operation (from original parallel_agent.py:494-521)
    # ------------------------------------------------------------------

    def _debug(self, parent: TreeNode) -> None:
        _step_t0 = time.monotonic()
        self._restore_snapshot(parent.code_snapshot)

        _token_start = len(self.token_usage_log)
        instruction = self._build_debug_instruction(parent)
        error_ctx = parent.error_context if parent.error_context else None
        edit_ok = self._edit_code(instruction, error_context=error_ctx)
        snap = self._snapshot_target_files()
        val = self._execute_val(run_id=self.step_count)

        node = TreeNode(
            plan=instruction[:500], action="debug",
            code_snapshot=snap, parent=parent,
            val_result=val,
            primary_metric=val.get("primary_metric"),
            is_buggy=not val.get("success", False),
            error_context=self._truncate_error(val.get("error", "")),
        )
        self.journal.append(node)
        self._analyze_execution(node, val)

        snap_path = self._save_step_code_snapshot(self.step_count, self._parent_workspace, snapshot=snap)
        self.all_steps.append(StepResult(
            step_id=self.step_count,
            idea_id=f"debug_{node.node_id}",
            idea_description=node.plan[:200],
            action="debug", edit_success=edit_ok,
            val_result=val, primary_metric=val.get("primary_metric"),
            token_usage=self._collect_step_tokens(_token_start),
            step_duration_seconds=time.monotonic() - _step_t0,
            metadata={
                "code_snapshot_path": snap_path,
                "instruction": self._last_edit_instruction,
                "analysis": node.analysis,
                "editor_log_path": getattr(self._last_edit_result, "log_path", None),
                "stage": getattr(self, '_current_stage', {}).get("name", ""),
                "substage": getattr(self, '_substage_num', 1),
            },
        ))

    # ------------------------------------------------------------------
    # Post-execution analysis (from original parse_exec_result)
    # ------------------------------------------------------------------

    def _analyze_execution(self, node: TreeNode, val_result: dict) -> None:
        """LLM analysis of execution results. NOT counted as a step."""
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
    # Prompt builders (faithful to original v2 prompts)
    # ------------------------------------------------------------------

    def _build_draft_instruction(self, memory: str, idea: Optional[Dict] = None) -> str:
        parts = [
            "## Introduction\n"
            "You are an AI researcher who is looking to publish a paper that will "
            "contribute significantly to the field. Your task is to implement a solid "
            "baseline based on your research idea provided below, from data preparation "
            "to model training, as well as evaluation. Focus on getting a simple but "
            "working implementation first, before any sophisticated improvements. "
            "We will explore more advanced variations in later stages.",
        ]

        if idea:
            parts.append(
                f"\n## Research Idea\n"
                f"Title: {idea.get('Title', 'Untitled')}\n"
                f"Experiment Plan: {idea.get('Experiment', '')}"
            )

        if memory and memory != "(No experiments completed yet.)":
            parts.append(f"\n## Memory\n{memory}")

        parts.append(
            f"\n## Baseline results\n"
            f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}"
        )

        parts.append(
            "\n## Experiment design sketch guideline\n"
            "- This first experiment design should be relatively simple, "
            "without extensive hyper-parameter optimization.\n"
            "- Take the Memory section into consideration when proposing the design.\n"
            "- The solution sketch should be 6-10 sentences.\n"
            "- Don't suggest to do EDA.\n"
            "- Make sure to create synthetic data if needed."
        )

        # Stage guidance (from 4-stage pipeline)
        if hasattr(self, '_current_goals') and self._current_goals:
            parts.append(f"\n## Stage Guidance\n{self._current_goals}")

        return "\n".join(parts)

    def _build_improve_instruction(self, parent: TreeNode, memory: str,
                                    idea: Optional[Dict] = None) -> str:
        parts = [
            "## Introduction\n"
            "You are an experienced AI researcher. You are provided with a previously "
            "developed implementation. Your task is to improve it based on the current "
            "experimental stage.",
        ]

        if idea:
            parts.append(
                f"\n## Research Idea\n"
                f"Title: {idea.get('Title', 'Untitled')}\n"
                f"Experiment Plan: {idea.get('Experiment', '')}"
            )

        if memory and memory != "(No experiments completed yet.)":
            parts.append(f"\n## Memory\n{memory}")

        # Parent results
        if parent.val_result:
            parts.append(
                f"\n## Previous results\n"
                f"{self._format_metric_line(parent.primary_metric, 'Previous result')}\n"
                f"Analysis: {parent.analysis}"
            )

        parts.append(
            f"\n## Baseline for comparison\n"
            f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}"
        )

        # Stage guidance
        if hasattr(self, '_current_goals') and self._current_goals:
            parts.append(f"\n## Stage Guidance\n{self._current_goals}")

        return "\n".join(parts)

    def _build_debug_instruction(self, parent: TreeNode) -> str:
        parts = [
            "## Introduction\n"
            "You are an experienced AI researcher. Your previous code for research "
            "experiment had a bug, so based on the information below, you should "
            "revise it in order to fix this bug.",
        ]

        # Research idea context from the root of this tree
        root_idea = parent.root_idea
        if root_idea:
            parts.append(
                f"\n## Research Idea\n"
                f"Title: {root_idea.get('Title', 'Untitled')}\n"
                f"Experiment Plan: {root_idea.get('Experiment', '')}"
            )

        if parent.analysis:
            parts.append(f"\n## Bug analysis\n{parent.analysis}")

        # Error context inline
        if parent.error_context:
            parts.append(
                f"\n## Error output\n```\n{parent.error_context}\n```"
            )

        # Debug depth awareness
        depth = parent.debug_depth
        if depth > 0:
            parts.append(
                f"\n## Debug depth\n"
                f"This is debug attempt #{depth}. "
                f"Previous debug attempts have not resolved the issue."
            )

        parts.append(
            "\n## Bugfix improvement sketch guideline\n"
            "- You should write a brief natural language description (3-5 sentences) "
            "of how the issue in the previous implementation can be fixed.\n"
            "- Don't suggest to do EDA."
        )

        # Stage guidance
        if hasattr(self, '_current_goals') and self._current_goals:
            parts.append(f"\n## Stage Guidance\n{self._current_goals}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_error(self, error: str) -> str:
        if not error:
            return ""
        if len(error) > self.max_stderr_output:
            return "..." + error[-self.max_stderr_output:]
        return error

    def _run_final_test(self):
        if self.best_code_snapshot is None:
            print("No best code snapshot found, skipping test.")
            return None
        try:
            benchmark_config = self.config.runtime_params.get("benchmark_config", {})
            agent_name = self.config.runtime_params.get("agent_name", "ai_scientist_v2")
            benchmark_name = self.config.runtime_params.get("benchmark_name", "benchmark")
            timeout = self.config.agent_params.get("execute_timeout", 2400)
            from datetime import datetime
            test_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.executor = BenchmarkExecutor(
                benchmark_config, agent_name, benchmark_name,
                f"{test_ts}_final_test", parent_timestamp=test_ts, timeout=timeout,
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
