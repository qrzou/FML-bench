"""AdaptiveSearch agent for FML-bench (regime-aware extension of autoresearch).

This is a STANDALONE agent. It is NOT a subclass of ``AutoresearchAgent`` —
the autoresearch greedy-hill-climb logic that powers Phase 1 has been COPIED
verbatim into this file so that AdaptiveSearch and autoresearch can evolve
independently. Sections of this file marked "[COPIED FROM AUTORESEARCH]" must
remain in lock-step with ``agents/autoresearch/agent.py`` for Phase 1 to be
behaviour-identical to vanilla autoresearch when the trigger never fires;
sections marked "[ADAPTIVESEARCH ADDITION]" are the regime-switch / multi-branch
logic specific to this agent.

Phase 1 (autoresearch verbatim) + one online stagnation check per step:
    slope_W=P1_W[k] = (improvement_curve[k] - improvement_curve[k-P1_W]) / P1_W
                  <= P1_EPS  triggers Phase 2 at step k >= P1_W + 1.

Phase 2 = autoresearch single-step semantics replicated per-branch, with
strict round-robin scheduling across N branches forked from the top-N kept
candidates of Phase 1. Two independent sub-rules adjust the LLM idea prompt:
    A: freq_W=20[k] <= 2  AND  reach_norm_max[k] <= 0.30   ("go deeper")
    B: freq_W=5[k]  <= 2  AND  effdim_cum[k] >= 1.25       ("stay coherent")
Both can fire => "consolidate AND deepen" prompt.

Final test code = global best across both phases (auto-tracked by
``BaseAgent._execute_val``). No external data files are loaded at runtime;
the per-task reach calibration is inlined as ``PER_TASK_MAX_REACH`` below.

References (locked in Steps 1-2 of the design):
    formal_results_analysis/analyze_adaptive_phase{1,2}.ipynb
    docs/v2_redesign/design/ADAPTIVESEARCH_DESIGN.md
"""
from __future__ import annotations

import json
import logging
import os
import os.path as osp
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.executor_factory import make_executor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt

from ..base import AgentConfig, AgentResult, BaseAgent, StepResult
from ..code_editor import CodeEditor
from ..llm import create_client, get_response_from_llm
from .embeddings import GraphCodeBERTEmbedder, participation_ratio
from .prompts import (
    BRANCH_INIT_TEMPLATE,
    SUBRULE_A_PREFIX,
    SUBRULE_AB_PREFIX,
    SUBRULE_B_PREFIX,
    JournalEntry,
    format_journal,
    format_sibling_block,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Per-task reach calibration constants — INLINED, no external file loads.
#
# Values are the column ``max_reach_18runs`` from the Step-2 calibration:
#     formal_results_analysis/adaptive_phase2/per_task_reach_norm.csv
#     produced by analyze_adaptive_phase2.ipynb (commit 5af5dc6).
#
# Status: PRIORS, not final. Multi-branch Phase-2 will mechanically raise
#         reach relative to single-branch history; the Step-3 4-task ablation
#         will refit if the deploy-time fire rates fall outside [0.20, 0.60].
# ===========================================================================
PER_TASK_MAX_REACH: Dict[str, float] = {
    # DENSE-OPP partition (phi_opp >= 0.1315)
    "Unlearning_open_unlearning":                  130.97540106999077,
    "Robustness_and_Reliability_art_default_hard": 228.13239743054416,
    "Privacy_privacymeter_corrected":              206.5509650892985,
    "Fairness_fairlearn":                          293.8351720083459,
    "Fairness_and_Bias_aif360_hard_postprocess":   369.8791343651161,
    "Federated_Learning_PFLlib":                   112.0861875548567,
    "Causality_gcastle":                           337.5422777609884,
    "Continual_Learning_continual_learning":       223.2756823948576,
    "Privacy_opacus":                               99.44490321155077,
    # SPARSE-OPP partition (phi_opp < 0.1315)
    "Generalization_domainbed":                    334.64680646396465,
    "Representation_Learning_solo_learn":           57.884839711096554,
    "Causality_causalml_hard":                     176.0618470058704,
    "Representation_Learning_lightly":             111.14783868086775,
    "Continual_Learning_pycil":                    265.0833114270035,
    "Data_Efficiency_usb":                         171.40061477849065,
    "Robustness_openood":                          433.04823524469464,
    "Generalization_domainbed_officehome":         183.49938985310192,
    "Data_Efficiency_easyfsl":                     281.1667135104085,
}


# ===========================================================================
# Per-task range-normalisation meta for the Phase-1 improvement curve.
#
# COPIED verbatim from formal_results_analysis/compute_auc_over_steps.py::
# RANGE_META (which produced the auc_curves.json that calibrated the
# slope[W=50, eps=0.0005] trigger). Format:
#     (direction_in_display_space, p_best, p_worst_or_"baseline")
# When p_worst is the literal string "baseline", denom uses the per-run
# transformed baseline value (collapsing the denominator to "headroom").
#
# Status: PRIORS for the trigger threshold. Step-3 4-task ablation will
# refit eps if deploy-time fire rates fall outside the calibration target.
# ===========================================================================
PER_TASK_RANGE_META: Dict[str, tuple] = {
    "Generalization_domainbed":                    ("higher", 1.0, 0.0),
    "Generalization_domainbed_officehome":         ("higher", 1.0, 0.0),
    "Data_Efficiency_easyfsl":                     ("higher", 1.0, 0.0),
    "Data_Efficiency_usb":                         ("higher", 1.0, 0.0),
    "Representation_Learning_lightly":             ("higher", 1.0, 0.0),
    "Representation_Learning_solo_learn":          ("higher", 1.0, 0.0),
    "Continual_Learning_continual_learning":       ("higher", 1.0, 0.0),
    "Continual_Learning_pycil":                    ("higher", 1.0, 0.0),
    "Causality_causalml_hard":                     ("lower",  0.0, "baseline"),
    "Causality_gcastle":                           ("lower",  0.0, "baseline"),
    "Robustness_and_Reliability_art_default_hard": ("higher", 1.0, 0.0),
    "Robustness_openood":                          ("higher", 1.0, 0.0),
    "Privacy_privacymeter_corrected":              ("lower",  0.0, 0.5),
    "Privacy_opacus":                              ("higher", 1.0, 0.0),
    "Fairness_and_Bias_aif360_hard_postprocess":   ("lower",  0.0, 1.0),
    "Fairness_fairlearn":                          ("lower",  0.0, 1.0),
    "Unlearning_open_unlearning":                  ("lower",  0.0, "baseline"),
    "Federated_Learning_PFLlib":                   ("higher", 1.0, 0.0),
}

UNLEARNING_TASK = "Unlearning_open_unlearning"


def _display_transform(value, task: str):
    """Match ``compute_auc_over_steps.display_transform`` byte-for-byte.

    Unlearning_open_unlearning: raw forget_quality (p-value) → -log10(p).
    All other tasks: identity.
    Returns None for non-positive Unlearning values or NaN.
    """
    if value is None:
        return None
    import math
    v = float(value)
    if math.isnan(v):
        return None
    if task == UNLEARNING_TASK:
        if v <= 0:
            return None
        return -math.log10(v)
    return v


# ===========================================================================
# [COPIED FROM AUTORESEARCH] In-memory experiment-log row.
# Mirrors agents/autoresearch/agent.py::ExperimentRecord. If the autoresearch
# definition is ever changed, this copy must be reviewed for consistency.
# ===========================================================================

@dataclass
class ExperimentRecord:
    """One row of the in-memory results log (replaces results.tsv).

    Original autoresearch columns: commit, val_bpb, memory_gb, status, description.
    Mapped to FML-bench: step_id, primary_metric, status, description.
    """

    step_id: int
    primary_metric: Optional[float]  # None for crashes
    status: str  # "keep" | "discard" | "crash"
    description: str
    error_context: Optional[str] = None


# ===========================================================================
# [ADAPTIVESEARCH ADDITION] Per-branch state for Phase 2.
# ===========================================================================

@dataclass
class Branch:
    branch_id: int
    parent_step_id: int                          # 0 if forked from baseline
    parent_metric: Optional[float]
    parent_idea: str
    snapshot: Dict[str, str]                     # branch incumbent code
    best_metric: Optional[float]                 # branch-local best
    history: List[ExperimentRecord] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    consecutive_crashes: int = 0
    pending_debug: bool = False
    last_idea: str = ""
    last_error: str = ""
    init_idea: Optional[str] = None              # first idea (for sibling diversity)


# ===========================================================================
# AdaptiveSearchAgent
# ===========================================================================

class AdaptiveSearchAgent(BaseAgent):
    """Greedy → frontier-expansion regime-switch agent (single switch).

    Phase 1 = autoresearch greedy hill-climb (logic copied verbatim from
    ``agents/autoresearch/agent.py``). Phase 2 = multi-branch frontier
    expansion + adaptive prompts triggered by an online stagnation check.
    """

    # ------------------------------------------------------------------
    # __init__: autoresearch state + AdaptiveSearch state
    # ------------------------------------------------------------------

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # [COPIED FROM AUTORESEARCH] ---------------------------------
        self.client = None
        self.max_debug_retries: int = 3
        self.max_stderr_output: int = 1500
        self.experiment_log: List[ExperimentRecord] = []      # P1 only
        self.current_snapshot: Dict[str, str] = {}            # P1 incumbent
        self.current_best_metric: Optional[float] = None
        self.consecutive_crashes: int = 0
        self.metric_direction: str = "higher"
        self.metric_name: str = ""
        self.task_description: str = ""
        self.baseline_results_filtered: Dict[str, Any] = {}
        self.all_steps: List[StepResult] = []
        self._pending_debug: bool = False
        self._last_idea: str = ""
        self._last_error: str = ""
        # -----------------------------------------------------------

        # [ADAPTIVESEARCH ADDITION] ---------------------------------
        self.phase: int = 1
        self.improvement_curve: List[float] = []   # higher-is-better, length == P1 step count

        p = config.agent_params

        # Phase-1 trigger constants
        self.P1_W = int(p.get("p1_window", 50))
        self.P1_EPS = float(p.get("p1_epsilon", 0.0005))

        # Phase-2 sub-rule constants
        self.A_freq_W = int(p.get("A_freq_window", 20))
        self.A_freq_max = int(p.get("A_freq_max", 2))
        self.A_reach_thr = float(p.get("A_reach_norm_max", 0.30))
        self.B_freq_W = int(p.get("B_freq_window", 5))
        self.B_freq_max = int(p.get("B_freq_max", 2))
        self.B_effdim_thr = float(p.get("B_effdim_min", 1.25))

        # Branch allocation cutoffs
        self.branch_skip_max = int(p.get("branch_skip_max", 3))
        self.branch_cutoff_1 = int(p.get("branch_cutoff_1", 15))
        self.branch_cutoff_2 = int(p.get("branch_cutoff_2", 30))

        # Embedding setup
        self.embed_model_id = str(p.get("graphcodebert_model", "microsoft/graphcodebert-base"))
        self.embed_max_len = int(p.get("embed_max_tokens", 512))
        self.embed_device = str(p.get("embed_device", "cpu"))

        # Lazy-allocated runtime state
        self.embedder: Optional[GraphCodeBERTEmbedder] = None
        self.g_baseline: Optional[np.ndarray] = None
        self.per_task_max_reach: Optional[float] = None
        # Range-normalisation constants for the improvement curve (matches
        # compute_auc_over_steps.compute_auc_for_run). Resolved in _pre_loop_setup.
        self._range_direction: Optional[str] = None
        self._imp_baseline: Optional[float] = None
        self._imp_denom: Optional[float] = None
        self._baseline_snapshot: Dict[str, str] = {}
        self.branches: List[Branch] = []
        self.journal: List[JournalEntry] = []
        self.next_branch_idx: int = 0
        self.phase2_started_at_step: Optional[int] = None
        self._trigger_fired: bool = False           # one-shot: suppress repeated trigger checks after the first fire
        # -----------------------------------------------------------

    # ==================================================================
    # [COPIED FROM AUTORESEARCH] Lifecycle: initialize / run
    # ==================================================================

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
        agent_name = self.config.runtime_params.get("agent_name", "adaptivesearch")
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
        eval_backend = self.config.runtime_params.get("eval_backend", "local")
        self.executor = make_executor(
            benchmark_config, agent_name, benchmark_name,
            f"{ts}_adaptivesearch", parent_timestamp=ts, timeout=timeout,
            output_dir=self._output_dir,
            eval_backend=eval_backend,
        )
        workspace = self.executor.setup_workspace()
        print(f"AdaptiveSearch workspace: {workspace}")

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

        # -- pre-loop setup (intentionally OUTSIDE the loop-body try/except,
        # but wrapped in its own try/finally so a failure cleans up the
        # executor before propagating). AdaptiveSearch-specific: load reach
        # calibration, GraphCodeBERT, baseline embedding. Must succeed before
        # any P1 step runs; failures propagate so the operator sees them,
        # rather than being swallowed by the loop-body except clause. --
        try:
            self._pre_loop_setup()
        except Exception:
            # Cleanup the executor we already created (line above) so
            # subprocess handles / .git chmod locks / temp dirs don't leak.
            if self.executor is not None:
                try:
                    self.executor.cleanup()
                except Exception as ce:
                    logger.error("Executor cleanup after _pre_loop_setup failure: %s", ce)
                self.executor = None
            raise

        # -- main loop --
        try:
            self._main_loop()
        except Exception as e:
            print(f"[AdaptiveSearch] Loop error: {e}")
            import traceback
            traceback.print_exc()

        # -- final test --
        test_result = self._run_final_test(
            benchmark_config, agent_name, benchmark_name, ts, timeout
        )

        # -- build result --
        best_step = self._find_best_step()
        n_keeps_p1 = sum(1 for r in self.experiment_log if r.status == "keep")
        n_keeps_p2 = sum(
            sum(1 for r in b.history if r.status == "keep") for b in self.branches
        )
        return AgentResult(
            all_steps=self.all_steps,
            best_step=best_step,
            test_result=test_result,
            total_steps=self.step_count,
            total_ideas=n_keeps_p1 + n_keeps_p2,
            token_usage=self.get_token_usage_summary(),
            parent_workspace=parent_workspace,
            metadata={
                "experiment_log": self._merged_experiment_log(),
                "adaptivesearch": self._adaptivesearch_metadata(),
            },
        )

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Main loop — phase-dispatch wrapper
    # ==================================================================

    def _pre_loop_setup(self) -> None:
        """Eager setup that must succeed before Phase 1 starts.

        Called from ``run()`` *before* the loop-body try/except so that any
        failure here (bad ``benchmark_name``, missing GraphCodeBERT cache, etc.)
        terminates the run cleanly rather than being silently swallowed.
        """
        self._baseline_snapshot = dict(self.current_snapshot)
        self._set_per_task_max_reach()
        self._set_improvement_curve_normalisation()
        self._ensure_embedder_loaded()
        self.g_baseline = self.embedder.embed_files(self._baseline_snapshot)

    def _set_improvement_curve_normalisation(self) -> None:
        """Resolve per-task range-normalisation constants for the
        improvement curve. Matches ``compute_auc_over_steps.compute_auc_for_run``
        so that ``slope[W=50, eps=0.0005]`` is on the same scale as calibration.
        """
        benchmark_name = self.config.runtime_params.get("benchmark_name", "")
        if benchmark_name not in PER_TASK_RANGE_META:
            raise ValueError(
                f"AdaptiveSearch: no improvement-curve range meta for "
                f"benchmark_name '{benchmark_name}'. Available tasks: "
                f"{sorted(PER_TASK_RANGE_META.keys())}"
            )
        direction, p_best, p_worst_spec = PER_TASK_RANGE_META[benchmark_name]
        self._range_direction = direction
        # Transform the baseline into display space (Unlearning: -log10).
        if self.baseline_primary_metric is None:
            raise RuntimeError(
                "AdaptiveSearch: baseline_primary_metric is None at "
                "_pre_loop_setup; cannot initialise improvement curve."
            )
        self._imp_baseline = _display_transform(
            self.baseline_primary_metric, benchmark_name
        )
        if self._imp_baseline is None:
            raise RuntimeError(
                f"AdaptiveSearch: display_transform of baseline "
                f"{self.baseline_primary_metric} for task {benchmark_name} "
                f"returned None; cannot compute imp_denom."
            )
        if p_worst_spec == "baseline":
            p_worst = self._imp_baseline
        else:
            p_worst = float(p_worst_spec)
        self._imp_denom = abs(float(p_best) - p_worst)
        if self._imp_denom == 0:
            raise RuntimeError(
                f"AdaptiveSearch: improvement-curve denom is 0 for task "
                f"{benchmark_name} (p_best={p_best}, p_worst={p_worst}); "
                f"baseline already at theoretical optimum?"
            )

    def _main_loop(self) -> None:
        # _pre_loop_setup() already populated _baseline_snapshot,
        # per_task_max_reach, embedder, g_baseline. Don't re-do them here.
        print(
            f"\n{'='*60}\n"
            f"AdaptiveSearch loop  |  budget={self.step_budget}  "
            f"P1=slope[W={self.P1_W},eps={self.P1_EPS}]  "
            f"A=(freq[W={self.A_freq_W}]<={self.A_freq_max}, reach_norm<={self.A_reach_thr})  "
            f"B=(freq[W={self.B_freq_W}]<={self.B_freq_max}, effdim>={self.B_effdim_thr})\n"
            f"{'='*60}\n"
        )

        while self.budget_remaining():
            if self.phase == 1:
                self._phase1_step()
                self._update_improvement_curve()
                # Only attempt the trigger check ONCE. After the first fire (whether
                # or not _setup_phase2 accepts), suppress further checks to avoid
                # log-spam and redundant re-sorting of experiment_log.
                if not self._trigger_fired and self._check_phase1_trigger():
                    self._trigger_fired = True
                    if self._setup_phase2():
                        self.phase = 2
                        self.phase2_started_at_step = self.step_count
            else:
                self._phase2_step()

    # ==================================================================
    # [COPIED FROM AUTORESEARCH] Phase-1 single-step body.
    # Wrapped in its own method so the outer phase dispatcher can interject.
    # The body is byte-for-byte identical to AutoresearchAgent._main_loop's
    # while-body, with one addition: metadata={"phase": 1, "branch_id": None}.
    # ==================================================================

    def _phase1_step(self) -> None:
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
                # ── adaptivesearch addition: phase / branch_id markers for
                #     downstream P1 vs P2 attribution; otherwise identical to
                #     autoresearch's StepResult metadata.
                "phase": 1,
                "branch_id": None,
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

    # ==================================================================
    # [COPIED FROM AUTORESEARCH] keep / discard / crash primitives (P1 only)
    # ==================================================================

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

    # ==================================================================
    # [COPIED FROM AUTORESEARCH] LLM idea generation + prompt builders
    # ==================================================================

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

    # ==================================================================
    # [COPIED FROM AUTORESEARCH] Final test + best-step lookup
    # ==================================================================

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
            eval_backend = self.config.runtime_params.get("eval_backend", "local")
            self.executor = make_executor(
                benchmark_config,
                agent_name,
                benchmark_name,
                f"{test_ts}_final_test",
                parent_timestamp=parent_ts,
                timeout=timeout,
                output_dir=self._output_dir,
                eval_backend=eval_backend,
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

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Phase-1 → Phase-2 trigger
    # ==================================================================

    def _update_improvement_curve(self) -> None:
        """Append the next entry of the improvement curve.

        Mirrors ``compute_auc_over_steps.compute_auc_for_run`` (lines 163-188):
        the curve stores **range-normalised improvement-from-baseline** in
        display space, monotone non-decreasing over P1 incumbent advances:

            "higher": imp = (best_so_far - baseline) / denom
            "lower":  imp = (baseline - best_so_far) / denom

        The slope[W=50, eps=0.0005] threshold was calibrated against this
        normalised curve, so this exact construction is required for the
        threshold to be meaningful.
        """
        benchmark_name = self.config.runtime_params.get("benchmark_name", "")
        raw_v = self.current_best_metric
        if raw_v is None:
            raw_v = self.baseline_primary_metric
        if raw_v is None:
            # Defensive fallback: both incumbent and baseline None — reuse the
            # previous curve value (or 0 when empty). Keeps slope well-defined.
            prev = self.improvement_curve[-1] if self.improvement_curve else 0.0
            self.improvement_curve.append(float(prev))
            return
        transformed_v = _display_transform(raw_v, benchmark_name)
        if transformed_v is None:
            # display_transform can return None on non-positive Unlearning vals.
            # Treat as "no information added"; reuse previous curve value.
            prev = self.improvement_curve[-1] if self.improvement_curve else 0.0
            self.improvement_curve.append(float(prev))
            return
        if self._range_direction == "higher":
            imp = (transformed_v - self._imp_baseline) / self._imp_denom
        else:
            imp = (self._imp_baseline - transformed_v) / self._imp_denom
        self.improvement_curve.append(float(imp))

    def _check_phase1_trigger(self) -> bool:
        k = self.step_count  # number of P1 val runs completed
        if k < self.P1_W + 1:
            return False
        curve = self.improvement_curve
        slope = (curve[k - 1] - curve[k - 1 - self.P1_W]) / self.P1_W
        triggered = slope <= self.P1_EPS
        if triggered:
            print(
                f"  [TRIGGER] step={k}  slope[W={self.P1_W}]={slope:.6f} "
                f"<= eps={self.P1_EPS}  -> entering Phase 2"
            )
        return triggered

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Phase-2 setup
    # ==================================================================

    def _setup_phase2(self) -> bool:
        remaining = self.step_budget - self.step_count
        if remaining <= self.branch_skip_max:
            print(
                f"  [TRIGGER] remaining={remaining} <= {self.branch_skip_max}; "
                f"too few steps left, staying in Phase 1."
            )
            return False
        if remaining <= self.branch_cutoff_1:
            n_branches = 1
        elif remaining <= self.branch_cutoff_2:
            n_branches = 2
        else:
            n_branches = 3

        keeps = sorted(
            [r for r in self.experiment_log if r.status == "keep"],
            key=lambda r: r.primary_metric,
            reverse=(self.metric_direction == "higher"),
        )

        self.branches = []
        for i in range(n_branches):
            if i < len(keeps):
                rec = keeps[i]
                snap = self._read_step_snapshot(rec.step_id)
                parent_idea = rec.description
                parent_metric = rec.primary_metric
                parent_step = rec.step_id
            else:
                snap = dict(self._baseline_snapshot)
                parent_idea = "(no kept candidate available — forking from baseline)"
                parent_metric = self.baseline_primary_metric
                parent_step = 0

            self.branches.append(Branch(
                branch_id=i,
                parent_step_id=parent_step,
                parent_metric=parent_metric,
                parent_idea=parent_idea,
                snapshot=snap,
                best_metric=parent_metric,
            ))

        # Embedder + g_baseline already loaded eagerly in _main_loop start.
        print(
            f"  [PHASE 2] starting with {n_branches} branch(es); "
            f"remaining_steps={remaining}; parents="
            + ", ".join(
                f"step {b.parent_step_id} ({self.metric_name}={b.parent_metric})"
                for b in self.branches
            )
        )
        return True

    def _ensure_embedder_loaded(self) -> None:
        if self.embedder is not None:
            return
        self.embedder = GraphCodeBERTEmbedder(
            model_id=self.embed_model_id,
            device=self.embed_device,
            max_len=self.embed_max_len,
        )

    def _set_per_task_max_reach(self) -> None:
        """Resolve per-task reach calibration from the inlined ``PER_TASK_MAX_REACH``
        dict. Fail loud if the active benchmark is not in the calibration set."""
        benchmark_name = self.config.runtime_params.get("benchmark_name", "")
        if benchmark_name not in PER_TASK_MAX_REACH:
            raise ValueError(
                f"AdaptiveSearch: no calibrated max_reach for benchmark_name "
                f"'{benchmark_name}'. Calibrated tasks: "
                f"{sorted(PER_TASK_MAX_REACH.keys())}"
            )
        self.per_task_max_reach = PER_TASK_MAX_REACH[benchmark_name]

    def _read_step_snapshot(self, step_id: int) -> Dict[str, str]:
        path = osp.join(self._parent_workspace, "step_snapshots",
                        f"step_{step_id:04d}_code.json")
        if not osp.exists(path):
            raise FileNotFoundError(
                f"AdaptiveSearch: cannot find step snapshot at {path}. "
                f"_save_step_code_snapshot may have failed for that step."
            )
        with open(path) as f:
            return json.load(f)

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Phase-2 single round-robin step
    # ==================================================================

    def _phase2_step(self) -> None:
        branch = self.branches[self.next_branch_idx]
        _token_start = len(self.token_usage_log)
        _step_t0 = time.monotonic()

        self._restore_snapshot(branch.snapshot)

        if branch.pending_debug:
            branch.pending_debug = False
            instruction = self._build_debug_instruction(
                branch.last_error, branch.last_idea
            )
            edit_ok = self._edit_code(instruction, error_context=branch.last_error)
            idea = f"[debug retry] {branch.last_idea}"
            action = "debug"
            A_fires, B_fires = False, False
            reach_norm_dbg = effdim_dbg = float("nan")
        elif not branch.history:
            idea = self._generate_branch_init_idea(branch)
            branch.last_idea = idea
            branch.init_idea = idea
            edit_ok = self._edit_code(idea)
            action = "branch_init"
            A_fires, B_fires = False, False
            reach_norm_dbg = effdim_dbg = float("nan")
        else:
            A_fires, B_fires, reach_norm_dbg, effdim_dbg = self._eval_subrules(branch)
            idea = self._generate_phase2_idea(branch, A_fires, B_fires,
                                              reach_norm_dbg, effdim_dbg)
            branch.last_idea = idea
            edit_ok = self._edit_code(idea)
            action = "improve"

        run_id = self.step_count
        val_result = self._execute_val(run_id)

        # Embed the post-edit submitted code (regardless of val success).
        # On failure: append a copy of g_baseline as a sentinel. This keeps
        # len(branch.embeddings) == len(branch.history) (so reach / effdim
        # windows align with freq windows), and contributes zero displacement
        # for that step (i.e. "this step did nothing").
        try:
            current_files = self._snapshot_target_files()
            g_k = self.embedder.embed_files(current_files)
        except Exception as e:
            logger.error("Phase-2 embedding failed at step %d: %s; using baseline as sentinel",
                         self.step_count, e)
            g_k = np.array(self.g_baseline, copy=True)
        branch.embeddings.append(g_k)

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
                "phase": 2,
                "branch_id": branch.branch_id,
                "subrule_A_fires": bool(A_fires),
                "subrule_B_fires": bool(B_fires),
                "reach_norm": reach_norm_dbg,
                "effdim": effdim_dbg,
                "code_snapshot_path": snap_path,
                "idea": idea,
                "instruction": self._last_edit_instruction,
                "editor_log_path": getattr(self._last_edit_result, "log_path", None),
            },
        )
        self.all_steps.append(step_result)

        if (
            not val_result.get("success")
            or val_result.get("primary_metric") is None
        ):
            self._handle_branch_crash(branch, val_result, idea)
            status = "crash"
            metric_for_journal = None
        else:
            metric = val_result["primary_metric"]
            if self._is_strict_branch_improvement(branch, metric):
                self._branch_keep(branch, metric, idea)
                status = "keep"
            else:
                self._branch_discard(branch, metric, idea)
                status = "discard"
            metric_for_journal = metric

        self.journal.append(JournalEntry(
            branch_id=branch.branch_id,
            step_id=self.step_count,
            status=status,
            primary_metric=metric_for_journal,
            idea=idea,
        ))

        self.next_branch_idx = (self.next_branch_idx + 1) % len(self.branches)

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Per-branch keep / discard / crash
    # ==================================================================

    def _is_strict_branch_improvement(self, branch: Branch, metric: float) -> bool:
        if branch.best_metric is None:
            return True
        if self.metric_direction == "lower":
            return metric < branch.best_metric
        return metric > branch.best_metric

    def _branch_keep(self, branch: Branch, metric: float, idea: str) -> None:
        branch.snapshot = self._snapshot_target_files()
        branch.best_metric = metric
        branch.consecutive_crashes = 0
        branch.history.append(ExperimentRecord(
            step_id=self.step_count,
            primary_metric=metric,
            status="keep",
            description=idea,
        ))
        print(
            f"  KEEP    | step={self.step_count} branch={branch.branch_id} | "
            f"{self.metric_name}={metric}"
        )

    def _branch_discard(self, branch: Branch, metric: float, idea: str) -> None:
        self._restore_snapshot(branch.snapshot)
        branch.consecutive_crashes = 0
        branch.history.append(ExperimentRecord(
            step_id=self.step_count,
            primary_metric=metric,
            status="discard",
            description=idea,
        ))
        print(
            f"  DISCARD | step={self.step_count} branch={branch.branch_id} | "
            f"{self.metric_name}={metric}"
        )

    def _handle_branch_crash(self, branch: Branch, val_result: dict, idea: str) -> None:
        error_msg = val_result.get("error", "Unknown error")
        truncated = self._truncate_error(error_msg)
        branch.history.append(ExperimentRecord(
            step_id=self.step_count,
            primary_metric=None,
            status="crash",
            description=idea,
            error_context=truncated[:300],
        ))
        branch.consecutive_crashes += 1
        print(
            f"  CRASH   | step={self.step_count} branch={branch.branch_id} | "
            f"consecutive={branch.consecutive_crashes}"
        )
        if branch.consecutive_crashes < self.max_debug_retries:
            branch.pending_debug = True
            branch.last_error = truncated
        else:
            self._restore_snapshot(branch.snapshot)
            branch.consecutive_crashes = 0
            branch.pending_debug = False
            print(f"  Giving up on this idea on branch {branch.branch_id}.")

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Sub-rule evaluation
    # ==================================================================

    def _eval_subrules(self, branch: Branch) -> Tuple[bool, bool, float, float]:
        # Short-history guard: with fewer history entries than the smaller freq
        # window (min(A_freq_W, B_freq_W) = 5 by default), the keep-count vs
        # raw-threshold comparison is statistically uninformative and can fire
        # spuriously on freshly-initialised branches. Skip eval until the
        # branch has accumulated enough samples for at least one freq window.
        min_steps = min(self.A_freq_W, self.B_freq_W)
        if len(branch.history) < min_steps:
            return False, False, float("nan"), float("nan")
        if not branch.embeddings:
            return False, False, float("nan"), float("nan")
        H = np.stack(branch.embeddings, axis=0)
        disp = H - self.g_baseline
        reach_cum = float(np.linalg.norm(disp, axis=1).max())
        reach_norm = (
            reach_cum / self.per_task_max_reach
            if self.per_task_max_reach
            else float("inf")
        )
        effdim = participation_ratio(disp)

        def freq_in_window(W: int) -> int:
            return sum(1 for r in branch.history[-W:] if r.status == "keep")

        A_fires = (freq_in_window(self.A_freq_W) <= self.A_freq_max) and (
            reach_norm <= self.A_reach_thr
        )
        B_fires = (freq_in_window(self.B_freq_W) <= self.B_freq_max) and (
            effdim >= self.B_effdim_thr
        )
        return A_fires, B_fires, reach_norm, effdim

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Phase-2 prompt construction
    # ==================================================================

    def _generate_branch_init_idea(self, branch: Branch) -> str:
        siblings = [
            (b.branch_id, b.init_idea)
            for b in self.branches
            if b.branch_id != branch.branch_id and b.init_idea is not None
        ]
        prompt = self._build_branch_init_prompt(branch, siblings)
        system_msg = (
            "You are an autonomous ML researcher running experiments to "
            "improve an ML algorithm. You propose concise, specific "
            "experimental ideas. When asked for mechanistic diversity, ensure "
            "your idea is qualitatively distinct from any sibling reference."
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
            logger.error("Branch-init idea generation failed: %s", e)
            return "Try a small hyperparameter change to improve the metric."

    def _generate_phase2_idea(
        self,
        branch: Branch,
        A_fires: bool,
        B_fires: bool,
        reach_norm: float,
        effdim: float,
    ) -> str:
        prompt = self._build_phase2_idea_prompt(
            branch, A_fires, B_fires, reach_norm, effdim
        )
        system_msg = (
            "You are an autonomous ML researcher running experiments to "
            "improve an ML algorithm. You propose concise, specific "
            "experimental ideas. Be creative but practical. Follow any "
            "[STRATEGY HINT] block at the top of the user message."
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
            logger.error("Phase-2 idea generation failed: %s", e)
            return "Try a small hyperparameter change to improve the metric."

    def _build_branch_init_prompt(
        self, branch: Branch, sibling_ideas: List[tuple]
    ) -> str:
        sibling_block = format_sibling_block(sibling_ideas)
        journal_block = format_journal(self.journal, self.metric_name)
        header = BRANCH_INIT_TEMPLATE.format(
            branch_id=branch.branch_id,
            parent_step=branch.parent_step_id,
            metric_name=self.metric_name,
            parent_metric=branch.parent_metric,
            parent_idea=branch.parent_idea,
            sibling_block=sibling_block,
            journal_block=journal_block,
        )
        body = self._build_idea_prompt_for_branch(branch)
        return header + "\n" + body

    def _build_phase2_idea_prompt(
        self,
        branch: Branch,
        A_fires: bool,
        B_fires: bool,
        reach_norm: float,
        effdim: float,
    ) -> str:
        prefix = ""
        keeps_A = sum(1 for r in branch.history[-self.A_freq_W:] if r.status == "keep")
        keeps_B = sum(1 for r in branch.history[-self.B_freq_W:] if r.status == "keep")
        if A_fires and B_fires:
            prefix = SUBRULE_AB_PREFIX.format(
                reach_norm=reach_norm, effdim=effdim,
            )
        elif A_fires:
            prefix = SUBRULE_A_PREFIX.format(
                keeps_in_window=keeps_A,
                window=self.A_freq_W,
                fmax=self.A_freq_max,
                reach_norm=reach_norm,
                reach_thr=self.A_reach_thr,
            )
        elif B_fires:
            prefix = SUBRULE_B_PREFIX.format(
                keeps_in_window=keeps_B,
                window=self.B_freq_W,
                fmax=self.B_freq_max,
                effdim=effdim,
                effdim_thr=self.B_effdim_thr,
            )

        body = self._build_idea_prompt_for_branch(branch)
        if prefix:
            return prefix + "\n" + body
        return body

    def _build_idea_prompt_for_branch(self, branch: Branch) -> str:
        """Mirror of ``_build_idea_prompt`` but using branch-local history +
        a journal of cross-branch activity. Same overall structure to keep
        prompt format stable across branches and across phases.
        """
        parts = []
        parts.append(
            "You are an autonomous ML researcher. Your goal is to propose "
            "the next experimental idea to improve the ML algorithm.\n"
        )
        parts.append(f"## Task Description\n{self.task_description}")

        if self.baseline_primary_metric is not None:
            parts.append(
                f"\n## Baseline Results\n"
                f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}"
            )

        if branch.best_metric is not None:
            parts.append(
                f"\n## Current Best Metric (this branch)\n"
                f"{self._format_metric_line(branch.best_metric, 'Branch Best')}"
            )

        parts.append(
            f"\n## Branch History (this branch only)\n"
            f"{self._format_branch_history(branch)}"
        )

        parts.append(
            f"\n## Cross-branch Journal (recent activity from all Phase-2 branches)\n"
            f"{format_journal(self.journal, self.metric_name)}"
        )

        parts.append(f"\n## Current Code\n{self._format_current_code()}")

        parts.append(
            "\n## Instructions\n"
            "Propose ONE specific experimental idea to try next. Consider:\n"
            "- What has worked and what hasn't on this branch (see Branch History)\n"
            "- What the other branches have been trying (see Cross-branch Journal)\n"
            "- The current code state and architecture\n"
            "- Architecture changes, optimizer changes, hyperparameter tuning, "
            "training loop modifications are all fair game\n"
            "- Keep changes focused and atomic so we can evaluate the effect\n"
            "- All else being equal, simpler is better\n\n"
            "Respond with a concise description (3-5 sentences) of what to "
            "change and why. Be specific enough that the change can be "
            "implemented directly."
        )
        return "\n".join(parts)

    def _format_branch_history(self, branch: Branch) -> str:
        if not branch.history:
            return "(no experiments on this branch yet.)"
        lines = [f"step | {self.metric_name:>10s} | status  | description"]
        lines.append("-" * 65)
        for rec in branch.history:
            if rec.primary_metric is not None:
                metric_str = f"{rec.primary_metric:.6f}"
            else:
                metric_str = "   CRASH  "
            lines.append(
                f"{rec.step_id:4d} | {metric_str:>10s} | "
                f"{rec.status:<7s} | {rec.description[:120]}"
            )
        return "\n".join(lines)

    # ==================================================================
    # [ADAPTIVESEARCH ADDITION] Result post-processing
    # ==================================================================

    def _merged_experiment_log(self) -> List[dict]:
        out = []
        for r in self.experiment_log:
            out.append({
                "phase": 1,
                "branch_id": None,
                "step_id": r.step_id,
                "primary_metric": r.primary_metric,
                "status": r.status,
                "description": r.description[:200],
            })
        for branch in self.branches:
            for r in branch.history:
                out.append({
                    "phase": 2,
                    "branch_id": branch.branch_id,
                    "step_id": r.step_id,
                    "primary_metric": r.primary_metric,
                    "status": r.status,
                    "description": r.description[:200],
                })
        out.sort(key=lambda d: d["step_id"])
        return out

    def _adaptivesearch_metadata(self) -> dict:
        return {
            "phase2_started_at_step": self.phase2_started_at_step,
            "n_branches": len(self.branches),
            "branch_summary": [
                {
                    "branch_id": b.branch_id,
                    "parent_step_id": b.parent_step_id,
                    "parent_metric": b.parent_metric,
                    "best_metric": b.best_metric,
                    "n_steps": len(b.history),
                    "n_keeps": sum(1 for r in b.history if r.status == "keep"),
                    "n_crashes": sum(1 for r in b.history if r.status == "crash"),
                }
                for b in self.branches
            ],
            "trigger": {
                "p1_window": self.P1_W,
                "p1_epsilon": self.P1_EPS,
            },
            "subrules": {
                "A": {
                    "freq_window": self.A_freq_W,
                    "freq_max": self.A_freq_max,
                    "reach_norm_max": self.A_reach_thr,
                },
                "B": {
                    "freq_window": self.B_freq_W,
                    "freq_max": self.B_freq_max,
                    "effdim_min": self.B_effdim_thr,
                },
            },
            "per_task_max_reach": self.per_task_max_reach,
        }
