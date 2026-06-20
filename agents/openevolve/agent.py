"""
OpenEvolve agent for FML-bench.

Faithfully ported from: https://github.com/codelion/openevolve
(Open-source implementation of Google's AlphaEvolve, arXiv:2506.13131)

Core algorithm: MAP-Elites evolutionary optimization with island model.
  - Population of programs in multi-dimensional feature grid (complexity x diversity)
  - Multiple islands evolve independently with periodic migration
  - Three sampling strategies: exploration (random), exploitation (archive), weighted
  - Rich prompt with parent metrics, evolution history, top performers, inspirations
  - Diff-based code modification

Original key features (preserved):
  - MAP-Elites archive per island with feature binning
  - Island isolation with ring-topology migration
  - Three-strategy parent sampling (exploration/exploitation/weighted ratios)
  - Inspiration programs for diverse context
  - Global + per-island best tracking
  - Post-execution LLM analysis

Adapted for FML-bench:
  - Parallel execution -> sequential round-robin across islands
  - Custom eval -> BenchmarkExecutor.run_val()
  - LLM ensemble -> single model via CodeEditor + llm.py
  - Embedding-based novelty -> edit-distance diversity
  - Single file evolution -> multi-file code_snapshot
  - 5 islands / 50 migration interval -> 3 islands / 10 migration interval
  - Cascade evaluation -> single _execute_val()
  - Checkpoint/resume to disk -> in-memory only

Reference: upstream openevolve —
  - openevolve/database.py (MAP-Elites, sampling, migration)
  - openevolve/prompt/sampler.py (prompt construction)
  - openevolve/process_parallel.py (iteration loop)
"""

import collections
import difflib
import json
import logging
import os
import os.path as osp
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmark.executor_factory import make_executor
from benchmark.utils import extract_primary_metric, get_filtered_results_for_prompt

from ..base import AgentConfig, AgentResult, BaseAgent, StepResult
from ..code_editor import CodeEditor
from ..llm import create_client, get_response_from_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures (ported from database.py)
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """A single program in the evolutionary population.

    Ported from OpenEvolve database.py Program dataclass.
    code_snapshot is Dict[str, str] (multi-file) instead of single 'code' string.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    code_snapshot: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    generation: int = 0
    island: int = 0
    step_id: Optional[int] = None
    action: str = ""  # "seed" | "evolve" | "debug"
    primary_metric: Optional[float] = None
    is_buggy: bool = True
    error_context: str = ""
    analysis: str = ""
    changes_description: str = ""
    complexity: float = 0.0
    diversity: float = 0.0
    migrated: bool = False  # True if migrated from another island
    debug_depth: int = 0  # consecutive debug ancestors count
    val_result: Optional[Dict[str, Any]] = None  # Full val result for logging
    edit_success: bool = True  # Whether CodeEditor edit succeeded


# ---------------------------------------------------------------------------
# ProgramDatabase: simplified MAP-Elites + island model
# (ported from database.py lines 112-1900+)
# ---------------------------------------------------------------------------

_DIVERSITY_CODE_MAX_LEN = 23000

class ProgramDatabase:
    """MAP-Elites population with island model.

    Simplified from OpenEvolve's ~3600-line database.py for sequential
    single-threaded execution in FML-bench.
    """

    def __init__(
        self,
        num_islands: int = 3,
        feature_bins: int = 5,
        archive_size: int = 30,
        exploration_ratio: float = 0.2,
        exploitation_ratio: float = 0.7,
        migration_interval: int = 10,
        migration_rate: float = 0.1,
        metric_direction: str = "higher",
        diversity_reference_size: int = 10,
        elite_selection_ratio: float = 0.1,
    ):
        self.num_islands = num_islands
        self.feature_bins = feature_bins
        self.archive_size = archive_size
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.metric_direction = metric_direction
        self.diversity_reference_size = diversity_reference_size
        # Inspiration sampling: fraction of n used as top-elite slots in step 2
        # of _sample_inspirations (matches original database.py:1601).
        self.elite_selection_ratio = elite_selection_ratio

        # Program storage
        self.programs: Dict[str, Program] = {}

        # Per-island structures
        self.islands: List[Set[str]] = [set() for _ in range(num_islands)]
        # MAP-Elites grid per island: feature_key -> program_id
        self.island_feature_maps: List[Dict[str, str]] = [
            {} for _ in range(num_islands)
        ]
        self.island_best_programs: List[Optional[str]] = [None] * num_islands
        self.island_generations: List[int] = [0] * num_islands

        # Global structures
        self.archive: Set[str] = set()
        self.best_program_id: Optional[str] = None
        self.last_migration_gen: int = 0

        # Feature scaling (running min/max for binning)
        self._complexity_min: float = float("inf")
        self._complexity_max: float = float("-inf")
        self._diversity_min: float = 0.0
        self._diversity_max: float = 1.0

        # Reference set for diversity calculation (greedy max-diversity selection)
        self._reference_programs: list = []

    # -- Core operations ---------------------------------------------------

    def add(self, program: Program, iteration: int = 0) -> None:
        """Add a program to the database. Handles MAP-Elites placement,
        archive update, and best-program tracking.

        Ported from database.py add() (lines 211-368).
        """
        self.programs[program.id] = program
        island_id = program.island

        # Add to island
        self.islands[island_id].add(program.id)

        # Update feature scaling
        if program.complexity < self._complexity_min:
            self._complexity_min = program.complexity
        if program.complexity > self._complexity_max:
            self._complexity_max = program.complexity

        # Calculate feature coordinates and place in MAP-Elites grid
        feature_key = self._calculate_feature_key(program)
        fmap = self.island_feature_maps[island_id]

        if feature_key not in fmap:
            # New cell — place directly
            fmap[feature_key] = program.id
        else:
            # Occupied cell — keep better program
            existing = self.programs.get(fmap[feature_key])
            if existing is None or self._is_better(program, existing):
                fmap[feature_key] = program.id

        # Update archive (top programs globally)
        self._update_archive(program)

        # Update best tracking
        self._update_best(program, island_id)

        # Update diversity reference set (greedy max-diversity selection)
        if not program.is_buggy:
            self._update_diversity_reference_set()

    def _update_diversity_reference_set(self) -> None:
        """Greedy max-diversity reference set selection.

        Ported from original database.py:2108-2146.  Selects a subset of
        programs that maximises pairwise diversity, used as the reference
        for ``compute_diversity()``.
        """
        all_codes = [
            self._concat_code(p.code_snapshot)[:_DIVERSITY_CODE_MAX_LEN]
            for p in self.programs.values()
            if not p.is_buggy and p.code_snapshot
        ]
        if not all_codes:
            return
        if len(all_codes) <= self.diversity_reference_size:
            self._reference_programs = list(all_codes)
            return

        # Greedy: start with random seed, add program maximising min-distance
        remaining = list(all_codes)
        selected = [remaining.pop(random.randint(0, len(remaining) - 1))]
        while len(selected) < self.diversity_reference_size and remaining:
            best_idx, best_min_div = -1, -1.0
            for i, candidate in enumerate(remaining):
                min_div = min(
                    1.0 - difflib.SequenceMatcher(None, candidate, sel).ratio()
                    for sel in selected
                )
                if min_div > best_min_div:
                    best_min_div = min_div
                    best_idx = i
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break
        self._reference_programs = selected

    def sample(self, island_id: int) -> Tuple[Program, List[Program]]:
        """Sample a parent + inspiration programs from an island.

        Three-strategy sampling (from database.py lines 1270-1288):
          - Exploration (20%): random from island
          - Exploitation (70%): from archive on this island
          - Weighted (10%): fitness-proportionate from island

        Returns: (parent, inspirations)
        """
        parent = self._sample_parent(island_id)
        # Use parent.island (parent's home) for inspirations — matches
        # original openevolve/database.py:1571 where _sample_inspirations
        # derives parent_island = parent.metadata.get("island", ...).
        # Critical for the cross-island archive fallback path in
        # _sample_parent: a parent borrowed from another island must
        # draw its inspirations from its own home island, preserving
        # island isolation semantics.
        inspirations = self._sample_inspirations(parent, parent.island)
        return parent, inspirations

    def get_top_programs(self, n: int, island_id: int) -> List[Program]:
        """Get top N programs from an island by metric.

        Ported from database.py get_top_programs() (lines 538-588).
        """
        island_programs = [
            self.programs[pid]
            for pid in self.islands[island_id]
            if pid in self.programs and not self.programs[pid].is_buggy
        ]
        if not island_programs:
            return []

        if self.metric_direction == "higher":
            island_programs.sort(
                key=lambda p: p.primary_metric if p.primary_metric is not None else float("-inf"),
                reverse=True,
            )
        else:
            island_programs.sort(
                key=lambda p: p.primary_metric if p.primary_metric is not None else float("inf"),
            )
        return island_programs[:n]

    def get_best_program(self) -> Optional[Program]:
        """Get the globally best program."""
        if self.best_program_id and self.best_program_id in self.programs:
            return self.programs[self.best_program_id]
        return None

    def migrate_programs(self) -> None:
        """Bidirectional ring-topology migration.

        Ported from database.py migrate_programs() (lines 1780-1877).
        Original uses bidirectional ring: each island donates to both neighbors.
        Migration count is per-island based on island size.
        Programs already migrated are skipped to prevent re-migration.
        """
        for src_island in range(self.num_islands):
            island_size = len(self.islands[src_island])
            n_migrate = max(1, int(island_size * self.migration_rate))

            # Bidirectional: donate to both neighbors
            dst_islands = [
                (src_island + 1) % self.num_islands,
                (src_island - 1) % self.num_islands,
            ]
            # Deduplicate (matters when num_islands == 2)
            dst_islands = list(set(dst_islands))

            top = self.get_top_programs(n_migrate, src_island)

            for prog in top:
                # Skip re-migration of already-migrated programs
                if prog.migrated:
                    continue
                for dst_island in dst_islands:
                    if dst_island == src_island:
                        continue
                    migrant = Program(
                        id=uuid.uuid4().hex[:12],
                        code_snapshot=dict(prog.code_snapshot),
                        parent_id=prog.id,
                        generation=prog.generation,
                        island=dst_island,
                        action="evolve",
                        primary_metric=prog.primary_metric,
                        is_buggy=prog.is_buggy,
                        analysis=prog.analysis,
                        changes_description=(
                            f"[migrated from island {src_island}] "
                            f"{prog.changes_description}"
                        ),
                        complexity=prog.complexity,
                        diversity=prog.diversity,
                        migrated=True,
                    )
                    self.add(migrant, iteration=0)

        self.last_migration_gen = max(self.island_generations)
        logger.info("Migration complete")

    # -- Internal helpers --------------------------------------------------

    def _sample_parent(self, island_id: int) -> Program:
        """Three-strategy sampling from an island.

        Ported from database.py _sample_parent() (lines 1270-1288) and the
        parallel path at sample_from_island() (lines 428-448).
        """
        island_pids = list(self.islands[island_id])
        if not island_pids:
            raise ValueError(f"Island {island_id} is empty, cannot sample.")

        r = random.random()

        # Exploration: random from island
        if r < self.exploration_ratio:
            pid = random.choice(island_pids)
            return self.programs[pid]

        # Exploitation: from archive, with multi-level fallback chain.
        # Ported faithfully from _sample_from_archive_for_island()
        # (database.py:1515-1552).
        if r < self.exploration_ratio + self.exploitation_ratio:
            # Clean up stale archive references
            valid_archive = [pid for pid in self.archive if pid in self.programs]

            if not valid_archive:
                # (a) Entire archive empty → fall back to weighted island
                return self._sample_from_island_weighted(island_id)

            archive_on_island = [
                pid for pid in valid_archive
                if self.programs[pid].island == island_id
            ]
            if archive_on_island:
                return self.programs[random.choice(archive_on_island)]

            # (b) Island has no archive programs → random from entire archive
            return self.programs[random.choice(valid_archive)]

        # Weighted: delegate to extracted helper (also used by fix #1 fallback).
        return self._sample_from_island_weighted(island_id)

    def _sample_from_island_weighted(self, island_id: int) -> Program:
        """Fitness-proportionate sampling from a specific island.

        Faithfully matches original database.py:1427-1482 for "higher is
        better" metrics. For "lower is better" (a FML-bench extension —
        the original only supports higher), we reflect the metric around
        the scored range midpoint so that:
          - The "lower is better" direction is symmetric to "higher is
            better" for the same metric values (same distribution shape,
            just with best/worst ordering flipped).
          - A single scored program always receives a positive fitness
            equal to its metric, so it dominates any buggy program.

        Weighting rules:
        - Empty island → global random fallback (original calls
          `_sample_random_parent()`; we use `random.choice` over all
          programs — equivalent).
        - Each program gets `max(fitness, EPSILON)` weight, matching the
          original's line 1463 `max(fitness, 0.001)` behavior.
        - Direction-aware raw fitness:
          - "higher is better" (default): fitness = primary_metric
          - "lower is better":           fitness = (max_s + min_s) - metric
            where max_s = max(scored_metrics), min_s = min(scored_metrics).
            This reflects the metric around the midpoint of the scored
            range — equivalent to using "higher is better" weighting on
            the metric values but with the ordering flipped.
          - Buggy / unmeasured programs: fitness = 0 → floored to EPSILON.

        Bug history (captured here so future maintainers understand why
        this is not simpler):
        1. Our first port used shift-based weighting (`metric - min + 1e-6`)
           which gave the worst scored program a weight of 1e-6, *lower*
           than buggy's EPSILON=1e-3, causing `buggy > worst_scored` in the
           sampling distribution. Fixed by switching to raw-fitness.
        2. The naive raw-fitness "lower" transform (`max_scored - metric`)
           has an edge case: when only 1 scored program exists, its fitness
           is `max - max = 0`, which then gets floored to EPSILON — tied
           with buggy. Fixed by using `(max + min) - metric` instead, so
           a single scored program receives fitness = metric > EPSILON.
        """
        island_pids = list(self.islands[island_id])
        if not island_pids:
            # Empty island → fall back to any program globally.
            # Matches original _sample_from_island_weighted line 1440-1443
            # which calls _sample_random_parent().
            if not self.programs:
                raise ValueError("Database has no programs")
            return random.choice(list(self.programs.values()))

        island_programs = [
            self.programs[pid] for pid in island_pids if pid in self.programs
        ]
        if not island_programs:
            return self.programs[random.choice(island_pids)]
        if len(island_programs) == 1:
            return island_programs[0]

        EPSILON = 1e-3  # Floor — matches original's max(fitness, 0.001)

        # Collect scored metrics to compute direction-aware fitness.
        scored_metrics = [
            p.primary_metric for p in island_programs
            if not p.is_buggy and p.primary_metric is not None
        ]
        if not scored_metrics:
            # All programs buggy/unmeasured → uniform random
            return random.choice(island_programs)

        # Direction-aware raw fitness per program.
        #   higher: fitness = primary_metric
        #   lower:  fitness = (max_scored + min_scored) - primary_metric
        # Buggy / unmeasured always get fitness = 0 → EPSILON after flooring.
        if self.metric_direction == "lower":
            max_scored = max(scored_metrics)
            min_scored = min(scored_metrics)
            reflection_sum = max_scored + min_scored
            def _fitness(p: Program) -> float:
                if p.is_buggy or p.primary_metric is None:
                    return 0.0
                return reflection_sum - p.primary_metric
        else:
            def _fitness(p: Program) -> float:
                if p.is_buggy or p.primary_metric is None:
                    return 0.0
                return p.primary_metric

        weights = [max(_fitness(p), EPSILON) for p in island_programs]

        # Normalize and sample (matches original lines 1466-1470).
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(island_programs)] * len(island_programs)
        return random.choices(island_programs, weights=weights, k=1)[0]

    def _sample_inspirations(
        self, parent: Program, island_id: int, n: int = 3
    ) -> List[Program]:
        """Sample inspiration programs (diverse approaches).

        Ported faithfully from database.py _sample_inspirations()
        (lines 1554-1676). Four-step structure:
          1. Island best (if != parent)
          2. get_top_programs(max(1, int(n * elite_selection_ratio)))
          3. Feature-neighborhood perturbation (±2 bins on both axes)
          4. Random fill from island to reach n

        Steps 3 and 4 are both nested inside the original's line-1608
        guard `len(island_programs) > n and len(inspirations) < n` — when
        the island is saturated (≤ n programs) both steps are skipped.
        """
        inspirations: List[Program] = []
        seen_ids = {parent.id}

        # Step 1: Island best (if != parent)
        best_pid = self.island_best_programs[island_id]
        if best_pid and best_pid != parent.id and best_pid in self.programs:
            inspirations.append(self.programs[best_pid])
            seen_ids.add(best_pid)

        # Step 2: Top elite by ratio (NOT n*2). With n=3 and default
        # elite_selection_ratio=0.1, top_n = max(1, int(0.3)) = 1. This
        # often duplicates the island-best so contributes zero after dedup.
        top_n = max(1, int(n * self.elite_selection_ratio))
        top = self.get_top_programs(top_n, island_id)
        for p in top:
            if p.id not in seen_ids:
                inspirations.append(p)
                seen_ids.add(p.id)

        # Steps 3+4 share the same guard (matches original database.py:1608):
        #   only fire when the island has strictly more than n programs AND
        #   we haven't already filled n slots. If the island has <= n
        #   programs we skip both (matches original behavior — there's
        #   nothing meaningful to perturb or fill against when saturated).
        island_pids = list(self.islands[island_id])
        if len(island_pids) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)
            parent_coords = self._calculate_feature_coords(parent)
            nearby: List[Program] = []

            # Build the island's feature-map: {feature_key: program_id}.
            # When multiple programs fall in the same cell, the last one
            # wins (same as original — dict assignment overwrites).
            island_feature_map: Dict[str, str] = {}
            for pid in island_pids:
                if pid in self.programs:
                    key = self._calculate_feature_key(self.programs[pid])
                    island_feature_map[key] = pid

            # Step 3: Feature-neighborhood perturbation.
            # With feature_bins=5 and ±2 perturbation on each axis, the
            # sampled cell can be ANY cell on the island (±2 covers the
            # full [0, feature_bins-1] range from any starting point).
            # Locality is coarse — matches original's behavior with
            # default feature_bins and is a known limitation of the
            # 5-bin setup. For larger feature_bins perturbation becomes
            # genuinely local.
            for _ in range(remaining_slots * 3):
                perturbed = (
                    max(0, min(self.feature_bins - 1,
                               parent_coords[0] + random.randint(-2, 2))),
                    max(0, min(self.feature_bins - 1,
                               parent_coords[1] + random.randint(-2, 2))),
                )
                key = f"{perturbed[0]}_{perturbed[1]}"
                cand_pid = island_feature_map.get(key)
                # Exclude parent + step-1/2 picks + already-nearby picks via
                # seen_ids (updated eagerly to allow step 4's single-set
                # exclusion check).
                if cand_pid and cand_pid not in seen_ids:
                    nearby.append(self.programs[cand_pid])
                    seen_ids.add(cand_pid)
                    if len(nearby) >= remaining_slots:
                        break

            # Step 4: Random fill from remaining island programs — still
            # inside the len(island_pids) > n guard, matching original
            # line 1646 where the fill is nested under the same condition.
            if len(inspirations) + len(nearby) < n:
                remaining = n - len(inspirations) - len(nearby)
                available = [
                    pid for pid in island_pids
                    if pid in self.programs and pid not in seen_ids
                ]
                if available:
                    fill_ids = random.sample(
                        available, min(remaining, len(available))
                    )
                    nearby.extend(self.programs[pid] for pid in fill_ids)

            inspirations.extend(nearby)

        return inspirations[:n]

    def _calculate_feature_coords(self, program: Program) -> Tuple[int, int]:
        """Return raw (c_bin, d_bin) integer coordinates for a program.

        Ported from database.py _calculate_feature_coords() (lines 834-900).
        Uses complexity (code length) and diversity (edit distance) as the
        two feature dimensions. Used by both _calculate_feature_key()
        (returns a string form) and _sample_inspirations() feature
        perturbation (needs raw ints).
        """
        # Bin complexity
        c_range = self._complexity_max - self._complexity_min
        if c_range > 0:
            c_norm = (program.complexity - self._complexity_min) / c_range
        else:
            c_norm = 0.5
        c_bin = min(int(c_norm * self.feature_bins), self.feature_bins - 1)

        # Bin diversity
        d_range = self._diversity_max - self._diversity_min
        if d_range > 0:
            d_norm = (program.diversity - self._diversity_min) / d_range
        else:
            d_norm = 0.5
        d_bin = min(int(d_norm * self.feature_bins), self.feature_bins - 1)

        return (c_bin, d_bin)

    def _calculate_feature_key(self, program: Program) -> str:
        """MAP-Elites feature key (string form, used as map key).

        Thin wrapper over _calculate_feature_coords. The string format
        `"c_bin_d_bin"` is used as the key in island_feature_maps.
        """
        c_bin, d_bin = self._calculate_feature_coords(program)
        return f"{c_bin}_{d_bin}"

    def _is_better(self, p1: Program, p2: Program) -> bool:
        """Compare two programs by primary metric."""
        if p1.primary_metric is None:
            return False
        if p2.primary_metric is None:
            return True
        if self.metric_direction == "higher":
            return p1.primary_metric > p2.primary_metric
        return p1.primary_metric < p2.primary_metric

    def _update_archive(self, program: Program) -> None:
        """Update global archive of elite programs.

        Ported from database.py archive management (lines 1131-1176).
        """
        if program.is_buggy:
            return

        self.archive.add(program.id)

        # Trim archive to max size
        if len(self.archive) > self.archive_size:
            archive_programs = [
                self.programs[pid]
                for pid in self.archive
                if pid in self.programs
            ]
            if self.metric_direction == "higher":
                archive_programs.sort(
                    key=lambda p: p.primary_metric if p.primary_metric is not None else float("-inf"),
                )
            else:
                archive_programs.sort(
                    key=lambda p: p.primary_metric if p.primary_metric is not None else float("inf"),
                    reverse=True,
                )
            # Remove worst
            while len(self.archive) > self.archive_size and archive_programs:
                worst = archive_programs.pop(0)
                self.archive.discard(worst.id)

    def _update_best(self, program: Program, island_id: int) -> None:
        """Update best program tracking (global + per-island).

        Ported from database.py best tracking (lines 1178-1217).
        """
        if program.is_buggy:
            return

        # Per-island best
        current_island_best = self.island_best_programs[island_id]
        if current_island_best is None or current_island_best not in self.programs:
            self.island_best_programs[island_id] = program.id
        elif self._is_better(program, self.programs[current_island_best]):
            self.island_best_programs[island_id] = program.id

        # Global best
        if self.best_program_id is None or self.best_program_id not in self.programs:
            self.best_program_id = program.id
        elif self._is_better(program, self.programs[self.best_program_id]):
            self.best_program_id = program.id

    @staticmethod
    def _concat_code(code_snapshot: Dict[str, str]) -> str:
        """Concatenate all files in a code snapshot into a single string."""
        return "\n".join(code_snapshot.get(k, "") for k in sorted(code_snapshot))


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def compute_complexity(code_snapshot: Dict[str, str]) -> float:
    """Complexity feature: total code length across all target files."""
    return float(sum(len(content) for content in code_snapshot.values()))


def compute_diversity(
    code_snapshot: Dict[str, str], reference_programs: List[str]
) -> float:
    """Diversity feature: average edit distance to reference set.

    Uses difflib.SequenceMatcher for fast approximation. Returns value in [0, 1].
    Ported from OpenEvolve database.py diversity_metric="edit_distance".
    """
    if not reference_programs:
        return 0.5  # neutral default

    code_str = ProgramDatabase._concat_code(code_snapshot)[:_DIVERSITY_CODE_MAX_LEN]
    total_dist = 0.0
    for ref in reference_programs:
        ratio = difflib.SequenceMatcher(None, code_str, ref).ratio()
        total_dist += 1.0 - ratio  # distance = 1 - similarity
    return total_dist / len(reference_programs)


# ---------------------------------------------------------------------------
# OpenEvolveAgent
# ---------------------------------------------------------------------------

class OpenEvolveAgent(BaseAgent):
    """
    OpenEvolve: MAP-Elites evolutionary agent for iterative ML improvement.

    Uses a population of programs in a feature grid (complexity x diversity),
    organized into islands with periodic migration. Parent selection uses
    three strategies (exploration / exploitation / weighted). Rich prompt
    context includes parent metrics, evolution history, top performers,
    and inspiration programs from the population.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        self.db: Optional[ProgramDatabase] = None
        self.all_steps: List[StepResult] = []

        # Config (set in initialize)
        self.num_islands: int = 3
        self.feature_bins: int = 5
        self.archive_size: int = 30
        self.exploration_ratio: float = 0.2
        self.exploitation_ratio: float = 0.7
        self.migration_interval: int = 10
        self.migration_rate: float = 0.1
        self.num_top_programs: int = 3
        self.num_inspirations: int = 3
        self.max_debug_depth: int = 2
        self.debug_prob: float = 0.3
        self.max_stderr_output: int = 1500
        self.diversity_reference_size: int = 10
        self.elite_selection_ratio: float = 0.1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if self.client is not None:
            return
        self.client, _ = create_client(self.config.model, self.config.provider)
        p = self.config.agent_params
        self.num_islands = int(p.get("num_islands", 3))
        self.feature_bins = int(p.get("feature_bins", 5))
        self.archive_size = int(p.get("archive_size", 30))
        self.exploration_ratio = float(p.get("exploration_ratio", 0.2))
        self.exploitation_ratio = float(p.get("exploitation_ratio", 0.7))
        self.migration_interval = int(p.get("migration_interval", 10))
        self.migration_rate = float(p.get("migration_rate", 0.1))
        self.num_top_programs = int(p.get("num_top_programs", 3))
        self.num_inspirations = int(p.get("num_inspirations", 3))
        self.max_debug_depth = int(p.get("max_debug_depth", 2))
        self.debug_prob = float(p.get("debug_prob", 0.3))
        self.max_stderr_output = int(p.get("max_stderr_output", 1500))
        self.diversity_reference_size = int(p.get("diversity_reference_size", 10))
        self.elite_selection_ratio = float(p.get("elite_selection_ratio", 0.1))
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
            baseline_results or {}, self.metric_name, include_datasets
        )
        agent_name = self.config.runtime_params.get("agent_name", "openevolve")
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
            f"{ts}_openevolve", parent_timestamp=ts, timeout=timeout,
            output_dir=self._output_dir,
            eval_backend=eval_backend,
        )
        workspace = self.executor.setup_workspace()
        print(f"OpenEvolve workspace: {workspace}")

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
        self.all_steps = []

        # Create database
        self.db = ProgramDatabase(
            num_islands=self.num_islands,
            feature_bins=self.feature_bins,
            archive_size=self.archive_size,
            exploration_ratio=self.exploration_ratio,
            exploitation_ratio=self.exploitation_ratio,
            migration_interval=self.migration_interval,
            migration_rate=self.migration_rate,
            metric_direction=self.metric_direction,
            diversity_reference_size=self.diversity_reference_size,
            elite_selection_ratio=self.elite_selection_ratio,
        )

        # -- seed population --
        baseline_snapshot = self._snapshot_target_files()
        self._seed_population(baseline_snapshot)

        # -- main evolution loop --
        try:
            self._evolution_loop()
        except Exception as e:
            print(f"[OpenEvolve] Evolution loop error: {e}")
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
            total_ideas=len(self.db.archive) if self.db else 0,
            token_usage=self.get_token_usage_summary(),
            parent_workspace=parent_workspace,
        )

    # ------------------------------------------------------------------
    # Seed population (evaluate baseline on each island)
    # ------------------------------------------------------------------

    def _seed_population(self, baseline_snapshot: Dict[str, str]) -> None:
        """Seed each island with the baseline program.

        Evaluates the baseline once via direct executor call (does NOT
        consume a step from the budget), then copies the result to all
        islands without additional evaluation.
        """
        print(
            f"\n{'='*60}\n"
            f"OpenEvolve  |  budget={self.step_budget}  islands={self.num_islands}  "
            f"bins={self.feature_bins}\n"
            f"{'='*60}\n"
        )

        # Evaluate baseline once (no step_count increment — seed is init, not a step)
        self._restore_snapshot(baseline_snapshot)
        val_result = self.executor.run_val(run_id=0)

        # Track best code manually (mirrors _execute_val logic)
        if val_result.get("success") and val_result.get("primary_metric") is not None:
            if self._should_update_best(val_result["primary_metric"], self.best_metric):
                self.best_metric = val_result["primary_metric"]
                self.best_code_snapshot = self._snapshot_target_files()

        baseline_metric = val_result.get("primary_metric")
        baseline_buggy = (
            not val_result.get("success") or baseline_metric is None
        )

        complexity = compute_complexity(baseline_snapshot)

        # Add seed to each island (same eval result, no extra steps)
        for island_id in range(self.num_islands):
            seed = Program(
                code_snapshot=dict(baseline_snapshot),
                island=island_id,
                generation=0,
                action="seed",
                primary_metric=baseline_metric,
                is_buggy=baseline_buggy,
                changes_description="baseline",
                complexity=complexity,
                diversity=0.5,
                step_id=0,
            )
            self.db.add(seed, iteration=0)

        print(f"  Seed metric: {baseline_metric} (buggy={baseline_buggy})")

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def _evolution_loop(self) -> None:
        """Main MAP-Elites evolution loop.

        Generation-based batching: sample parents for ALL islands at the
        start of each generation (frozen DB state), then execute
        sequentially.  This simulates the original's parallel workers
        where same-generation islands cannot see each other's results.
        """
        iteration = 0

        while self.budget_remaining():
            # === Generation start: freeze DB state, sample all islands ===
            generation_plan = []
            for island_id in range(self.db.num_islands):
                if not self.budget_remaining():
                    break
                try:
                    parent, inspirations = self.db.sample(island_id)
                except ValueError:
                    logger.warning("Island %d empty, skipping", island_id)
                    iteration += 1
                    continue
                # Top programs follow parent.island so that every piece of
                # evolutionary context (inspirations, top, history, prompt)
                # is consistent with where the parent actually lives — matches
                # original openevolve/database.py where all evolution context
                # is derived from parent.metadata["island"].
                top_programs = self.db.get_top_programs(
                    self.num_top_programs, parent.island
                )
                generation_plan.append({
                    # island_id is the SCHEDULING slot (used only for
                    # island_generations accounting in the migration trigger).
                    # The actual evolution work happens on parent.island.
                    "island_id": island_id,
                    "parent": parent,
                    "inspirations": inspirations,
                    "top_programs": top_programs,
                    "iteration": iteration,
                })
                iteration += 1

            # === Execute all steps in this generation sequentially ===
            for plan in generation_plan:
                if not self.budget_remaining():
                    break

                _token_start = len(self.token_usage_log)
                _step_t0 = time.monotonic()

                parent = plan["parent"]
                # Scheduling slot — still needed for island_generations bookkeeping
                # (migration cadence), but NOT passed to _debug / _evolve; those
                # use parent.island internally.
                island_id = plan["island_id"]

                if (
                    parent.is_buggy
                    and parent.debug_depth < self.max_debug_depth
                    and random.random() < self.debug_prob
                ):
                    child = self._debug(parent)
                else:
                    child = self._evolve(
                        parent, plan["inspirations"],
                        plan["top_programs"], plan["iteration"]
                    )
                child._token_start = _token_start
                child._step_duration = time.monotonic() - _step_t0

                self.all_steps.append(self._make_step_result(child))
                self.db.island_generations[island_id] += 1

            # === Generation end: check migration ===
            if self.db.island_generations:
                max_gen = max(self.db.island_generations)
                if (
                    max_gen - self.db.last_migration_gen
                    >= self.migration_interval
                ):
                    self.db.migrate_programs()

    # ------------------------------------------------------------------
    # Evolve operation (from process_parallel.py _run_iteration_worker)
    # ------------------------------------------------------------------

    def _evolve(
        self,
        parent: Program,
        inspirations: List[Program],
        top_programs: List[Program],
        iteration: int,
    ) -> Program:
        """Generate and evaluate a child program via evolution.

        The child is placed on ``parent.island`` (parent's home) to match
        original openevolve/database.py add() behavior, which inherits
        parent.metadata["island"] for child placement. When cross-island
        archive fallback has sourced this parent from a different island
        than the scheduling slot, the child returns to the parent's home.
        """
        # Restore parent's code
        self._restore_snapshot(parent.code_snapshot)

        # Build instruction with evolutionary context
        instruction = self._build_evolution_instruction(
            parent, inspirations, top_programs, iteration
        )

        # Edit via CodeEditor
        edit_success = self._edit_code(instruction)

        # Snapshot modified code
        child_snapshot = self._snapshot_target_files()

        # Evaluate (costs 1 step)
        val_result = self._execute_val(self.step_count)
        child_val_duration = self._last_val_duration

        # Create child program
        child = Program(
            code_snapshot=child_snapshot,
            parent_id=parent.id,
            generation=parent.generation + 1,
            island=parent.island,
            step_id=self.step_count,
            action="evolve",
            primary_metric=val_result.get("primary_metric"),
            is_buggy=(
                not val_result.get("success")
                or val_result.get("primary_metric") is None
            ),
            error_context=self._truncate_error(val_result.get("error", "")),
            complexity=compute_complexity(child_snapshot),
            diversity=compute_diversity(
                child_snapshot, list(self.db._reference_programs)
            ),
        )

        child.val_duration = child_val_duration  # store per-program for StepResult
        child.val_result = val_result
        child.edit_success = edit_success
        child._instruction = instruction
        child._editor_log_path = getattr(self._last_edit_result, "log_path", None)

        # LLM analysis (not a step)
        self._analyze_execution(child, val_result)

        # Infer changes description from analysis
        child.changes_description = child.analysis[:150] if child.analysis else ""

        # Add to database
        self.db.add(child, iteration=iteration)

        status = "OK" if not child.is_buggy else "BUGGY"
        print(
            f"  [{status}] island={parent.island} gen={child.generation} "
            f"metric={child.primary_metric} iter={iteration}"
        )

        return child

    # ------------------------------------------------------------------
    # Debug operation
    # ------------------------------------------------------------------

    def _debug(self, parent: Program) -> Program:
        """Attempt to fix a buggy parent program.

        Child is placed on ``parent.island`` to match original
        openevolve/database.py add() parent-island inheritance.
        """
        self._restore_snapshot(parent.code_snapshot)

        error_ctx = parent.error_context or "(No error captured.)"
        instruction = (
            f"The previous code modification caused an error. "
            f"Please fix the issue.\n\n"
            f"## Instructions\n"
            f"Fix the bug while preserving the intended improvement.\n"
            f"Make minimal changes to resolve the error."
        )

        edit_success = self._edit_code(instruction, error_context=error_ctx)
        child_snapshot = self._snapshot_target_files()

        val_result = self._execute_val(self.step_count)
        debug_val_duration = self._last_val_duration

        child = Program(
            code_snapshot=child_snapshot,
            parent_id=parent.id,
            generation=parent.generation + 1,
            island=parent.island,
            step_id=self.step_count,
            action="debug",
            primary_metric=val_result.get("primary_metric"),
            is_buggy=(
                not val_result.get("success")
                or val_result.get("primary_metric") is None
            ),
            error_context=self._truncate_error(val_result.get("error", "")),
            changes_description=f"Debug fix for {parent.id}",
            complexity=compute_complexity(child_snapshot),
            diversity=compute_diversity(
                child_snapshot, list(self.db._reference_programs)
            ),
            debug_depth=parent.debug_depth + 1,
        )

        child.val_duration = debug_val_duration  # store per-program for StepResult
        child.val_result = val_result
        child.edit_success = edit_success
        child._instruction = instruction
        child._editor_log_path = getattr(self._last_edit_result, "log_path", None)
        self._analyze_execution(child, val_result)
        self.db.add(child, iteration=0)

        return child

    # ------------------------------------------------------------------
    # Prompt construction (ported from prompt/sampler.py build_prompt)
    # ------------------------------------------------------------------

    def _build_evolution_instruction(
        self,
        parent: Program,
        inspirations: List[Program],
        top_programs: List[Program],
        iteration: int,
    ) -> str:
        """Build the CodeEditor instruction with OpenEvolve's evolutionary context.

        Replicates the semantic content from prompt/sampler.py diff_user.txt:
        - Current metrics
        - Areas for improvement
        - Evolution history (recent attempts on this island)
        - Top performers
        - Inspiration programs

        All island-scoped context (history, top performers, displayed island)
        uses ``parent.island`` — matching original openevolve/database.py
        where the whole prompt flow is anchored to parent.metadata["island"].
        """
        parts = []

        # Current solution performance
        dir_label = "higher" if self.metric_direction == "higher" else "lower"
        parts.append(
            f"## Current Solution Performance\n"
            f"{self._format_metric_line(parent.primary_metric, 'Current solution')}\n"
            f"- Generation: {parent.generation}, Island: {parent.island}"
        )

        # Baseline results
        parts.append(
            f"\n## Baseline Results\n"
            f"{self._format_metric_line(self.baseline_primary_metric, 'Baseline')}"
        )

        # Areas for improvement (auto-identified from population)
        improvement_areas = self._identify_improvement_areas(
            parent, top_programs
        )
        if improvement_areas:
            parts.append(f"\n## Areas for Improvement\n{improvement_areas}")

        # Evolution history (recent attempts on parent's island)
        history = self._build_evolution_history(parent.island)
        if history:
            parts.append(f"\n## Evolution History (Recent Attempts)\n{history}")

        # Top performers
        if top_programs:
            top_section = []
            for i, p in enumerate(top_programs):
                top_section.append(
                    f"{i+1}. [{self.metric_name}: {p.primary_metric}] "
                    f"{p.changes_description[:100]}"
                )
            parts.append(
                f"\n## Top Performers (Island {parent.island})\n"
                + "\n".join(top_section)
            )

        # Inspiration programs (diverse approaches)
        if inspirations:
            insp_section = []
            for i, p in enumerate(inspirations):
                insp_section.append(
                    f"{i+1}. [{self.metric_name}: {p.primary_metric}, Gen: {p.generation}] "
                    f"{p.changes_description[:100]}"
                )
            parts.append(
                "\n## Inspiration Programs (Diverse Approaches)\n"
                + "\n".join(insp_section)
            )

        # Instructions
        parts.append(
            "\n## Instructions\n"
            f"Improve the ML code to achieve better {self.metric_name} "
            f"({dir_label} is better).\n"
            "- Focus on a single, targeted change\n"
            "- Do not repeat approaches from Evolution History\n"
            "- Consider insights from Top Performers and Inspirations\n"
            "- Keep changes minimal and focused on one idea"
        )

        return "\n".join(parts)

    def _identify_improvement_areas(
        self, parent: Program, top_programs: List[Program]
    ) -> str:
        """Auto-identify improvement areas (from prompt/sampler.py)."""
        areas = []
        if parent.primary_metric is not None and top_programs:
            best_metric = top_programs[0].primary_metric
            if best_metric is not None and parent.primary_metric != best_metric:
                gap = abs(best_metric - parent.primary_metric)
                areas.append(
                    f"- Gap to best on this island: {gap:.6f}"
                )
        if parent.analysis:
            areas.append(f"- Last analysis: {parent.analysis[:200]}")
        return "\n".join(areas) if areas else ""

    def _build_evolution_history(self, island_id: int) -> str:
        """Build evolution history for an island (recent attempts).

        Includes feature coordinates and outcome classification to match
        original OpenEvolve's richer prompt context.
        """
        island_programs = [
            self.db.programs[pid]
            for pid in self.db.islands[island_id]
            if pid in self.db.programs
        ]
        # Sort by step_id (most recent first)
        island_programs.sort(
            key=lambda p: p.step_id if p.step_id is not None else 0,
            reverse=True,
        )

        lines = []
        for p in island_programs[:8]:
            # Status
            status = "success" if not p.is_buggy else "failed"

            # Feature coordinates (complexity/diversity bins)
            feat_key = self.db._calculate_feature_key(p)
            feat_str = f"features=({feat_key})"

            # Outcome classification vs parent
            outcome = ""
            if p.parent_id and p.parent_id in self.db.programs:
                parent_prog = self.db.programs[p.parent_id]
                if parent_prog.primary_metric is not None and p.primary_metric is not None:
                    delta = p.primary_metric - parent_prog.primary_metric
                    if self.metric_direction == "lower":
                        delta = -delta  # positive = improvement for "lower is better"
                    if delta > 0:
                        outcome = f"improved(+{abs(delta):.6f})"
                    elif delta < 0:
                        outcome = f"declined(-{abs(delta):.6f})"
                    else:
                        outcome = "stable"

            desc = p.changes_description[:100]
            metric_str = f"{self.metric_name}={p.primary_metric}"
            parts = [f"[{status}]", desc, f"| {metric_str}", feat_str]
            if outcome:
                parts.append(outcome)
            lines.append("- " + " ".join(parts))
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Post-execution analysis (not counted as step)
    # ------------------------------------------------------------------

    def _analyze_execution(self, program: Program, val_result: dict) -> None:
        """LLM analysis of execution results. NOT counted as a step."""
        success = val_result.get("success", False)
        metric = val_result.get("primary_metric")
        error = val_result.get("error", "")
        if error and len(error) > self.max_stderr_output:
            error = "..." + error[-self.max_stderr_output :]

        prompt = (
            "Briefly analyze this ML experiment result.\n\n"
            f"## Task\n{self.task_description[:500]}\n\n"
            f"## Execution Result\n"
            f"Success: {success}\n"
            f"Primary Metric ({self.metric_name}): {metric}\n"
        )
        if error:
            prompt += f"Error output:\n```\n{error[:500]}\n```\n\n"
        prompt += (
            "Provide a concise summary (2-3 sentences): what was attempted, "
            "what the result was, and what to try next."
        )

        try:
            text, _, usage = get_response_from_llm(
                prompt,
                client=self.client,
                model=self.config.model,
                system_message="You are an ML experiment analyst. Be concise.",
            )
            if usage:
                self.token_usage_log.append(usage)
            program.analysis = text.strip()[:300]
        except Exception as e:
            program.analysis = f"(analysis failed: {e})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_error(self, error: str) -> str:
        if not error:
            return ""
        if len(error) > self.max_stderr_output:
            return "..." + error[-self.max_stderr_output :]
        return error

    def _make_step_result(self, program: Program) -> StepResult:
        # Map OpenEvolve action to FML-bench action vocabulary
        action_map = {"seed": "draft", "evolve": "improve", "debug": "debug"}
        action = action_map.get(program.action, "improve")

        sid = program.step_id if program.step_id is not None else self.step_count
        token_start = getattr(program, "_token_start", None)
        step_tokens = self._collect_step_tokens(token_start) if token_start is not None else None
        snap_path = self._save_step_code_snapshot(
            sid, self._parent_workspace, snapshot=program.code_snapshot
        )

        meta = {
            "island": program.island,
            "generation": program.generation,
            "parent_id": program.parent_id or "",
            "code_snapshot_path": snap_path,
            "instruction": getattr(program, "_instruction", None),
            "analysis": program.analysis if program.analysis else None,
            "editor_log_path": getattr(program, "_editor_log_path", None),
        }

        return StepResult(
            step_id=sid,
            idea_id=program.id,
            idea_description=program.changes_description[:200],
            action=action,
            edit_success=program.edit_success,
            val_result=program.val_result if program.val_result is not None else {
                "primary_metric": program.primary_metric, "success": not program.is_buggy
            },
            primary_metric=program.primary_metric,
            token_usage=step_tokens,
            step_duration_seconds=getattr(program, "_step_duration", None),
            metadata=meta,
        )

    def _run_final_test(
        self, benchmark_config, agent_name, benchmark_name, parent_ts, timeout
    ) -> Optional[dict]:
        best_prog = self.db.get_best_program() if self.db else None
        if best_prog is None:
            print("No best program found, skipping test.")
            return None

        self.best_code_snapshot = best_prog.code_snapshot
        print(f"\nBest program: id={best_prog.id} metric={best_prog.primary_metric}")

        try:
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
