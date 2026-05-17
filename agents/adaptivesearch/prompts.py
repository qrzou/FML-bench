"""Prompt templates for AdaptiveSearch Phase-2.

Three sub-rule prefixes (A, B, A∧B) are inserted at the top of an autoresearch-style
idea prompt; a separate branch-init template is used on each branch's first step
to enforce mechanistic diversity across siblings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


SUBRULE_A_PREFIX = (
    "[STRATEGY HINT — GO DEEPER]\n"
    "This branch has stagnated (only {keeps_in_window} new best metric(s) in the last "
    "{window} steps; threshold {fmax}) and your modifications remain close to the "
    "baseline (reach_norm = {reach_norm:.3f}; threshold {reach_thr:.2f}).\n"
    "Make a substantively DEEPER algorithmic change in your next idea:\n"
    "  1. The mechanism should differ qualitatively from the baseline — a different "
    "algorithmic family, a different mathematical formulation, OR a different "
    "conceptual approach.\n"
    "  2. The change should affect a substantial portion of the code, not merely "
    "hyperparameter tweaks.\n"
)

SUBRULE_B_PREFIX = (
    "[STRATEGY HINT — STAY COHERENT]\n"
    "This branch has stagnated (only {keeps_in_window} new best metric(s) in the last "
    "{window} steps; threshold {fmax}) yet your recent attempts span many directions "
    "(effective dimension = {effdim:.3f}; threshold {effdim_thr:.2f}).\n"
    "Maintain the OVERALL DIRECTION of recent improvements:\n"
    "  1. Refine within the same family of techniques and the same conceptual "
    "approach as the most successful recent attempts.\n"
    "  2. Avoid jumping to a fundamentally new mechanism.\n"
)

SUBRULE_AB_PREFIX = (
    "[STRATEGY HINT — CONSOLIDATE AND DEEPEN]\n"
    "This branch has stagnated despite many attempts that remain close to the "
    "baseline (reach_norm = {reach_norm:.3f}, effective dimension = {effdim:.3f}).\n"
    "CONSOLIDATE the most promising recent direction (same family of techniques) "
    "AND extend it with a substantively deeper algorithmic change (different "
    "mechanism, larger code change) within that direction.\n"
)


BRANCH_INIT_TEMPLATE = (
    "[BRANCH INITIALISATION]\n"
    "You are starting exploration branch {branch_id} from a previously promising "
    "candidate found by greedy search.\n\n"
    "## Parent candidate\n"
    "- Step {parent_step}, {metric_name} = {parent_metric}\n"
    "- Description: {parent_idea}\n\n"
    "## Other branches' first ideas (siblings; for diversity)\n"
    "{sibling_block}\n\n"
    "## Cross-branch journal (recent activity from all branches)\n"
    "{journal_block}\n\n"
    "## REQUIRED PROPERTY OF YOUR FIRST IDEA\n"
    "Your first idea on this branch MUST be **mechanistically different** from any "
    "sibling first idea listed above. Specifically:\n"
    "  - A different algorithmic family, OR\n"
    "  - A different mathematical formulation, OR\n"
    "  - A different conceptual approach.\n"
    "Hyperparameter tweaks or surface-level edits of a sibling's idea do NOT "
    "count as mechanistically different.\n"
)


@dataclass
class JournalEntry:
    branch_id: int
    step_id: int
    status: str
    primary_metric: Optional[float]
    idea: str  # full text, no truncation


def format_journal(entries: List[JournalEntry], metric_name: str, n_recent: int = 30) -> str:
    """Flatten the most recent ``n_recent`` cross-branch entries into a textual log.

    Ideas are kept verbatim. A safety valve trims to 800 chars per idea only when
    the joined output would exceed ~30k chars (~7-8k tokens).
    """
    if not entries:
        return "(no Phase-2 activity yet)"
    recent = entries[-n_recent:]
    lines = []
    for e in recent:
        if e.primary_metric is None:
            metric_str = "CRASH"
        else:
            metric_str = f"{e.primary_metric:.6f}"
        lines.append(
            f"[branch {e.branch_id} | step {e.step_id} | {e.status:<7s} | "
            f"{metric_name}={metric_str}]\n{e.idea}"
        )
    out = "\n\n".join(lines)
    if len(out) > 30_000:
        # safety valve: re-render with per-idea truncation
        lines = []
        for e in recent:
            metric_str = "CRASH" if e.primary_metric is None else f"{e.primary_metric:.6f}"
            idea = e.idea if len(e.idea) <= 800 else e.idea[:800] + " […truncated…]"
            lines.append(
                f"[branch {e.branch_id} | step {e.step_id} | {e.status:<7s} | "
                f"{metric_name}={metric_str}]\n{idea}"
            )
        out = "\n\n".join(lines)
    return out


def format_sibling_block(sibling_ideas: List[tuple]) -> str:
    """Format ``[(branch_id, idea), ...]`` (full ideas, no truncation)."""
    if not sibling_ideas:
        return "(none — you are the first branch to initialise)"
    return "\n\n".join(
        f"### sibling branch {bid} — first idea\n{idea}"
        for bid, idea in sibling_ideas
    )
