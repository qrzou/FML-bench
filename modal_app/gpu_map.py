"""
Per-task GPU + image/volume registry for the Modal eval backend.

This module is the single source of truth for *which* GPU type a task's eval
sandbox requests, and (lazily) which Modal Image / Volumes back it. It is
imported by both ``benchmark/modal_executor.py`` (to size the sandbox) and
``modal_app/provision.py`` (to build images). It does NOT import the Modal SDK
at module import time beyond what ``images.py`` needs, and contains no SDK
calls itself, so it stays cheap to import.

GPU strings are Modal GPU specifiers passed verbatim to the sandbox API. Every
GPU-using task uses "A100" (project choice); CPU-only tasks map to None (no GPU).
# VERIFY (the exact "A100" label must be confirmed against the live Modal API).

Tasks are keyed by their FML-bench task name (== ``benchmark_name`` ==
``ml_tasks/<task>/`` directory name).
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Per-task GPU selection.
# --------------------------------------------------------------------------
# Default GPU for any GPU-using task: A100 for ALL of them (per project choice —
# every GPU task uses A100). A task not explicitly listed below falls back here.
DEFAULT_GPU = "A100"  # VERIFY (Modal GPU label)

# Per-task GPU. Every GPU-using task uses the A100 default; the dict only needs
# to record the CPU-ONLY tasks (mapped to None -> the sandbox is created without
# a gpu= request). To pin a specific task to a different GPU later, add it here.
TASK_GPU = {
    "Fairness_fairlearn": None,   # sklearn-only env -> no GPU requested
    "Causality_gcastle":  None,   # cpu-only torch env -> no GPU requested
}


def task_gpu(task: str):
    """Return the Modal GPU specifier string for *task*, or None for a CPU-only
    task (the sandbox is then created WITHOUT a gpu= request). Unlisted tasks get
    DEFAULT_GPU."""
    return TASK_GPU.get(task, DEFAULT_GPU)
