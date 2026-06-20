"""
modal_app — the Modal-side package for the FML-bench remote-GPU eval backend.

This package is the ONLY place that touches the Modal SDK. It is imported
lazily (and only) when ``--eval-backend modal`` is selected, so the local eval
path never imports ``modal`` (which is not installed locally). Every Modal SDK
call in this package is marked ``# VERIFY`` — none can be live-verified without
a Modal account.

Public interface consumed by ``benchmark/modal_executor.ModalExecutor``:

    app                      -> the modal.App ("fml-bench-eval")
    REMOTE_ROOT              -> remote PROJECT_ROOT path on the baked image
    MANIFEST_DIR             -> dir (under REMOTE_ROOT) holding per-task tree-hash manifests
    task_image(task)         -> per-task modal.Image (conda env + post-setup tree)
    task_gpu(task)           -> Modal GPU specifier string for the task's sandbox
    task_volumes(task)       -> {remote_mount_path: modal.Volume} for the task (may be empty)

``provision.py`` provides the build/deploy + reaper entrypoints separately.
"""
from __future__ import annotations

import modal  # noqa: F401  # VERIFY (Modal SDK; absent locally, present on Modal hosts)

from modal_app.gpu_map import task_gpu
from modal_app.images import MANIFEST_DIR, REMOTE_ROOT, task_image

# Single application object that owns all sandboxes for the eval backend.
app = modal.App("fml-bench-eval")  # VERIFY (App constructor)

# Tag stamped on every sandbox at creation so the reaper (provision.py::reap)
# can find ONLY this backend's orphans and never mass-terminate unrelated
# sandboxes on a shared account.
SANDBOX_TAG = {"fml_bench": "fml-bench-eval"}


def task_volumes(task: str) -> dict:
    """Return ``{remote_mount_path: modal.Volume}`` for *task* (default: none).

    Most tasks ship their data committed inside the baked repo (so no Volume is
    needed). Only genuinely external/huge/gated datasets are mounted from a
    read-only Volume; those are added here task-by-task as they are validated.
    """
    return {}


__all__ = [
    "app",
    "SANDBOX_TAG",
    "REMOTE_ROOT",
    "MANIFEST_DIR",
    "task_image",
    "task_gpu",
    "task_volumes",
]
