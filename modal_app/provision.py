"""
Modal provisioning entrypoints for the FML-bench eval backend.

Run with the Modal CLI on the orchestrator host (after `pip install modal` and
`modal token new`):

    modal run modal_app/provision.py::build           --task Causality_causalml
    modal run modal_app/provision.py::build_all
    modal run modal_app/provision.py::reap            [--max-age-seconds 7200]

`build`/`build_all` materialize the per-task image (which runs the REAL setup
functions at build time and records the working-tree hash — see images.py).
Image builds are content-addressed and idempotent: re-running with an unchanged
project rebuilds nothing.

`reap` is the orphan-sandbox backstop: graceful SIGINT is handled by
ModalExecutor's atexit/finally teardown, but hard kills (SIGKILL/OOM/host
crash) skip atexit and can leak a running sandbox; this command terminates
stale sandboxes tagged by this app (it is DRY-RUN by default — pass --force to
actually terminate). No server-side TTL is set on sandboxes (the unbounded
final test would be at risk), so `reap` is the hard-kill backstop; an operator
who wants a server-side cap can add `timeout=` in ModalExecutor._sb_create.

This module imports the Modal SDK at top level (fine for py_compile and for the
Modal CLI host where modal is installed). It is never imported on the local
eval path.
"""
from __future__ import annotations

import modal  # VERIFY (Modal SDK; absent locally, present on the Modal CLI host)

from modal_app import (
    MANIFEST_DIR,
    REMOTE_ROOT,
    SANDBOX_TAG,
    app,
    task_gpu,
    task_image,
    task_volumes,
)


def _all_tasks() -> list:
    """Task names to provision (sourced from setup.TASKS, re-rooted-safe import)."""
    import importlib
    import os
    import sys
    # Runs on the local Modal CLI host (cwd is usually the repo root, so `setup`
    # is normally importable), but make the repo-root path explicit so it works
    # regardless of how `modal run` set sys.path[0].
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    setup = importlib.import_module("setup")
    return list(setup.TASKS.keys())


@app.local_entrypoint()  # VERIFY (local_entrypoint decorator)
def build(task: str):
    """Force-build (or no-op if cached) the per-task image for *task*.

    Instantiating a sandbox forces Modal to materialize the image layers,
    including the build-time ``.run_function(_provision_task)`` that runs the
    real setup.py steps and records the tree-hash manifest. If you skip this
    entrypoint the image still builds lazily on the first eval's sandbox
    creation — ``build`` just pre-warms it and surfaces build errors early.
    A no-GPU throwaway sandbox is enough to trigger materialization (easyfsl's
    GPU build step is attached to its image's run_function, not the sandbox).
    """
    img = task_image(task)
    print(f"[provision] building image for task={task} gpu={task_gpu(task)} "
          f"volumes={list(task_volumes(task))} ...")
    sb = modal.Sandbox.create(app=app, image=img)  # VERIFY (forces image build)
    try:
        p = sb.exec("bash", "-c",
                    f"cat {REMOTE_ROOT}/{MANIFEST_DIR}/{task}.json")  # VERIFY
        p.wait()  # VERIFY
        out = p.stdout.read() if hasattr(p, "stdout") else ""  # VERIFY
        print(f"[provision] {task} manifest: {(out or '').strip() or '(not found)'}")
    finally:
        sb.terminate()  # VERIFY
    print(f"[provision] built image for task={task}")
    return img


@app.local_entrypoint()  # VERIFY
def build_all():
    """Build images for every task in setup.TASKS."""
    for task in _all_tasks():
        build(task)


def _sb_has_app_tag(sb) -> bool:
    """Best-effort: does *sb* carry THIS app's SANDBOX_TAG? Unknown -> False.

    Returning False for any sandbox we can't positively identify guarantees the
    reaper NEVER terminates an untagged / unrelated sandbox. # VERIFY (get_tags)
    """
    try:
        tags = sb.get_tags()  # VERIFY (Sandbox.get_tags)
        return all(tags.get(k) == v for k, v in SANDBOX_TAG.items())
    except Exception:  # noqa: BLE001
        return False


def _sb_age_seconds(sb):
    """Seconds since *sb* was created, or None if the accessor is unknown. # VERIFY."""
    import datetime as _dt
    import time
    try:
        created = sb.created_at  # VERIFY (created-at accessor: datetime or epoch seconds)
        if isinstance(created, _dt.datetime):
            return (_dt.datetime.now(created.tzinfo) - created).total_seconds()
        return time.time() - float(created)
    except Exception:  # noqa: BLE001
        return None


@app.local_entrypoint()  # VERIFY
def reap(max_age_seconds: int = 7200, force: bool = False):
    """Terminate ORPHANED fml-bench sandboxes (hard-kill backstop).

    Graceful SIGINT is handled by ModalExecutor's atexit/finally teardown; this
    is for hard kills (SIGKILL/OOM/host crash) that skip atexit. Lists ONLY
    sandboxes carrying this app's SANDBOX_TAG and older than *max_age_seconds*.

    DRY-RUN BY DEFAULT: without ``--force`` it only prints what it would reap, so
    it can never mass-terminate unrelated sandboxes on a shared account. The
    exact Sandbox.list filter + created-at/tag accessors are # VERIFY.
    """
    # Prefer a server-side tag filter; fall back to listing + client-side filter.
    try:
        sandboxes = list(modal.Sandbox.list(tags=SANDBOX_TAG))  # VERIFY (list tags= filter)
    except TypeError:
        sandboxes = [sb for sb in modal.Sandbox.list() if _sb_has_app_tag(sb)]  # VERIFY

    candidates = []
    for sb in sandboxes:
        age = _sb_age_seconds(sb)
        if age is None:
            # Fail CLOSED: never reap a sandbox whose age we cannot determine
            # (the created-at accessor may be wrong) — it could be a live eval.
            print(f"[reap] WARNING: cannot determine age of "
                  f"{getattr(sb, 'object_id', sb)}; skipping (fail-closed)")
            continue
        if age < max_age_seconds:
            continue  # too young -> leave it (likely a live eval)
        candidates.append(sb)

    if not candidates:
        print(f"[reap] no orphaned fml-bench sandboxes older than {max_age_seconds}s")
        return 0

    mode = "terminating" if force else "DRY-RUN (pass --force to terminate)"
    print(f"[reap] {len(candidates)} orphaned fml-bench sandbox(es) "
          f">{max_age_seconds}s old — {mode}:")
    reaped = 0
    for sb in candidates:
        sid = getattr(sb, "object_id", sb)
        if not force:
            print(f"[reap]   would terminate {sid}")
            continue
        try:
            sb.terminate()  # VERIFY (Sandbox.terminate)
            reaped += 1
            print(f"[reap]   terminated {sid}")
        except Exception as e:  # noqa: BLE001
            print(f"[reap]   failed to terminate {sid}: {e}")
    print(f"[reap] {'terminated' if force else 'would terminate'} "
          f"{reaped if force else len(candidates)} sandbox(es) (root={REMOTE_ROOT})")
    return reaped if force else len(candidates)
