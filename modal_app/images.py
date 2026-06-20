"""
Per-task Modal Image builder for the FML-bench eval backend.

Strategy (docs/MODAL_DESIGN.md §6): reproduce the *exact* local post-`setup.py`
working tree on the image filesystem, plus the task's conda env, by RE-ROOTING
``setup.PROJECT_ROOT`` / ``setup.WORKSPACE`` / ``setup.ML_TASKS`` onto the image
FS and calling the REAL setup functions:

    setup.CONDA_ENVS[<env>]()          # create the conda env (identical to local)
    setup.TASKS["<task>"][0](args)     # clone+commit repo & data (identical to local)

so the remote ``conda run -n <env>`` binary/semantics and the workspace tree
match local byte-for-byte (verified later by a recorded git tree-hash).

Order is env-then-task, EXCEPT ``Privacy_privacymeter`` whose env step is
``pip install -r <repo path>`` and therefore needs the repo cloned FIRST
(task-then-env). See ``_build_steps``.

This module imports the Modal SDK at top level. That is fine for
``python -m py_compile`` (compile does not execute imports) and for the Modal
build host (where ``modal`` is installed). It is NEVER imported on the local
eval path: ``benchmark/executor_factory.make_executor`` only imports
``benchmark.modal_executor`` when ``eval_backend == "modal"``, and
``modal_executor`` imports this lazily. Every Modal SDK touch is marked
``# VERIFY`` because it cannot be live-verified here (no Modal account).

Layout on the image (matches local repo layout so the relative ``cp
../../../ml_tasks/<task>/...`` commands in val_command resolve identically):

    /root/fml-bench/                 <- PROJECT_ROOT (REMOTE_ROOT)
        ml_tasks/<task>/...          <- baked, overwritten per-eval by the executor
        workspace/<task>/<repo>/...  <- post-setup working tree (the eval cwd)
        .fmlbench_manifest/<task>.json   <- recorded tree-hash manifest
"""
from __future__ import annotations

import modal  # noqa: F401  # VERIFY (Modal SDK; absent locally, present on build host)

# Remote project root on the image. Mirrors the local repo layout exactly so
# that repo_dir depth (and thus the relative cp paths in val/test commands)
# is preserved.
REMOTE_ROOT = "/root/fml-bench"
MANIFEST_DIR = ".fmlbench_manifest"  # under REMOTE_ROOT


# --------------------------------------------------------------------------
# Build-time helpers (run INSIDE the image during `.run_function(...)`).
# --------------------------------------------------------------------------
# These functions execute on the Modal build host where `modal` + the cloned
# repo are present. They are pure-Python + subprocess; no Modal SDK calls here.

def _reroot_setup(setup_module, project_root: str):
    """Point setup.py's module globals at the image FS instead of the local cwd."""
    from pathlib import Path
    root = Path(project_root)
    setup_module.PROJECT_ROOT = root
    setup_module.WORKSPACE = root / "workspace"
    setup_module.ML_TASKS = root / "ml_tasks"


def _build_args(skip_data: bool = False):
    """Construct the argparse-style namespace that the setup_<task> fns read.

    They only consult ``args.skip_data``; ``skip_data=False`` reproduces the
    full local clone+commit+data result.
    """
    import argparse
    return argparse.Namespace(skip_data=skip_data)


def _record_tree_hash(repo_dir: str, task: str, project_root: str) -> str:
    """Record `git rev-parse HEAD^{tree}` for the built repo into a manifest.

    Tree-hash (not commit SHA): commits embed nonreproducible timestamps, so
    only the working-tree content hash is reproducible across machines. The
    executor asserts the remote tree-hash equals this recorded value before the
    first eval.
    """
    import json
    import os
    import subprocess
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout.strip()
    manifest_dir = os.path.join(project_root, MANIFEST_DIR)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = {"task": task, "repo_dir": repo_dir, "tree_hash": tree}
    with open(os.path.join(manifest_dir, f"{task}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return tree


def _provision_task(task: str, project_root: str = REMOTE_ROOT):
    """Build-time entrypoint: create env(s) + workspace tree for *task*.

    Runs INSIDE the image (via Image.run_function). Imports the real ``setup``
    module that was copied onto the image, re-roots its globals, and calls the
    identical CONDA_ENVS / TASKS functions used by local ``python setup.py``.
    """
    import importlib
    import os
    import sys

    # Modal's run_function loader sets sys.path[0] to its own runtime entry, and
    # `.workdir(project_root)` only sets cwd (Python does NOT add cwd to sys.path
    # for an imported, non-__main__ module). The repo was baked via
    # `add_local_dir(".", remote_path=project_root, copy=True)` (a generic file
    # copy, not a Python-source mount), so `setup.py` at <project_root>/setup.py
    # is NOT importable unless we put project_root on sys.path explicitly. Without
    # this, `import_module("setup")` raises ModuleNotFoundError and the whole
    # "call the real setup functions" provisioning strategy fails at build time.
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    setup = importlib.import_module("setup")
    _reroot_setup(setup, project_root)

    setup_fn, envs = setup.TASKS[task]

    # privacy_meter needs a 3-step order: its env step is `pip install -r <repo
    # path>` (needs the repo cloned first), AND its data step (skip_data=False)
    # runs `conda_run("privacy_meter", ...)` which needs that env to already
    # exist. A single env-then-task or task-then-env pass is circular. So:
    #   1. setup_fn(skip_data=True): clone+commit the repo only (no env needed).
    #   2. CONDA_ENVS[env](): create the env from the now-present requirements.
    #   3. setup_fn(skip_data=False): clone_repo is idempotent (skips the clone),
    #      so this only runs the data-gen step, now that the env exists.
    # Every other task is env-then-task with a single full setup_fn call.
    if task == "Privacy_privacymeter":
        setup_fn(_build_args(skip_data=True))     # 1. clone repo only
        for env in envs:
            setup.CONDA_ENVS[env]()               # 2. create env from cloned requirements
        setup_fn(_build_args(skip_data=False))    # 3. generate data with env present
    else:
        for env in envs:
            setup.CONDA_ENVS[env]()
        setup_fn(_build_args(skip_data=False))

    # Resolve the post-setup repo_dir from the task's config.json (same source
    # the runner reads). gcastle's repo_dir is the inner trustworthyAI/gcastle
    # (a fresh `git init` single commit); config.json already encodes that path.
    import json
    cfg_path = os.path.join(project_root, "ml_tasks", task, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    repo_dir = os.path.join(project_root, cfg["repo_dir"])

    tree = _record_tree_hash(repo_dir, task, project_root)
    print(f"[modal-build] {task}: repo_dir={repo_dir} tree_hash={tree}")


# --------------------------------------------------------------------------
# Image construction (runs on the orchestrator / Modal client at deploy time).
# --------------------------------------------------------------------------

def _base_image():
    """continuumio/miniconda3 base with git + the project sources baked in."""
    return (
        modal.Image.from_registry("continuumio/miniconda3")  # VERIFY
        .apt_install("git", "wget", "unzip", "curl")          # VERIFY
        # Bake the whole project (setup.py + ml_tasks/ + configs) so the real
        # setup functions can run at build. ml_tasks is small/versioned; it is
        # re-pushed fresh per-eval by the executor regardless.
        #
        # ignore=: the build must reproduce the EXACT post-setup state, so it
        # must NOT bake host-side state. A populated local workspace/ would make
        # clone_repo short-circuit on an existing .git (setup.py) -> a
        # non-hermetic, host-dependent baseline. Exclude workspace/, the
        # project's own .git, result dirs, and caches; keep setup.py, ml_tasks/,
        # configs/, modal_app/, benchmark/ (all needed at build).
        .add_local_dir(  # VERIFY (add_local_dir ignore= pattern semantics)
            ".", remote_path=REMOTE_ROOT, copy=True,
            ignore=[
                ".git", ".git/**",
                "workspace/**",
                "results/**", "benchmark_results/**",
                "**/__pycache__/**", "**/*.pyc",
                "*.egg-info/**",
            ],
        )
        # Make the baked repo importable: `import setup` (and any project module
        # the setup fns reach for) needs REMOTE_ROOT on sys.path. `.workdir(...)`
        # only sets cwd, which Modal's run_function loader does NOT add to
        # sys.path; without PYTHONPATH the build-time `import_module("setup")`
        # would raise ModuleNotFoundError. _provision_task also inserts it
        # defensively, but setting it here covers subprocess/conda children too.
        #
        # GIT_*: setup.py's clone_repo/gcastle commit setup_files & datasets with
        # `git commit` but never configures a git identity. A Modal build host
        # has no ~/.gitconfig, so those commits would fail ("Author identity
        # unknown") -> gcastle's fresh `git init` makes no HEAD and
        # `git rev-parse HEAD^{tree}` (check=True) aborts the build; clone tasks
        # silently drop the setup-file commit. These env vars give git an
        # identity so the build reproduces the same committed HEAD as a local
        # setup.py run (which works because the author's machine has an identity).
        .env({
            "PYTHONPATH": REMOTE_ROOT,
            "GIT_AUTHOR_NAME": "fml-bench", "GIT_AUTHOR_EMAIL": "fml-bench@localhost",
            "GIT_COMMITTER_NAME": "fml-bench", "GIT_COMMITTER_EMAIL": "fml-bench@localhost",
        })                                                     # VERIFY
        .workdir(REMOTE_ROOT)                                  # VERIFY
    )


def task_image(task: str):
    """Return the per-task Modal Image (conda env + post-setup workspace tree).

    Special-cased tasks (easyfsl GPU builder + secrets, etc.) attach their
    extra build resources here; the build *logic* is uniform — call the real
    setup functions via ``_provision_task`` at build time.
    """
    img = _base_image()

    build_kwargs = {}
    # easyfsl runs `predict_embeddings --device=cuda` at BUILD time, so it needs
    # a GPU builder. Its downloads use bare unauthenticated curl/gdown (see
    # setup.py setup_data_efficiency_easyfsl), so NO Modal secrets are attached
    # here — Secret.from_name(...) would error at deploy if the secrets are
    # absent. If those endpoints ever require auth, attach the secret then.
    if task == "Data_Efficiency_easyfsl":
        build_kwargs["gpu"] = "A100"  # VERIFY (GPU builder; A100 like all GPU use)

    return img.run_function(  # VERIFY (build-time execution of _provision_task)
        _provision_task,
        args=(task,),
        **build_kwargs,
    )
