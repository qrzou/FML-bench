#!/usr/bin/env python3
"""
Task-level parallel launcher for the Modal eval backend (MODAL_DESIGN.md s7).

Spawns one local subprocess per task, each running the UNMODIFIED
run_agent_benchmark.py with --eval-backend modal. Local subprocess concurrency is
bounded by an asyncio.Semaphore; Modal schedules the GPU sandboxes concurrently and
bills GPU time only while an eval actually runs (each ModalExecutor manages its own
ephemeral sandboxes + atexit teardown). There is no seed/repeat knob -- FML-bench has
no external seed (task train seeds are hardcoded), so each task runs exactly once.

Usage:
    python launch_benchmark.py \\
      --agent-config configs/agents/aide.yaml \\
      --tasks all | Causality_causalml,Privacy_opacus \\
      --model gpt-5 --provider OpenAI \\
      --max-concurrency 16 \\
      --output-dir benchmark_results [--continue-on-failure] [key=value overrides...]

Pure stdlib only -- this orchestrator never imports modal; the spawned children own
the Modal SDK via their ModalExecutor.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TASK_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "tasks")
ML_TASKS_DIR = os.path.join(PROJECT_ROOT, "ml_tasks")
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "run_agent_benchmark.py")


# ---------------------------------------------------------------------------
# Task <-> config resolution
# ---------------------------------------------------------------------------

def _read_benchmark_name(config_path):
    """Best-effort read of the `benchmark.name` field from a task config YAML.

    Uses a tiny line scanner (no yaml dependency needed for this top-level field)
    so the launcher stays pure-stdlib; the spawned child does the real YAML load.
    """
    name = None
    in_benchmark = False
    try:
        with open(config_path, "r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                if indent == 0:
                    in_benchmark = stripped.startswith("benchmark:")
                    continue
                if in_benchmark and stripped.startswith("name:"):
                    name = stripped.split(":", 1)[1].strip().strip("'\"")
                    break
    except OSError:
        return None
    return name


def build_task_config_map():
    """Map each task name (== benchmark.name == ml_tasks/ dir == setup.TASKS key)
    to its task-config path under configs/tasks/."""
    mapping = {}
    if not os.path.isdir(TASK_CONFIG_DIR):
        return mapping
    for fname in sorted(os.listdir(TASK_CONFIG_DIR)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(TASK_CONFIG_DIR, fname)
        name = _read_benchmark_name(path)
        if name:
            mapping[name] = path
    return mapping


def list_all_tasks():
    """Resolve "all" to the task names known to the benchmark.

    Prefer setup.TASKS keys (the canonical registry); fall back to the ml_tasks/
    directory listing if setup.py cannot be imported.
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        import setup as _setup  # noqa: F401  (local import; never modal)
        return list(_setup.TASKS.keys())
    except Exception:
        if os.path.isdir(ML_TASKS_DIR):
            return sorted(
                d for d in os.listdir(ML_TASKS_DIR)
                if os.path.isdir(os.path.join(ML_TASKS_DIR, d))
            )
        return []


def resolve_tasks(tasks_arg, config_map):
    """Resolve the --tasks value to a list of (task_name, config_path) pairs.

    Errors (unknown task, no config) are returned as messages so the caller can
    report them all at once.
    """
    if tasks_arg.strip() == "all":
        requested = list_all_tasks()
    else:
        requested = [t.strip() for t in tasks_arg.split(",") if t.strip()]

    resolved, errors = [], []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        cfg = config_map.get(name)
        if cfg is None:
            errors.append(f"  - {name}: no task config found in {TASK_CONFIG_DIR}")
            continue
        resolved.append((name, cfg))
    return resolved, errors


# ---------------------------------------------------------------------------
# Subprocess orchestration
# ---------------------------------------------------------------------------

def build_child_command(task_config, args):
    """Construct the argv for the unmodified run_agent_benchmark.py child."""
    cmd = [
        sys.executable, RUN_SCRIPT,
        "--agent-config", args.agent_config,
        "--task-config", task_config,
        "--eval-backend", "modal",
        "--output-dir", args.output_dir,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.provider:
        cmd += ["--provider", args.provider]
    if args.workspace_label:
        cmd += ["--workspace-label", args.workspace_label]
    if args.save_code_backup:
        cmd += ["--save-code-backup"]
    # Positional key=value overrides, passed through verbatim.
    cmd += list(args.overrides)
    return cmd


async def run_task(name, task_config, args, log_dir, semaphore, status):
    """Run a single task as a child subprocess, bounded by the semaphore."""
    if _SHUTTING_DOWN:
        return _skip_task(name, task_config, status)
    async with semaphore:
        # Re-check after acquiring the slot: a SIGINT may have arrived while we
        # were queued. Freeing a semaphore slot must NOT let a queued task spawn
        # a fresh Modal GPU sandbox after shutdown began.
        if _SHUTTING_DOWN:
            return _skip_task(name, task_config, status)
        cmd = build_child_command(task_config, args)
        stdout_path = os.path.join(log_dir, f"{name}.stdout")
        stderr_path = os.path.join(log_dir, f"{name}.stderr")
        exit_path = os.path.join(log_dir, f"{name}.exit")

        start = datetime.now()
        status[name] = {
            "task": name,
            "task_config": task_config,
            "command": cmd,
            "status": "running",
            "started_at": start.isoformat(),
        }
        print(f"[launch] starting {name}", flush=True)

        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=out,
                stderr=err,
                cwd=PROJECT_ROOT,
                # New process group so a SIGINT to the launcher can be broadcast to
                # children explicitly (and each child's own signal handler tears down
                # its ModalExecutor sandboxes via atexit / finally).
                start_new_session=True,
            )
            _RUNNING[name] = proc
            try:
                returncode = await proc.wait()
            finally:
                _RUNNING.pop(name, None)

        end = datetime.now()
        with open(exit_path, "w") as f:
            f.write(f"{returncode}\n")

        status[name].update({
            "status": "succeeded" if returncode == 0 else "failed",
            "returncode": returncode,
            "finished_at": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            "stdout": stdout_path,
            "stderr": stderr_path,
        })
        print(
            f"[launch] {'done' if returncode == 0 else 'FAILED'} {name} "
            f"(exit={returncode}, {status[name]['duration_seconds']:.1f}s)",
            flush=True,
        )
        return returncode


# Live child subprocesses, for SIGINT broadcast.
_RUNNING = {}
# Set once a SIGINT/teardown begins so queued tasks skip spawning new sandboxes.
_SHUTTING_DOWN = False


def _skip_task(name, task_config, status):
    """Record a task skipped because shutdown began (it was never spawned)."""
    status[name] = {
        "task": name,
        "task_config": task_config,
        "status": "skipped",
        "returncode": 130,
    }
    print(f"[launch] skipping {name} (shutting down)", flush=True)
    return 130


def _terminate_children():
    """Signal shutdown, then forward SIGINT to every live child process group
    (graceful teardown: each child's signal handler runs ModalExecutor
    atexit/finally to release its sandboxes). Best-effort; idempotent. Setting
    _SHUTTING_DOWN makes queued tasks skip spawning instead of opening new
    Modal sandboxes after Ctrl-C."""
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True
    for name, proc in list(_RUNNING.items()):
        if proc.returncode is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass


def write_status(log_dir, status, meta):
    """Write the aggregate launch_status.json snapshot."""
    payload = dict(meta)
    payload["tasks"] = status
    path = os.path.join(log_dir, "launch_status.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Task-level parallel launcher for the FML-bench Modal eval backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_benchmark.py --agent-config configs/agents/aide.yaml \\
      --tasks all --model gpt-5 --provider OpenAI --max-concurrency 16
  python launch_benchmark.py --agent-config configs/agents/aide.yaml \\
      --tasks Causality_causalml,Privacy_opacus --continue-on-failure \\
      agent.aide.num_drafts=3
        """,
    )
    p.add_argument("--agent-config", type=str, required=True,
                   help="Agent config file (e.g., configs/agents/aide.yaml)")
    p.add_argument("--tasks", type=str, required=True,
                   help="'all' or a comma-separated list of task names "
                        "(e.g., Causality_causalml,Privacy_opacus)")
    p.add_argument("--model", type=str,
                   help="Model to use (e.g., gpt-5, gemini-2.5-pro)")
    p.add_argument("--provider", type=str,
                   help="Provider to use (e.g., OpenAI, Google, OpenRouter)")
    p.add_argument("--output-dir", type=str, default="benchmark_results",
                   help="Root directory for experiment outputs (default: benchmark_results)")
    p.add_argument("--workspace-label", type=str, default=None,
                   help="Optional label forwarded to each child run.")
    p.add_argument("--save-code-backup", action="store_true", default=False,
                   help="Forward --save-code-backup to each child run.")
    p.add_argument("--max-concurrency", type=int, default=16,
                   help="Max concurrent local child subprocesses (default: 16). "
                        "Modal schedules the GPU sandboxes concurrently.")
    p.add_argument("--continue-on-failure", action="store_true", default=False,
                   help="Keep launching remaining tasks if one fails "
                        "(the default: a failed task never aborts the others, "
                        "but the launcher exits non-zero unless this is set).")
    p.add_argument("--list-tasks", action="store_true",
                   help="List resolvable task names (those with a task config) and exit.")
    p.add_argument("overrides", nargs="*",
                   help="Config overrides in format key=value, forwarded to each child.")
    return p.parse_args()


async def _amain(args):
    config_map = build_task_config_map()

    if args.list_tasks:
        all_tasks = list_all_tasks()
        print("Resolvable tasks (task name -> config):")
        for name in all_tasks:
            cfg = config_map.get(name)
            print(f"  - {name}: {cfg if cfg else '(no config found)'}")
        return 0

    if args.max_concurrency < 1:
        print("Error: --max-concurrency must be >= 1", file=sys.stderr)
        return 2

    if not os.path.isfile(args.agent_config):
        print(f"Error: agent config not found: {args.agent_config}", file=sys.stderr)
        return 2

    resolved, errors = resolve_tasks(args.tasks, config_map)
    if errors:
        print("Error: could not resolve some tasks:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
    if not resolved:
        print("Error: no runnable tasks resolved.", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(PROJECT_ROOT, "launch_logs", ts)
    os.makedirs(log_dir, exist_ok=True)

    meta = {
        "timestamp": ts,
        "agent_config": args.agent_config,
        "model": args.model,
        "provider": args.provider,
        "output_dir": args.output_dir,
        "eval_backend": "modal",
        "max_concurrency": args.max_concurrency,
        "continue_on_failure": args.continue_on_failure,
        "overrides": list(args.overrides),
        "num_tasks": len(resolved),
    }

    print(f"[launch] {len(resolved)} task(s); logs -> {log_dir}", flush=True)
    status = {}
    write_status(log_dir, status, meta)

    semaphore = asyncio.Semaphore(args.max_concurrency)

    # SIGINT/SIGTERM: broadcast to children, let their handlers tear down sandboxes.
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _terminate_children)
        except (NotImplementedError, RuntimeError):
            pass  # e.g. non-main thread / unsupported platform

    tasks = [
        asyncio.ensure_future(
            run_task(name, cfg, args, log_dir, semaphore, status)
        )
        for name, cfg in resolved
    ]

    # return_exceptions=True: with task-level isolation, one task's crash never
    # aborts the others (subprocesses are already isolated; this guards launcher
    # plumbing errors too).
    results = await asyncio.gather(*tasks, return_exceptions=True)

    write_status(log_dir, status, meta)

    failures = []
    for (name, _cfg), res in zip(resolved, results):
        if isinstance(res, Exception):
            failures.append(name)
            status.setdefault(name, {"task": name})
            status[name].update({"status": "error", "error": repr(res)})
        elif res != 0:
            failures.append(name)
    write_status(log_dir, status, meta)

    succeeded = len(resolved) - len(failures)
    print(f"\n[launch] {succeeded}/{len(resolved)} succeeded; "
          f"status -> {os.path.join(log_dir, 'launch_status.json')}", flush=True)
    if failures:
        print(f"[launch] failed: {', '.join(failures)}", flush=True)

    if failures and not args.continue_on_failure:
        return 1
    return 0


def main():
    args = parse_args()
    try:
        rc = asyncio.run(_amain(args))
    except KeyboardInterrupt:
        _terminate_children()
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
