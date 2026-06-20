# FML-bench × Modal remote-GPU eval backend — Design Plan v2.1

> Status: **implemented on branch `public_modal`** (local path verified; Modal path
> not yet live-verified — no Modal account). v2.1 folds in the corrections from a second
> adversarial plan audit (see §0a); a third (code) audit's fixes are applied.
>
> **Implementation note — `CONDA_ENV_SPECS` superseded:** the build does NOT add a
> `CONDA_ENV_SPECS` table to `setup.py`. Instead `modal_app/images.py` re-roots
> `setup.PROJECT_ROOT/WORKSPACE/ML_TASKS` and calls the REAL `setup.CONDA_ENVS[<env>]()`
> + `setup.TASKS[<task>][0](args)` at image build — identical commands to local, and
> **`setup.py` is left byte-unchanged** (strictly better for R1). Wherever this doc says
> "`CONDA_ENV_SPECS`", read "call the real setup functions at build". privacy_meter is
> handled by call ordering (below), not a placeholder table.

## 0. Goals & hard constraints

- **Goal:** keep the agent loop (AIDE/MCTS search, per-step `CodeEditor` LLM edits,
  in-memory target-file snapshots, metric parsing, result-dir writing) running
  locally/centrally, and dispatch **only the experiment execution**
  (`BenchmarkExecutor.run_val` / `run_test`) to a Modal GPU sandbox. This gives
  task-level parallelism with no local GPU fleet and no manual `CUDA_VISIBLE_DEVICES`
  pinning.
- **R1 — local path byte-identical:** running the original README way (no `--eval-backend
  modal`) must behave exactly as today. Modal is strictly opt-in; not installing the
  `modal` package must not affect the local path.
- **R2 — Modal == local fidelity:** the agent logic, the benchmark-framework logic, and
  the task content must all be identical to the original; the *only* difference is that
  the experiment-execution step runs on Modal. Ideal: with identical hardware/software
  and a deterministic LLM, results with and without Modal are identical.
- **R3 — no unfounded claims:** every design claim must match the actual codebase.
- **Operating constraint:** the author has **no Modal account/credits**, so the remote
  path cannot be live-verified here. All Modal SDK calls are isolated behind a thin
  adapter and marked `# VERIFY`; everything not requiring Modal is verified locally; a
  `docs/MODAL.md` runbook lets the author verify the remote path later.

## 0a. v2 → v2.1 audit corrections

The second audit confirmed v2 fixed every v1 mistake and that **Phase 1 (the no-Modal
local refactor) is sound to start with no blocking issue**. The Modal phases are
architecturally sound; v2.1 corrects these (none affect R1):

1. **Pilot is TensorFlow, not "CPU/torch".** `Causality_causalml` uses TensorFlow
   (`ml_tasks/Causality_causalml/train.py`) and grabs a GPU if one is visible. Relabeled
   throughout; the pilot should run on a GPU sandbox so it actually exercises
   device-selection scrubbing.
2. **Add a stateful checkpoint-handoff gate.** causalml's `final_test` reads no
   `pre_test_val` state, so the pilot validates single-eval parity but NOT the shared-sandbox
   `pre_test_val → final_test` checkpoint handoff that ~8 stateful tasks rely on. Add a
   second gate on a stateful task (**`Privacy_opacus`**) before broad rollout.
3. **easyfsl is not a CPU/clean builder.** `setup_data_efficiency_easyfsl` runs a gated
   `gdown` + a Kaggle download + `predict_embeddings ... --device=cuda` *at build*. It needs
   a GPU builder + build secrets, or the embeddings/model pre-baked onto a Volume — it is
   NOT covered by the "uniformly call the setup fn at build" framing.
4. **privacy_meter needs a 3-step build order.** Its env step is `pip install -r
   {WORKSPACE/...}` (needs the repo cloned first), AND its data step runs
   `conda_run("privacy_meter", ...)` (needs that env to exist) — so a single pass either way
   is circular. Implemented as: (1) `setup_fn(skip_data=True)` clones+commits the repo only;
   (2) `CONDA_ENVS[env]()` creates the env from the now-present requirements; (3)
   `setup_fn(skip_data=False)` (clone is idempotent) runs only the data-gen step with the env
   present. No placeholder substitution is needed.
5. **gcastle HEAD composition differs.** Its `repo_dir` is the inner `trustworthyAI/gcastle`
   with a **fresh `git init` single commit** (sparse-checkout of the outer repo @`58abc35`,
   then inner `git init`), so "HEAD = pinned + setup + data on one repo" does not apply to it.
6. **Fidelity check by git *tree-hash*, not commit SHA.** Commits embed nonreproducible
   timestamps, so commit SHAs differ across machines; compare the working-tree / `HEAD^{tree}`
   content hash instead.
7. **Graceful SIGINT must not leak the sandbox.** Because `kill_running_process` is decoupled
   from sandbox teardown, `ModalExecutor` itself owns teardown via a `_run_phase` try/finally
   **and an `atexit` hook** (the shared signal handler calls `sys.exit`, which runs `atexit`)
   — no change to the shared handler (R1-safe). `_sb_create` sets an explicit sandbox
   max-lifetime (`self.timeout + 1800s`, or a 24h cap for the unbounded final test) that
   auto-terminates a leaked sandbox server-side; the tagged-sandbox reaper (`provision.py::reap`)
   is the earlier hard-kill backstop.
8. **Soften universal wording** ("byte-identical", "exact HEAD", "pilot confirms hermeticity")
   to the scoped, code-grounded claims below.

## 1. Changes from v1 (audit-driven)

| # | v1 (wrong/incomplete) | v2 / v2.1 (fixed) | Why |
|---|---|---|---|
| 1 | Bake the bare `pinned_commit` | **Reproduce the exact local post-`setup.py` working tree** by running the real per-task `setup.py` step at image build; verify by git **tree-hash** (not commit SHA). Placement varies: `clone_repo` commits `setup_files`; `git_commit_data` commits data into the repo; gcastle is a fresh-init inner repo | `_reset_git` targets the local post-setup tree; commits embed timestamps so only the tree content is reproducible |
| 2 | Datasets on a uniform read-only Volume | **Per-task data placement**: in-repo committed data ships inside the baked repo; domainbed reads a repo sibling `../data`; only genuinely external/huge/gated data goes on a Volume (+ secrets) | Most datasets are committed into the repo; causalml reads an in-repo CSV; domainbed reads `../data` |
| 3 | Never push the per-run `cp ../../../ml_tasks/...` harness files | **Push the whole local `ml_tasks/<task>/` fresh into the remote tree on every eval** | Those harness scripts (`train.py`, `original_file_backup/*`) are not in `target_files`; a baked snapshot can silently go stale |
| 4 | Ephemeral sandbox **git-resets every step** | **No per-step reset**: each ephemeral sandbox starts from the exact baked baseline. Search-loop evals are hermetic *by code* (every `val` trains from scratch); this is gated by the byte-diff. Persistent-working-dir fallback for any non-hermetic task | Local `_reset_git` runs **once** in `setup_workspace`, never per `run_val`; v2.1 states this diverges deliberately and is sound only for hermetic evals |
| 5 | `timeout`+pgkill, `None` case unspecified | **`self.timeout is None` ⇒ unbounded remote exec** (final test); a positive timeout is enforced via `Sandbox.exec(timeout=)` + `ExecTimeoutError` (**Option B** — modal 1.5 has no per-process kill) | base.py sets `executor.timeout = None` for the final test |
| 6 | Remote exec contract left to a `# VERIFY` comment | **Pinned exec contract**: byte-identical argv `conda run --no-capture-output -n <env> bash -c <command>`, `cwd`=remote repo_dir, scrubbed env (no Modal-injected `CUDA_VISIBLE_DEVICES`/login-shell/extra CUDA paths) | Otherwise device selection / nondeterministic kernels can diverge |
| 7 | `kill_running_process` semantics | **modal 1.5 has no per-process kill, so `kill_running_process` terminates the sandbox** (which stops the eval), no-op when idle; teardown also lives in `_run_phase` finally + an `atexit` hook + `cleanup` (**Option B**) | Also called on the happy-path `finally` in `run_agent_benchmark.py`, where it is a benign idempotent no-op when idle; teardown must not depend on it |
| 8 | `setup.py` "additive export" (vague) | **Implemented differently — see header note:** the build calls the REAL `setup.CONDA_ENVS[env]()` at image-build time, so **`setup.py` is unchanged** (no `CONDA_ENV_SPECS` table). **privacy_meter** is special-cased by call ordering (clone→env→data) | Reuses the real conda-env commands verbatim; `setup.py` stays byte-identical (better for R1) |
| 9 | "reuse setup fns **and** checkout config.json pinned_commit" (contradictory) | **Reuse the setup fns** (they own the commit); the provisioner **records the built tree-hash**; `ModalExecutor` asserts the remote tree-hash matches it | setup fns hardcode their commit and never read config.json; `pinned_commit` is only presence-checked in `runner.py`'s new-format required-keys list |
| 10 | "result-dir writing byte-identical" | Scope that claim to the **metric-bearing `*_info.json`**; `--save-code-backup` is warned/no-op'd in Modal mode | `_backup_files` would otherwise capture only the local target-file diff (scoring never reads `code_backup/`) |

## 2. Architecture

```
LOCAL / central host (loop unchanged)                    MODAL (opt-in only)
 run_agent_benchmark.py / launch_benchmark.py
   └ BenchmarkRunner ─ agent.run()
       ├ CodeEditor edits target_files (local LLM call)   per eval: push target files +
       ├ _snapshot / _restore (local disk)                ml_tasks/<task>/ bytes (KB–MB)
       └ make_executor(..., eval_backend)        ───────► ephemeral GPU Sandbox
           local → BenchmarkExecutor (IDENTICAL)            image = conda env + EXACT
           modal → ModalExecutor                            post-setup workspace/<task>/
               _collect_results / metric parse  ◄───────    (repo working tree + data)
               run LOCALLY on pulled JSON          pull      conda run -n <env> bash -c <cmd>
                                                  *_info.json + logs + returncode
   one local subprocess per task  ⇒  Modal schedules sandboxes concurrently; GPU billed
                                      ONLY while an eval runs (ephemeral per eval)
```

## 3. Backend switch (R1 — unchanged from v1, audit-confirmed safe)

New flag `--eval-backend {local,modal}`, **default `local`** (env fallback
`FMLBENCH_EVAL_BACKEND`), threaded via `runtime_params["eval_backend"]`.

```python
# benchmark/executor_factory.py
def make_executor(*args, eval_backend="local", **kwargs):
    if eval_backend == "modal":
        from benchmark.modal_executor import ModalExecutor   # lazy; modal never imported on local path
        return ModalExecutor(*args, **kwargs)
    return BenchmarkExecutor(*args, **kwargs)                 # identical class + identical args
```

When `local` (default/unset): returns `BenchmarkExecutor(*args, **kwargs)` with the exact
args call sites pass today; `eval_backend` is consumed by the factory and never forwarded
to the constructor; `modal` is never imported; no method is overridden. The local control
flow does not change.

## 4. Execution & state model

**Primary model = stateless-per-eval from the exact baked baseline (with the test pair
sharing one sandbox).**

| Phase | run_id | Sandbox | GPU |
|---|---|---|---|
| Search-loop `run_val` | int 0,1,2… | create (fresh container = exact baked baseline) → push files → run → **terminate** | only during eval |
| Candidate generation (CodeEditor/AIDE) | — | none | **not billed** |
| Final test `run_val("pre_test_val")` | `"pre_test_val"` | create → run → **keep** | during eval |
| Final test `run_test("final_test")` | `"final_test"` | **reuse the kept sandbox** → run → terminate | during eval |

- A fresh container from the baked image *is* the clean post-setup baseline. v2.1 is explicit
  that this **deliberately diverges** from local semantics — local `_reset_git` runs only once
  in `setup_workspace`, so local accumulates untracked artifacts across search steps. It is
  equivalent **only for hermetic search-loop evals**, which holds *by code*: every `val`
  re-runs the train script from scratch and reads only target files + data.
- **Gate, not assumption:** the Modal-vs-local **byte-diff** on the pilot is the hard check;
  if any task proves non-hermetic across steps, that executor uses the **per-task persistent
  working-dir Volume** fallback (seeded once, reset once, mounted RW by each ephemeral
  sandbox), restoring exact local cumulative semantics while keeping GPU ephemeral. The Volume
  is a per-task fallback, **not** the default.
- `pre_test_val` and `final_test` share one sandbox (no think-time between them), so the
  checkpoint trained by `pre_test_val` is read by `final_test` on the same filesystem. Because
  causalml does not exercise this handoff, a **separate stateful-task gate (`Privacy_opacus`)**
  validates it before generalizing.

**Per-eval sync protocol:** push = current `target_files` bytes + the whole local
`ml_tasks/<task>/` (overwriting the baked copy, guaranteeing the `cp`-ed harness scripts
are current; pushed before the command runs, never clobbering agent edits to target files);
pull = `results_tmp/<phase>_info.json` + stdout/stderr/returncode. Never push the
repo/datasets; never pull checkpoints.

## 5. `ModalExecutor` (`benchmark/modal_executor.py`)

`ModalExecutor(BenchmarkExecutor)`; **`benchmark/executor.py` is not changed**. Overrides:

- `__init__` — `super().__init__(...)`; **assert `"val_command" in config`** (Modal backend
  supports only new-format configs; old-format `execute_commands` is refused with a clear
  error); derive the remote repo path preserving exact directory depth (gcastle's inner
  `trustworthyAI/gcastle` → `../../../../`); `self._sb = None`; register an `atexit` hook that
  terminates `self._sb` if still alive.
- `setup_workspace()` — `super().setup_workspace()` builds the local result dir (local
  `_reset_git` on the local copy is harmless). Does **not** create a sandbox yet.
- `_run_phase(...)` — **wholesale override** mirroring the parent's structure but executing
  in one sandbox, reusing the parent's `_collect_results` / `_extract_primary_metric` /
  `_filter_results` / `_save_bug_execution_record` unchanged, **wrapped in try/finally** so
  the sandbox is torn down even on interruption. Sequence: resolve command → local: clean
  `results_tmp` + make `execution_<ts>/` → ensure sandbox (create if `self._sb is None`,
  asserting the remote tree-hash matches the recorded build hash) → push target files +
  `ml_tasks/<task>/` → remote `rm -rf results_tmp` → (new-format only) main command via
  `_sb_exec` → remote integrity check → on failure: save bug record locally + teardown
  (unless `pre_test_val`) + return parent-shaped failure dict → pull
  `results_tmp/<phase>_info.json` to local `repo_dir/results_tmp/` → teardown sandbox (unless
  `run_id == "pre_test_val"`) → return `super()._collect_results(...)`.
- `_sb_exec(command)` — runs the **pinned exec contract**: argv exactly
  `["conda","run","--no-capture-output","-n",conda_env,"bash","-c",command]`, `cwd`=remote
  repo_dir, scrubbed env. **Timeout (Option B, revised after the modal 1.5.0 offline contract
  check):** a positive `self.timeout` is passed as `Sandbox.exec(timeout=N)` so modal enforces
  the wall-clock server-side and raises `modal.exception.ExecTimeoutError`, which `_sb_exec`
  catches and maps to the byte-identical `SubprocessResult(1, stderr=f"Timeout after
  {self.timeout} seconds")`. (modal 1.5's `ContainerProcess` has NO per-process kill, so the
  earlier client-side `threading.Timer` + process-group-kill design could not work.) **If
  `self.timeout is None`, the exec is unbounded** (mirrors local `communicate(timeout=None)`
  for the final test).
- `_check_workspace_integrity()` — checks the **remote** `.git` + target files; same error
  strings as the parent.
- `kill_running_process()` — modal 1.5's `ContainerProcess` has no per-process kill, so this
  tears the sandbox down (the only kill primitive), which also stops the in-flight eval;
  idempotent + **no-op when idle**. (Revised from "kill only the process group" — Option B.)
- `cleanup()` — terminate any lingering sandbox, then `super().cleanup()` (local git reset).
- **SIGINT/teardown safety (R1-safe):** the shared signal handler
  (`run_agent_benchmark.py`'s `_cleanup_on_signal`) calls `sys.exit(1)`, which runs `atexit`; the
  `ModalExecutor` `atexit` hook + the `_run_phase` finally terminate the sandbox. No change
  to the shared handler. Hard kills (SIGKILL/OOM/crash) that skip `atexit` are caught by the
  explicit sandbox max-lifetime `_sb_create` sets (`self.timeout + 1800s`, or a 24h cap for the
  unbounded final test) and, sooner, by the tagged-sandbox reaper (`provision.py::reap`) — §12.3.
- `.git` protection — during the shared `pre_test_val`→`final_test` sandbox, wrap the main
  command in remote `chmod 0555/0755 <repo>/.git` (try/finally) to mirror `_protect_git_dir`
  (for fresh-container search evals it is unnecessary).
- **Thin SDK adapter** `_sb_create/_sb_spawn/_sb_wait/_sb_write/_sb_read/_sb_read_bytes/
  _sb_terminate` (+ `_exec_timeout_exc`, module-level `modal_Sandbox_create`) — the only place
  that touches the Modal SDK; each call marked `# VERIFY`.

## 6. Provisioning (R2 — reproduce the exact local baseline)

Per task, build a Modal **Image** whose `workspace/<task>/` is content-identical to a local
`setup.py --task <T>` result, plus the conda env. The general mechanism — **run the real
`setup.TASKS["<task>"][0](args)` at build** — is correct, but three tasks need special
handling (below).

1. Base `continuumio/miniconda3`; **real `conda env create`** for the task's env(s)
   (`TASKS[<task>][1]`) by calling the REAL `setup.CONDA_ENVS[<env>]()` at build (no
   `CONDA_ENV_SPECS` table; see header note) — so remote `conda run -n <env>` is the identical
   binary/semantics to local. (privacy_meter env is special-cased — see below.)
2. Bake `ml_tasks/` into the image (small, versioned).
3. Re-root `setup.PROJECT_ROOT/WORKSPACE/ML_TASKS` onto the image FS and call
   `setup.TASKS["<task>"][0](args)` with an args namespace whose `skip_data=False` — reusing
   `clone_repo` (clone@pinned + commit `setup_files`) and `git_commit_data` verbatim,
   producing the post-setup working tree with in-repo data and any sibling `../data`.
4. **Record the built working tree-hash** (`git rev-parse HEAD^{tree}`, or a content hash of
   tracked files) in a per-task manifest; `ModalExecutor` asserts the remote tree-hash matches
   before the first eval. (Not the commit SHA — commits embed timestamps.)

**Per-task exceptions (must be handled, not assumed away):**
- **easyfsl** — its setup runs gated `gdown` + Kaggle + `predict_embeddings --device=cuda` at
  build. Provision on a **GPU builder with Kaggle/gdown secrets**, or **pre-bake the
  embeddings/model onto a Volume** and skip that branch.
- **privacy_meter** — its env step is `pip install -r {WORKSPACE/...}` (host path, repo must
  exist first) and its data step needs that env. Build order = **clone (`skip_data=True`) →
  create env → data (`skip_data=False`)** (see §0a item 4).
- **gcastle** — `repo_dir` is the inner `trustworthyAI/gcastle` with a **fresh `git init`**;
  reproduce that exact layout; the asserted tree-hash is recorded from the build.

**Data placement (three patterns, derived from each `setup_<task>`):** (a) in-repo committed
data (via `clone_repo` `setup_files` and/or `git_commit_data`) ships inside the baked repo;
(b) domainbed-style sibling data baked at `../data`; (c) genuinely external/huge/gated data on
a read-only Volume (+ build secrets).

**Pilots:** **(A) `Causality_causalml`** — TensorFlow, synthetic in-repo CSV, only
`split_config.json` setup-committed, no secrets; run it **on a GPU sandbox** to exercise
device-selection scrubbing; byte-diff `env` + stdout + `*_info.json` vs a local run. **(B)
`Privacy_opacus`** — a **stateful** task (torch + CIFAR) to validate the `pre_test_val →
final_test` checkpoint handoff in the shared sandbox. Only after both pass, generalize
task-by-task.

**`setup.py` is unchanged**: the provisioner imports it and calls the real
`setup.CONDA_ENVS[...]()` / `setup.TASKS[...][0](...)` at image build (re-rooting the module
globals), so the remote env/workspace is produced by the identical commands and local
`python setup.py` is byte-for-byte unaffected.

## 7. Launcher `launch_benchmark.py` (task-level parallelism only)

```bash
python launch_benchmark.py \
  --agent-config configs/agents/aide.yaml \
  --tasks all | Causality_causalml,Privacy_opacus \
  --model gpt-5.4 --provider OpenAI \
  --max-concurrency 16 \
  --output-dir results [--continue-on-failure] [key=value overrides...]
```

- One local subprocess per task running the **unmodified** `run_agent_benchmark.py
  --eval-backend modal ...`, bounded by `asyncio.Semaphore(max_concurrency)`; Modal schedules
  the GPU concurrency.
- **No seeds/repeats** — FML-bench has no external seed knob (task train seeds are hardcoded;
  no `val/test` command takes `--seed`). Each task runs once.
- Result layout unchanged; `launch_logs/<ts>/{<task>.{stdout,stderr,exit}, launch_status.json}`
  aggregates per-task status/metric/duration. Each task is an isolated subprocess + sandbox;
  one failure doesn't abort the rest. SIGINT broadcasts to children (existing handlers; each
  child's `ModalExecutor` tears down its own sandbox via §5).

## 8. Change set

**New:** `benchmark/executor_factory.py`, `benchmark/modal_executor.py`,
`modal_app/{__init__,images,provision,gpu_map}.py`, `launch_benchmark.py`, `docs/MODAL.md`
(runbook), `tests/test_executor_factory.py`.

**Changed (behavior-preserving for local):**
- 14 `BenchmarkExecutor(...)` call sites (2 per agent × 7) → `make_executor(...,
  eval_backend=eval_backend)` + a one-line `eval_backend =
  self.config.runtime_params.get("eval_backend","local")` read; import →
  `from benchmark.executor_factory import make_executor` (keep theaiscientist's separate
  `SubprocessResult` import).
- `run_agent_benchmark.py` — add `--eval-backend` (default `local`) + write into
  `runtime_params`.
- `setup.py` — **not changed** (the provisioner calls the real `setup` functions at image build).
- `benchmark/executor.py` — **not changed**; `benchmark/__init__.py` still exports
  `BenchmarkExecutor`.

## 9. Phased plan & verification gates

| Phase | Content | Verifiable here? |
|---|---|---|
| 1 | Factory + 14 call sites + `--eval-backend` (**no Modal**) | ✅ unit test: default returns `BenchmarkExecutor`; import works with `modal` absent (it is); 7 agents import; `--help` shows the flag. ⚠️ full local golden run needs `setup.py --task Causality_causalml` first (optional) |
| 2 | _(folded into provisioning — no `setup.py` change; build calls the real `setup` fns)_ | — |
| 3 | Modal SDK spike (confirm real API) | ❌ needs account → runbook |
| 4 | Provision pilot image (causalml, exact tree-hash) | ❌ needs account → runbook |
| 5 | `ModalExecutor` end-to-end on pilot A (causalml, GPU sandbox) + **byte-diff vs local** | ❌ needs account → runbook |
| 6 | **Stateful pilot B (`Privacy_opacus`)**: checkpoint-handoff + timeout/kill/integrity parity | ❌ needs account → runbook |
| 7 | Launcher (orchestration verifiable via mock backend) | ✅ mock-backend dry run; ❌ real run needs account |
| 8 | Generalize task-by-task (handle easyfsl/privacy_meter/gcastle/big-asset); full matrix + docs | ❌ needs account |

## 10. R1 / R2 / R3 compliance self-audit

- **R1:** factory returns the identical object+args when local; lazy modal import (works with
  `modal` absent); `--eval-backend` defaults to `local` and only adds a `runtime_params` key;
  `executor.py` unchanged; `setup.py` additive; the SIGINT/atexit teardown lives entirely in
  `ModalExecutor` (shared handler untouched). Regression checks: factory unit test,
  modal-absent import test, optional local golden diff on the pilot.
- **R2:** remote reproduces the exact post-setup **working tree** (verified by tree-hash) +
  per-task data placement; `ml_tasks` pushed fresh each run; metric parsing runs locally on
  pulled JSON; exec contract byte-identical; `timeout=None` unbounded; real conda. Residual
  fidelity assumption = per-eval hermeticity (true by code for search-loop evals), **gated by
  the pilot byte-diff**, with the **stateful checkpoint-handoff gate (opacus)** and the
  persistent-working-dir fallback for any task that fails.
- **R3:** claims re-grounded against `setup.py` (`TASKS`, `clone_repo`, `git_commit_data`,
  `CONDA_ENVS`, the easyfsl/privacy_meter/gcastle specifics, `runner.py`'s required-keys list); the v1
  aider/seeds/`skip_conda`/bare-commit and the v2 CPU-pilot/byte-identical/exact-HEAD
  overstatements are removed.

## 11. What the author does to run it (later, with a Modal account)

1. `pip install modal` + `modal token new` on the orchestrator host.
2. Locally: `setup.py --task <T> --skip-envs --skip-data` (enough for `CodeEditor` to edit
   target files; envs/data live remotely — exact flag combo confirmed in the runbook:
   `--skip-data` still leaves `clone_repo`'s committed `setup_files`).
3. `modal run modal_app/provision.py::build --task <T>` to build the image (+ data Volume /
   build secrets for easyfsl/officehome/open_unlearning).
4. `python run_agent_benchmark.py --eval-backend modal ...` for one task, or
   `python launch_benchmark.py --tasks ...` for many (the launcher always uses the Modal
   backend for its children).
5. Follow `docs/MODAL.md` for the byte-diff fidelity check, the opacus checkpoint-handoff
   gate, and the orphan-sandbox reaper.

## 12. Risks needing live verification (no Modal account here)

1. **Exec/env parity (R2 core):** confirm Modal's exec does not inject
   `CUDA_VISIBLE_DEVICES`/login-shell/extra CUDA paths that perturb device selection or
   nondeterministic kernels — the byte-diff of `env` + stdout + `*_info.json`, **run on a GPU
   sandbox** (the TF pilot grabs a GPU if visible), is the key check.
2. **`timeout=None` mapping:** confirm the chosen Modal exec API treats `None` as unbounded
   (not "kill immediately") on a long-running task.
3. **Sandbox leak:** graceful SIGINT is handled by the `ModalExecutor` `atexit`/finally
   teardown (§5, R1-safe). Hard kills (SIGKILL/OOM/host-crash) skip `atexit` → the explicit
   sandbox max-lifetime `_sb_create` sets (`self.timeout + 1800s`, or a 24h cap for the
   unbounded final test) auto-terminates it server-side; the tagged-sandbox reaper
   (`provision.py::reap`) is the earlier backstop.
4. **Hermeticity / checkpoint handoff:** search-loop hermeticity holds by code; the
   shared-sandbox checkpoint handoff is validated by the **opacus** stateful gate before
   generalization; non-hermetic tasks fall back to the persistent working-dir Volume.
5. **Per-task generalization:** **easyfsl** (GPU builder + Kaggle/gdown secrets, or pre-baked
   assets), **privacy_meter** (clone-before-env + requirements-path substitution), **gcastle**
   (fresh-init inner repo + `../../../../` depth), and big-asset/gated tasks
   (officehome/open_unlearning) each need the "reproduce exact local tree + data placement +
   secrets" validation, confirmable only once images build.

## 13. Out of scope (flagged, untouched)

Vestigial aider references (`benchmark/utils.py:155` dead `extract_token_usage_from_aider`,
`setup.py` leftover `pip install aider-chat`) — unrelated to this work.
