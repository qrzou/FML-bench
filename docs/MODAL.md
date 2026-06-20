# FML-bench × Modal eval backend — operator runbook

> **Status: the remote (Modal) path is verified OFFLINE; live runtime behavior is still
> unconfirmed.** Three offline test layers (§0.2) now pass, including the SDK *contract*
> (method names / kwargs / signatures + the process-handle & timeout API) checked against
> **`modal==1.5.0`** with no account. That offline pass already retired the
> "wrong-method-name / wrong-kwarg" slice of the `# VERIFY` risk and caught one real bug
> (the timeout fix below). What remains unconfirmed is the **runtime behavior** of each
> `# VERIFY` point (does `exec` actually capture stdout / enforce `timeout=` / raise
> `ExecTimeoutError`; does the image build reproduce the tree-hash; is `"A100"` accepted;
> volumes; reaper) and the §5–§6 **fidelity gates**. Every Modal SDK call lives behind
> `ModalExecutor`'s `_sb_*` adapter and the `modal_app/` package, each uncertain API point
> carrying a trailing `# VERIFY` comment. Until §3–§6 pass on a real account, treat the
> Modal backend's live behavior as *unconfirmed*.
>
> The **local** path (no `--eval-backend modal`) is unaffected by any of this: the `modal`
> package is not imported unless `eval_backend == "modal"`, so the default README workflow
> behaves exactly as before (constraint R1).
>
> See `docs/MODAL_DESIGN.md` (v2.1) for the full design and rationale. Section numbers in
> this runbook (e.g. §5, §12.3) refer to that design doc.

---

## 0. Prerequisites & one-time setup

### 0.1 Install the Modal SDK and authenticate

The `modal` package is **not** installed in the repo's environment by design (the local
path must run without it). Install it into whatever Python environment will run the
orchestrator (`run_agent_benchmark.py` / `launch_benchmark.py`):

```bash
pip install modal
modal token new
```

`modal token new` opens a browser to create/authorize an API token and writes it to
`~/.modal.toml`. Expected output ends with something like:

```
Launching login page in your browser window...
Web authentication finished successfully!
Token written to ~/.modal.toml in profile <your-workspace>.
```

Confirm the SDK is usable and the credentials resolve:

```bash
python -c "import modal; print(modal.__version__)"
modal profile current        # prints your active workspace name
```

If `modal profile current` errors with an auth message, re-run `modal token new`.

> The exact `modal` version matters: the `# VERIFY` markers were written against the SDK's
> documented surface, not a pinned version. Record the version you install here and use it
> for every step below so the verification stays consistent.

### 0.2 Offline test suite — run this BEFORE spending credits

Three stdlib-`unittest` layers (no pytest) live in `tests/`. They need **no Modal account
and cost nothing**; run them first so logic bugs are caught locally before a single
GPU-second is billed. (This suite already caught one real bug — see the timeout note in §1.)

**Layers 1 & 2 + the factory test — run in the repo's normal (modal-absent) env:**

```bash
python tests/test_executor_factory.py             # backend dispatch (local vs modal)
python tests/test_local_path_unaffected.py        # R1: local path byte-unaffected, no modal import
python tests/test_modal_executor_orchestration.py # ModalExecutor end-to-end over a FakeSandbox
```

- **Layer 1** (`test_local_path_unaffected`) guards R1: the 5 behavior-defining files
  (`benchmark/executor.py`, `agents/{base,code_editor,llm}.py`, `setup.py`) are byte-identical
  to the branch's fork point; the factory's local path never imports `modal` (proven by
  banning it via a meta-path finder in a subprocess); all 7 agents route through
  `make_executor(eval_backend=...)`.
- **Layer 2** (`test_modal_executor_orchestration`) drives the *real* `ModalExecutor` against
  a `FakeSandbox` that maps the remote root onto a local temp dir and runs commands with real
  bash/git (only the Modal boundary is faked). It asserts the exact conda argv + verbatim
  command, byte-identical input push / result pull, the shared `pre_test_val → final_test`
  checkpoint handoff, teardown on every exit path, the tree-hash-mismatch → failure-dict
  contract, and the server-side `timeout=`. It proves *our orchestration*, NOT Modal's behavior.

**Layer 3 — the SDK contract — run in a SEPARATE throwaway venv:**

```bash
python -m venv /tmp/modalvenv
/tmp/modalvenv/bin/pip install modal           # no account / no `modal token new` needed
/tmp/modalvenv/bin/python tests/test_modal_api_contract.py
```

Layer 3 introspects the installed SDK: `Sandbox.create/exec/filesystem/set_tags/get_tags/
terminate/list`, the `Sandbox.filesystem` namespace (`read_bytes/read_text/write_bytes/
make_directory`, incl. write_bytes' `(data, remote_path)` arg order — `Sandbox.open` was
deprecated 2026-03-09), the `Image` builder chain, `add_local_dir(remote_path=,copy=,ignore=)`,
`run_function(args=,gpu=)`, the `ExecTimeoutError` import path, the `ContainerProcess`
handle (`wait/poll/returncode/stdout/stderr` — and, deliberately, the ABSENCE of a
per-process kill), and the real `modal_app` constants / `gpu_map` A100 defaults. It SKIPS
if `modal` is not importable.

> **Do NOT `pip install modal` into the env that runs the other layers.**
> `test_executor_factory.py` asserts modal is *absent* (that absence is part of the R1
> contract), so it would start failing if modal were importable there. Record the version
> Layer 3 printed (`[layer3] tested against modal==X.Y.Z`) here and in §0.1; re-run Layer 3
> after any modal upgrade, since the SDK surface can drift.

---

## 1. Read the source first — enumerate the `# VERIFY` points

Before spending account credits, list every Modal API the backend actually calls so the
spike in §3 covers exactly those — no more, no less. The Modal SDK is touched in only two
places:

```bash
# the thin adapter (the ONLY methods that call the Modal SDK)
grep -n "VERIFY\|_sb_\|modal\." benchmark/modal_executor.py

# the provisioning / image-build package
grep -rn "VERIFY\|modal\." modal_app/
```

Derived from the design (§5 "Thin SDK adapter" + §6 "Provisioning"), the API points the
spike in §3 must confirm are:

| Adapter / build call | Modal SDK surface to confirm | Why it is `# VERIFY` |
|---|---|---|
| `_sb_create` | `Sandbox` creation from a built `Image` on a GPU type | exact constructor/`create` signature, how the image + gpu + volume are attached |
| `_sb_exec(command)` | sandbox "exec" / process API | argv passthrough, `cwd`, env override, **stdout/stderr capture**, returncode |
| `_sb_spawn` timeout | `Sandbox.exec(timeout=N)` server-side wall-clock → `ExecTimeoutError` | **[contract confirmed offline, modal 1.5.0]** `exec` accepts `timeout=`; live-confirm it actually kills the proc and raises `ExecTimeoutError` from `.wait()` so `_sb_exec` maps it to `Timeout after <N> seconds` |
| `_sb_exec(timeout=None)` | unbounded exec semantics | confirm `None` means "run to completion", **not** "kill immediately" (final-test path) |
| `_sb_write` | write bytes/file into the sandbox FS | path semantics, overwrite behavior |
| `_sb_read` | read a file out of the sandbox FS | returns bytes; behavior when the file is missing |
| `kill_running_process` | no per-process kill in modal 1.5 → `Sandbox.terminate` | **[contract confirmed offline]** `ContainerProcess` has no `terminate/kill/signal`, so an external signal tears the sandbox down (which stops the in-flight eval) |
| `_sb_terminate` | terminate the sandbox | idempotent; safe from `atexit` + `finally` |
| image build | `Image` conda env build via `run_function` (runs the real `setup.*` fns at build) | how a Python builder fn runs in the image build, on a **GPU builder** when needed, with **secrets** |
| `Volume` | per-task data / persistent-working-dir volume | create/mount RO and RW, commit/reload semantics |
| gpu strings | `gpu="A100"` (the only GPU all GPU-tasks use) + no `gpu=` for CPU-only tasks | that `"A100"` is the right Modal label and that omitting `gpu=` gives a CPU sandbox |
| reaper | `Sandbox.set_tags` / `Sandbox.list(tags=)` / `get_tags()` reads the `created_at` TAG we stamp at create (modal 1.5 `Sandbox` has no `created_at` attr); a sandbox max-lifetime IS set (§7) | the tag / list-by-tag / get_tags API used by `provision.py::reap` |

> If the table above and the actual `# VERIFY` comments in the source disagree, **the
> source wins** — re-run the two `grep`s and reconcile before the spike. (At the time this
> runbook was written, `benchmark/modal_executor.py` and `modal_app/` were being authored in
> parallel; this table is the design-level checklist, the comments are the ground truth.)

---

## 2. Local prep (no Modal needed) — enough for `CodeEditor` to edit target files

The agent loop runs locally; only experiment *execution* goes to Modal. The local checkout
still needs each task's **target files** present on disk so `CodeEditor` can snapshot and
edit them. Envs and datasets live on the remote image, so skip them:

```bash
python setup.py --task Causality_causalml --skip-envs --skip-data
```

Expected: it clones the task repo at its pinned commit and copies in the setup files
(`clone_repo` still runs and commits `setup_files` even with `--skip-data`), but skips
conda env creation and dataset download. Verify the target files exist:

```bash
ls workspace/Causality_causalml/                 # repo present
grep -nE 'repo_dir|target_files|val_command' benchmark/runner.py   # how config is built
```

> `--skip-data` leaves `clone_repo`'s committed `setup_files` in place — that is exactly
> what `CodeEditor` needs. It does **not** download datasets or build the conda env, both of
> which are reproduced on the Modal image (§4). If a task's target files live only behind a
> data step, fall back to `--skip-envs` alone (slower, but local-complete).

For the stateful pilot, do the same:

```bash
python setup.py --task Privacy_opacus --skip-envs --skip-data
```

---

## 3. Modal SDK spike — confirm the `# VERIFY` API points

Spend the **minimum** credits to confirm each row of §1's table against the live SDK, in
isolation, before wiring it into the executor. Run a tiny throwaway script (do **not** put
this in the repo). The goal is to turn each `# VERIFY` into "confirmed for SDK version X".

Confirm, one at a time:

1. **Sandbox create + exec + capture + returncode.** Create a sandbox from a trivial image,
   exec `bash -c 'echo hi; exit 7'`, assert stdout is `hi\n` and returncode is `7`. This
   pins how `_sb_exec` must build its `SubprocessResult(returncode, stdout, stderr)`.
2. **Exec `cwd` + env override.** Exec `bash -c 'pwd; echo $CUDA_VISIBLE_DEVICES'` with a
   chosen working dir and a scrubbed env; confirm Modal does **not** silently inject
   `CUDA_VISIBLE_DEVICES`, a login shell, or extra CUDA paths (Risk §12.1). Whatever the
   spike shows here is the env-scrub the executor must apply to keep device selection
   identical to local.
3. **Timeout semantics (Option B).** Exec a `sleep 60` via `Sandbox.exec(timeout=2)`; confirm
   modal kills it server-side and that `.wait()` (or the stream read) raises
   `modal.exception.ExecTimeoutError`, so `_sb_exec` maps it to the local-identical
   `SubprocessResult(1, stderr="Timeout after <N> seconds")`. (The `timeout=` kwarg and the
   exception import path are already confirmed offline by Layer 3 — this confirms the runtime
   raise.)
4. **`timeout=None` is unbounded.** Exec a `sleep 5` with the timeout disabled; confirm it
   runs to completion (returncode 0), i.e. `None` is *not* "kill immediately" (Risk §12.2).
   This is the final-test path (`base.py` sets `executor.timeout = None`).
5. **Kill == sandbox terminate (no per-process kill in modal 1.5).** `ContainerProcess` has no
   `terminate/kill/signal` (confirmed offline by Layer 3), so `kill_running_process` and the
   timeout both rely on `Sandbox.terminate` / server-side `timeout=`. Confirm that terminating
   the sandbox while `bash -c 'sleep 300 & sleep 300'` runs kills **all** children (no orphan
   survives the sandbox) — the parity goal vs the local `os.killpg(..., SIGKILL)`.
6. **File I/O.** `_sb_write` a file in, exec `cat` it, then `_sb_read` it back; confirm bytes
   round-trip and overwrite works (used to push `target_files` + `ml_tasks/<task>/`).
7. **Image conda build via `run_function`.** Build a throwaway image that runs a Python fn
   importing nothing exotic; confirm a builder fn can run during image build, on a **GPU
   builder** if requested, with a **secret** attached.
8. **Volume RO/RW.** Create a Volume, write to it from one sandbox, mount it read-only in
   another, confirm visibility (fallback / gated-data path).
9. **GPU strings.** All GPU-using tasks use `gpu="A100"` (`modal_app/gpu_map.py`); confirm
   `"A100"` is accepted and `nvidia-smi` runs, and that a CPU-only task (no `gpu=`) starts.
10. **Reaper.** Confirm `Sandbox.set_tags`, `Sandbox.list(tags=...)`, and `get_tags()` work so
    `provision.py::reap` can list/terminate ONLY this app's tagged sandboxes by age. modal 1.5's
    `Sandbox` has NO `created_at` attribute, so `_sb_create` stamps a `created_at` epoch into the
    tags and `reap` reads it back via `get_tags()`. A sandbox max-lifetime IS set at create (§7).

Tear down everything the spike created:

```bash
modal app list          # confirm no orphan apps/sandboxes from the spike
```

After the spike, update every confirmed `# VERIFY` comment in
`benchmark/modal_executor.py` and `modal_app/` to note the SDK version it was confirmed
against. **Anything not confirmed here stays `# VERIFY` and is not trusted.**

---

## 4. Provision the pilot image (causalml) and assert the tree-hash

The image build re-roots `setup.PROJECT_ROOT` / `WORKSPACE` / `ML_TASKS` onto the image FS
and **calls the real `setup.*` functions** — the identical commands as a local
`setup.py --task Causality_causalml` (env create via `setup.CONDA_ENVS["causalml"]()`, then
clone+commit+data via `setup.TASKS["Causality_causalml"][0](args)` with `skip_data=False`).
Order is **env-then-task** for causalml.

```bash
modal run modal_app/provision.py::build --task Causality_causalml
```

Expected: the build clones the repo at its pinned commit, copies `split_config.json`,
creates the `causalml` conda env, and records the built **git working-tree hash**
(`git rev-parse HEAD^{tree}`) into a per-task manifest. **Record the commit SHA is NOT
used** — commits embed timestamps and differ across machines; only the tree content is
reproducible (§ design row 6).

Assert the local post-setup tree-hash matches what the build recorded, so a future eval
can fail fast on drift. Locally (after a full local `setup.py --task Causality_causalml`):

```bash
git -C workspace/Causality_causalml rev-parse 'HEAD^{tree}'
# compare against the value the build wrote into the manifest
grep -n "tree" modal_app/<wherever the manifest is written>     # confirm the recorded hash
```

`ModalExecutor` asserts this match before the first eval; if it mismatches, the remote
baseline is not byte-equivalent to local and you must rebuild — **do not** proceed to the
fidelity gate with a mismatched tree.

---

## 5. FIDELITY GATE — Modal vs local byte-diff (pilot A: causalml)

This is the hard check that the Modal path is faithful (R2). causalml uses TensorFlow and
grabs a GPU if one is visible. NOTE: `modal_app/gpu_map.py` now maps `Causality_causalml ->
None` (CPU-only by project choice), so by default this eval runs on a **CPU sandbox** — which
verifies the full push/exec/pull pipeline but NOT GPU device-selection scrubbing. To exercise
the GPU-parity check, temporarily map `Causality_causalml` to a GPU (e.g. `"A100"`) in
`gpu_map.py` and run both the local and Modal sides on a GPU, or use another GPU-mapped task.

**Local reference run** (full local setup first: `python setup.py --task Causality_causalml`):

```bash
python run_agent_benchmark.py --eval-backend local  <the rest of your normal args>
```

**Modal run** (identical args, only the backend differs):

```bash
python run_agent_benchmark.py --eval-backend modal  <the same args>
# or equivalently: FMLBENCH_EVAL_BACKEND=modal python run_agent_benchmark.py <same args>
```

Then byte-diff the three things that must be identical:

1. **`env`** captured inside the exec (local subprocess env vs the scrubbed remote exec
   env) — especially `CUDA_VISIBLE_DEVICES` and any CUDA/login-shell injections.
2. **stdout/stderr** of the train command.
3. **The metric-bearing `results_tmp/*_info.json`** — this is the scoped fidelity claim
   (design row 10): only `*_info.json` must match byte-for-byte. `code_backup/` is **not**
   compared (it is `--save-code-backup`-only and warned/no-op'd in Modal mode; scoring never
   reads it).

```bash
# example: diff the metric JSON pulled by the Modal run against the local run's
diff <(python -m json.tool workspace/Causality_causalml/results_tmp/val_info.json) \
     <(python -m json.tool /path/to/local-run/results_tmp/val_info.json)
# expected: no output (identical)
```

> With a deterministic LLM and identical hardware/software, val/test metrics should match.
> If they diverge, the most likely cause is exec-env contamination (re-check §3 step 2) or a
> non-hermetic task (see §6 and the persistent-working-dir Volume fallback in design §4).
> **The pilot passing here is what makes the search-loop hermeticity claim a verified fact
> rather than an assumption.**

---

## 6. STATEFUL CHECKPOINT-HANDOFF GATE (pilot B: Privacy_opacus)

causalml's `final_test` reads no state from `pre_test_val`, so it validates single-eval
parity but **not** the shared-sandbox `pre_test_val → final_test` checkpoint handoff that
~8 stateful tasks rely on. `Privacy_opacus` (torch + CIFAR) writes a checkpoint in
`pre_test_val` that `final_test` reads back — on the **same** sandbox filesystem (the two
phases share one kept sandbox, terminated only after `final_test`; design §4 table).

Provision and run:

```bash
modal run modal_app/provision.py::build --task Privacy_opacus
python run_agent_benchmark.py --eval-backend modal  <opacus args, final-test path>
```

Confirm:

1. The sandbox created for `run_id == "pre_test_val"` is **kept** (not terminated) and
   **reused** for `run_id == "final_test"`; it is terminated only after `final_test`.
2. `final_test` reads the checkpoint `pre_test_val` wrote (the test metric is sane, not a
   cold-start value). Byte-diff `test_info.json` against a local opacus run as in §5.
3. **`.git` protection across the shared sandbox**: during the shared `pre_test_val →
   final_test` window the remote `.git` is `chmod 0555` then restored (mirrors local
   `_protect_git_dir`), so the train command cannot corrupt the baked repo.
4. **timeout / kill / integrity parity**: a timeout produces the byte-identical
   `SubprocessResult(1, stderr="Timeout after <N> seconds")`; an integrity-check failure
   produces the same error strings as the local `_check_workspace_integrity`.

> Only after **both** §5 (causalml) and §6 (opacus) pass should you generalize task-by-task
> (§8). A failure here means a stateful task needs the persistent-working-dir Volume
> fallback (design §4) rather than the fresh-container default.

---

## 7. Orphan-sandbox reaper

Sandboxes cost GPU money for as long as they live, so teardown must be robust to crashes.
There are three layers (design §5 + §12.3):

1. **Graceful SIGINT** — the shared signal handler (`run_agent_benchmark.py`) calls
   `sys.exit(1)`, which runs `atexit`; `ModalExecutor`'s `atexit` hook + the `_run_phase`
   try/finally terminate `self._sb`. (The shared handler is **not** modified — R1-safe.)
   Verify by Ctrl-C'ing a Modal run mid-eval and confirming the sandbox disappears:

   ```bash
   modal app list      # the run's sandbox should be gone shortly after Ctrl-C
   ```

2. **Server-side max-lifetime IS set** — `Sandbox.create`'s default `timeout` is 300s (a
   5-min cap, NOT unbounded), so `ModalExecutor._sb_create` passes an explicit lifetime:
   `self.timeout + 1800s` for a bounded eval, or a 24h cap for the unbounded final test
   (override via the `modal_sandbox_max_lifetime_seconds` config). This both prevents a long
   eval from being killed at 5 min AND auto-terminates a sandbox leaked by a **hard** kill
   (SIGKILL / OOM / host crash) that skips `atexit`. The reaper (below) cleans up sooner and
   handles a sandbox whose create-time tag is unreadable. (The 24h cap value and the exact
   mid-exec termination surface are not yet live-verified — lower the config if Modal rejects it.)

3. **Tagged reaper** — every sandbox is tagged at creation with the static tag
   `{"fml_bench": "fml-bench-eval", "task": "<task>"}` (NOT a per-run id). The `reap`
   entrypoint sweeps ONLY sandboxes carrying that app tag and older than `--max-age-seconds`,
   and is **DRY-RUN by default** (prints what it would terminate); pass `--force` to actually
   terminate:

   ```bash
   modal run modal_app/provision.py::reap --max-age-seconds 7200            # dry-run
   modal run modal_app/provision.py::reap --max-age-seconds 7200 --force    # terminate
   modal app list                                                           # cross-check
   ```

> `Sandbox.set_tags` / `Sandbox.list(tags=)` are live-confirmed (the smoke's 0-orphan check
> used them). modal 1.5's `Sandbox` has **no `created_at` attribute**, so `_sb_create` stamps a
> `created_at` epoch into the tags and `reap` reads age back via `get_tags()`. That age-based
> reap path is not yet live-exercised, so after an abnormal exit still **run `reap` and check
> `modal app list`** to confirm nothing leaked.

---

## 7a. Mid-eval connection drops — automatic reconnect & recovery

A **transient** Modal connection error *during* an eval (e.g. "Could not connect to the Modal
server") does NOT waste the run: `ModalExecutor` catches it, reconnects to the still-running
sandbox by id via `Sandbox.from_id`, and **polls for the eval's exit-code sentinel** (written by
the remote command as its last action) — recovering the eval's real result **without re-running
it**. In the log:

```
Connection lost during val_command; reconnecting and polling for the result (no re-run): ...
Reconnected; eval finished (rc=0) — recovered without re-running.
```

- **Scope:** only the narrow transient `ConnectionError` triggers this (auth / not-found / a
  nonzero eval exit are NOT retried — they fail fast). Idempotent control ops (sandbox create,
  manifest read, result pull) are separately retried with exponential backoff for up to
  `modal_retry_budget_seconds` (default 1800s).
- **Poll deadline:** `self.timeout` seconds measured from reconnect (the per-eval wall-clock),
  or a 7200s cap when `self.timeout is None` (the unbounded final test).
- **Fidelity:** the recovered returncode is the eval's TRUE exit code (read from the rc
  sentinel), so a nonzero-exit eval is reported as a failure exactly like the no-drop path —
  not a false success from mere result-file presence.

---

## 8. Per-task generalization notes

After the two pilots pass, generalize one task at a time. Most tasks follow the plain
**env-then-task** recipe (build the conda env from `CONDA_ENVS[<env>]`, then run
`TASKS[<task>][0](args)` with `skip_data=False`, record the tree-hash, run §5's byte-diff).
The following tasks are **not** plain and must be handled explicitly:

### easyfsl (`Data_Efficiency_easyfsl`) — GPU builder + secrets
`setup_data_efficiency_easyfsl` runs, **at build time**, a gated `gdown` (`--id …`), a Kaggle
download (`curl …/kaggle.com/api/v1/datasets/download/…`), and
`predict_embeddings … --device=cuda`. So the image must be built on a **GPU builder** with
**Kaggle + gdown secrets** attached, **or** the embeddings/model must be **pre-baked onto a
Volume** and the build's download branch skipped. It is *not* covered by the "just call the
setup fn on a CPU builder" recipe.

> **R2 caveat (audit M-1):** `setup_data_efficiency_easyfsl` does NOT `git_commit_data` the
> generated embeddings/model (they are `.gitignore`d), so they sit OUTSIDE the working-tree
> hash and are NOT pushed per-eval — unlike every other task, whose data is committed. The
> tree-hash assertion therefore does NOT verify easyfsl's baked embeddings. Before trusting
> easyfsl R2: either `git_commit_data` the embeddings+model into the baked tree, push them
> per-eval, or record+assert a content hash of those dirs in the manifest; at minimum validate
> one easyfsl eval live (the build-time GPU forward pass is not bit-guaranteed across GPUs).

### privacy_meter (`Privacy_privacymeter`) — clone BEFORE env
Its env step is `pip install -r {WORKSPACE/Privacy_privacymeter/ml_privacy_meter/requirements.txt}`
— a host absolute path that requires the repo to exist first. So the build order is
**task-then-env** (clone the repo, substitute the requirements path onto the image FS, then
create the env), the reverse of every other task. It is excluded from any static env table.

### gcastle (`Causality_gcastle`) — inner fresh-init repo + deep `repo_dir`
`repo_dir` is the **inner** `trustworthyAI/gcastle`: a sparse-checkout of the outer repo at
`58abc35`, then `algorithm.py` copied in, then an inner **fresh `git init` single commit**.
Reproduce that exact layout at build and record the tree-hash from the build (not from a
pinned upstream commit). The deeper path also means `ModalExecutor` must derive the remote
repo path preserving the directory depth (the inner repo is several levels down).

### officehome / open_unlearning — gated / large assets
`setup_generalization_officehome` downloads a ~2.4GB OfficeHome zip via `gdown` at build.
`setup_unlearning` (`Unlearning_open_unlearning`) only **clones** the repo at setup; its
model/dataset are fetched from the **HF Hub at EVAL time** (`from_pretrained` in
`train_eval_baseline.py`), so the eval sandbox must carry the relevant HF env vars
(`HF_TOKEN`/`HF_HOME`/`HF_ENDPOINT`/proxy) — `ModalExecutor._scrubbed_env` inherits them from
the orchestrator (see the env-parity note). Put genuinely external/huge/gated data on a
**read-only Volume** (+ secrets) rather than re-fetching every run. Validate each with the §5
byte-diff once its image builds.

---

## 9. Honesty checklist (what is and is not verified)

- ✅ **Local path** (`--eval-backend local`, the default): unchanged, runs with `modal`
  absent. This is the only path verified during implementation.
- ❌ **Remote path** (`--eval-backend modal`): every step in §3–§8 is **unverified** until
  run on a live Modal account. The `# VERIFY` comments in `benchmark/modal_executor.py` and
  `modal_app/` mark exactly what is unconfirmed.
- The fidelity claim is **scoped**: only `env`, stdout/stderr, and the metric-bearing
  `*_info.json` are asserted byte-identical (not `code_backup/`, not checkpoints).
- Per-eval **hermeticity** of search-loop evals is true *by code* but is only a *verified*
  fact once §5 (causalml byte-diff) passes; the **checkpoint handoff** is only verified once
  §6 (opacus) passes. Any task that fails either gate uses the persistent-working-dir Volume
  fallback (design §4), not the fresh-container default.

When in doubt, re-run the two `grep`s in §1 and trust the source comments over this prose.
