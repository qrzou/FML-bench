"""
ModalExecutor: remote-GPU eval backend for FML-bench.

A drop-in subclass of ``BenchmarkExecutor`` that keeps the entire agent loop
(search, CodeEditor edits, metric parsing, result-dir writing) running locally
and dispatches ONLY the experiment-execution step (the val/test command) to an
ephemeral Modal GPU sandbox built from the exact baked post-`setup.py` baseline.

Selected only when ``--eval-backend modal`` (via ``benchmark.executor_factory``);
``modal`` is never imported on the local path. ALL Modal SDK calls live behind
the ``_sb_*`` adapter methods and ``modal_app``, each marked ``# VERIFY`` —
none can be live-verified here (no Modal account).

Execution & state model (docs/MODAL_DESIGN.md §4-5):
  - Stateless per eval: each search-loop ``run_val`` creates a fresh sandbox
    (== exact baked baseline), pushes the current target files + the whole
    local ``ml_tasks/<task>/``, runs the command, pulls the metric JSON, and
    tears the sandbox down.
  - The final-test pair ``pre_test_val`` -> ``final_test`` SHARES one sandbox so
    the checkpoint trained by pre_test_val is read by final_test on the same FS.
  - ``_run_phase`` is wrapped in try/finally so the sandbox is torn down even on
    interruption; an ``atexit`` hook is the SIGINT backstop (the shared signal
    handler calls ``sys.exit`` -> atexit). modal 1.5's ContainerProcess has no
    per-process kill primitive, so ``kill_running_process`` tears the sandbox
    down (the only kill available), which also stops the in-flight command.

The parent's ``_collect_results`` / ``_extract_primary_metric`` /
``_filter_results`` / ``_save_bug_execution_record`` are reused unchanged: the
metric JSON is pulled to the LOCAL ``repo_dir/results_tmp/`` before they run.
"""
from __future__ import annotations

import atexit
import json
import os
import os.path as osp
import shutil
from datetime import datetime
from typing import Optional

from benchmark.executor import (
    TEST_OUTPUT,
    VAL_OUTPUT,
    BenchmarkExecutor,
    SubprocessResult,
)


class ModalExecutor(BenchmarkExecutor):
    """Run val/test commands in an ephemeral Modal GPU sandbox (R2 fidelity)."""

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Modal backend supports only new-format configs (single val/test
        # command string). Old-format `execute_commands` is refused clearly.
        if "val_command" not in self.config:
            raise ValueError(
                "ModalExecutor (eval_backend='modal') requires a new-format "
                "config with 'val_command'; old-format 'execute_commands' is "
                "not supported on the Modal backend."
            )
        # Task name == benchmark_name == ml_tasks/<task>/ dir name.
        self.task = self.benchmark_name
        # Remote repo_dir: reproduce the local layout with the ORIGINAL
        # (non-suffixed) task dir so repo_dir depth — and the relative
        # `cp ../../../ml_tasks/<task>/...` paths in val_command — match local.
        self._remote_repo_dir = self._derive_remote_repo_dir()
        self._sb = None              # the live Modal sandbox handle (or None)
        self._tree_hash_checked = False
        # SIGINT backstop: shared handler calls sys.exit -> atexit runs this.
        atexit.register(self._atexit_terminate)

    # ------------------------------------------------------------------
    # Remote-path derivation
    # ------------------------------------------------------------------

    def _derive_remote_repo_dir(self) -> str:
        """Map the local (suffixed) repo_dir onto the remote baked layout.

        Local repo_dir == "workspace/<task>__<suffix>/<repo_subpath>" (the
        runner rewrites it). The remote baked tree uses the original task dir
        "workspace/<task>/<repo_subpath>" under REMOTE_ROOT, preserving depth.
        """
        from modal_app import REMOTE_ROOT  # lazy: pulls in modal only on modal path
        parts = self.repo_dir.split(os.sep)
        # parts[0] == "workspace", parts[1] == "<task>__<suffix>", rest == repo_subpath
        repo_subpath = os.sep.join(parts[2:]) if len(parts) > 2 else ""
        return "/".join([REMOTE_ROOT, "workspace", self.task, repo_subpath]).rstrip("/")

    @property
    def _remote_root(self) -> str:
        from modal_app import REMOTE_ROOT
        return REMOTE_ROOT

    # ------------------------------------------------------------------
    # Workspace setup (local result dir only; no sandbox yet)
    # ------------------------------------------------------------------

    def setup_workspace(self) -> str:
        """Build the local result dir (parent behavior); create NO sandbox yet."""
        return super().setup_workspace()

    # ------------------------------------------------------------------
    # Phase execution (wholesale override of the parent _run_phase)
    # ------------------------------------------------------------------

    def run_val(self, run_id) -> dict:
        return self._run_phase(run_id, "val_command", VAL_OUTPUT)

    def run_test(self, run_id) -> dict:
        return self._run_phase(run_id, "test_command", TEST_OUTPUT, backup=False)

    def _run_phase(self, run_id, command_key: str, output_path: str,
                   backup: bool = True) -> dict:
        """Remote mirror of BenchmarkExecutor._run_phase (same return shapes).

        Sequence: resolve command -> local clean results_tmp + make
        execution_<ts>/ -> ensure sandbox (create + assert tree-hash if needed)
        -> push target files + ml_tasks/<task>/ -> remote rm -rf results_tmp ->
        run via _sb_exec (with remote .git protection) -> remote integrity check
        -> on failure: save bug record + teardown (unless pre_test_val) + return
        parent-shaped failure dict -> pull results_tmp/<phase>_info.json to local
        repo_dir/results_tmp/ -> teardown (unless pre_test_val) ->
        super()._collect_results(...). Wrapped in try/finally for teardown.
        """
        # 0. Resolve command (new-format only; asserted in __init__).
        command = self.config.get(command_key)
        if not command:
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": f"No '{command_key}' specified in config (Modal backend)",
            }

        keep_sandbox = (run_id == "pre_test_val")  # share sandbox with final_test

        try:
            # 1. Clean LOCAL results_tmp/ if it exists (so a stale pull can't
            #    masquerade as fresh results).
            results_tmp_dir = osp.join(osp.abspath(self.repo_dir), "results_tmp")
            if osp.exists(results_tmp_dir):
                print(f"Cleaning existing results_tmp/ directory: {results_tmp_dir}")
                shutil.rmtree(results_tmp_dir)

            # 2. Create run and execution directories (local; identical to parent).
            execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
            execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
            if osp.exists(execution_dir):
                print(f"Warning: Removing existing execution directory: {execution_dir}")
                shutil.rmtree(execution_dir)
            os.makedirs(execution_dir, exist_ok=True)

            # 3. code_backup is warned/no-op'd in Modal mode: _backup_files reads
            #    the LOCAL git-changed tree, which never runs the remote command,
            #    so it would capture only the local target-file diff. Scoring
            #    never reads code_backup/ (see executor.py rationale).
            if backup and self.config.get("save_code_backup", False):
                print("Warning: save_code_backup is a no-op on the Modal backend "
                      "(remote execution leaves no local artifacts to back up).")

            # 4. Ensure a sandbox exists (create from the baked image on demand).
            self._ensure_sandbox()

            # 5. Push current target_files bytes + the whole local ml_tasks/<task>/
            #    into the remote tree (overwrites the baked copy; never clobbers
            #    agent edits to target files because the push happens before run).
            self._push_inputs()

            # 6. Remote: clean results_tmp so we never pull stale output.
            self._sb_exec(f"rm -rf {self._shq(self._remote_repo_dir + '/results_tmp')}",
                          use_conda=False)

            # 7. Run the main command (with remote .git protection mirroring
            #    _protect_git_dir). Fresh-container search evals don't need it,
            #    but the shared pre_test_val->final_test sandbox does.
            print(f"Running {command_key}: {command}")
            result = self._run_command(command)

            # 8. Verify remote workspace integrity after execution.
            integrity_error = self._check_workspace_integrity()
            if integrity_error:
                print(f"ERROR: {integrity_error}")
                return {
                    "success": False,
                    "results": None,
                    "primary_metric": None,
                    "error": f"Workspace corrupted: {integrity_error}",
                }

            if result.returncode != 0:
                phase = "val" if command_key == "val_command" else "test"
                self._save_bug_execution_record(run_id, execution_timestamp, phase, result)
                return {
                    "success": False,
                    "results": None,
                    "primary_metric": None,
                    "error": result.stderr,
                }

            # 9. Pull results_tmp/<phase>_info.json to LOCAL repo_dir/results_tmp/
            #    so the parent's _collect_results runs unchanged.
            pulled = self._pull_results(output_path)
            if not pulled:
                error_msg = f"Results file not found on remote: {output_path}"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "results": None,
                    "primary_metric": None,
                    "error": error_msg,
                }

            return super()._collect_results(run_id, execution_timestamp, output_path)

        except Exception as e:  # noqa: BLE001
            # Local run_val/run_test return a failure DICT (they do not raise) for
            # eval problems, and the agent search loops consume a dict without
            # catching. Mirror that contract: a Modal lifecycle/sync error
            # (sandbox create, tree-hash assert, push/pull) becomes a failed step
            # so the loop records it and continues, instead of crashing the run.
            print(f"ModalExecutor error during {command_key}: {e}")
            # The sandbox may be in an unknown state after a raise; tear it down
            # unconditionally so a broken sandbox is never reused (even across the
            # pre_test_val -> final_test handoff).
            self._teardown_sandbox()
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": str(e),
            }
        finally:
            # Tear the sandbox down even on interruption, EXCEPT keep it alive
            # across the pre_test_val -> final_test handoff (no-op if already
            # torn down above). _teardown_sandbox is idempotent.
            if not keep_sandbox:
                self._teardown_sandbox()

    # ------------------------------------------------------------------
    # Remote command execution (pinned exec contract)
    # ------------------------------------------------------------------

    def _run_command(self, command: str) -> SubprocessResult:
        """Run the val/test command remotely with the .git-protection wrapper.

        Mirrors the parent's _protect_git_dir try/finally so the shared
        pre_test_val->final_test sandbox cannot have .git mutated mid-command.
        """
        git = self._shq(self._remote_repo_dir + "/.git")
        # Non-recursive, mirroring the parent's os.chmod(git_dir, 0o555)/0o755
        # (executor.py _protect_git_dir/_unprotect_git_dir): guards the .git
        # directory entry itself, not its contents.
        protect = f"chmod 0555 {git} 2>/dev/null || true"
        unprotect = f"chmod 0755 {git} 2>/dev/null || true"
        # The actual eval command runs under the pinned exec contract in
        # _sb_exec; the git chmod guards run in the same shell invocation so a
        # failed command still restores writability.
        wrapped = f"{protect}; ( {command} ); __rc=$?; {unprotect}; exit $__rc"
        return self._sb_exec(wrapped, use_conda=True)

    def _sb_exec(self, command: str, use_conda: bool = True) -> SubprocessResult:
        """Pinned exec contract: run *command* in the sandbox.

        For eval commands (use_conda=True) the argv is EXACTLY:
            ["conda","run","--no-capture-output","-n",conda_env,"bash","-c",command]
        cwd = remote repo_dir, scrubbed env (no Modal-injected
        CUDA_VISIBLE_DEVICES / login-shell / extra CUDA paths).

        Timeout: the wall-clock limit is enforced SERVER-SIDE by passing
        ``timeout=self.timeout`` to ``Sandbox.exec`` (see _sb_spawn). modal 1.5's
        ContainerProcess has NO per-process kill primitive, so a client-side
        timer cannot stop a running remote eval; instead modal kills the process
        on expiry and raises ``ExecTimeoutError`` from .wait()/.read(). We catch
        it and return the byte-identical SubprocessResult(1, stderr=f"Timeout
        after {self.timeout} seconds"). self.timeout None => the exec is UNBOUNDED
        (mirrors local communicate(timeout=None) for the final test).
        """
        if use_conda:
            argv = ["conda", "run", "--no-capture-output", "-n", self.conda_env,
                    "bash", "-c", command]
            cwd = self._remote_repo_dir
        else:
            argv = ["bash", "-c", command]
            cwd = self._remote_repo_dir

        exec_timeout = self._exec_timeout_exc()
        # The spawn+wait is wrapped so that (a) a server-side timeout becomes the
        # parent's byte-identical timeout result and (b) a transient Modal
        # spawn/stream error is recorded as a failed eval (str(e)) so the agent
        # loop continues — mirroring the parent _run_command's except branches
        # (executor.py) — rather than crashing the whole run.
        try:
            proc = self._sb_spawn(argv, cwd=cwd)  # VERIFY (returns a remote process handle)
            # Mark that an eval is in flight, mirroring the parent's
            # self._current_proc bookkeeping (executor.py). On an external signal,
            # base.py.kill_running_process tears the whole sandbox down (modal 1.5
            # has no per-process kill), which stops this eval regardless of the
            # handle. Cleared in the finally below.
            self._current_proc = proc
            returncode, stdout, stderr = self._sb_wait(proc)  # VERIFY (blocks until exit)
        except exec_timeout:  # VERIFY (modal.exception.ExecTimeoutError on exec timeout=)
            # modal enforced the server-side wall-clock and killed the process.
            # Return the parent's byte-identical timeout result so a timed-out
            # remote eval reports exactly like local (executor.py TimeoutExpired).
            print(f"Command timed out after {self.timeout} seconds")
            return SubprocessResult(1, stderr=f"Timeout after {self.timeout} seconds")
        except Exception as e:  # noqa: BLE001
            print(f"Error running remote command: {e}")
            return SubprocessResult(1, stderr=str(e))
        finally:
            self._current_proc = None

        return SubprocessResult(returncode, stdout=stdout, stderr=stderr)

    def _exec_timeout_exc(self):
        """The modal exec-timeout exception class (lazy; modal-path only).

        Returns ``modal.exception.ExecTimeoutError`` — raised by .wait()/.read()
        when a ``Sandbox.exec(timeout=...)`` wall-clock expires. Falls back to a
        never-raised sentinel when modal is unavailable (off the modal path / in
        tests) so the ``except`` clause stays harmless. # VERIFY (import path).
        """
        try:
            from modal.exception import ExecTimeoutError
            return ExecTimeoutError
        except Exception:  # noqa: BLE001
            class _NeverRaised(Exception):
                pass
            return _NeverRaised

    def kill_running_process(self):
        """Stop the in-flight remote eval on an external signal.

        modal 1.5's ContainerProcess exposes NO per-process kill primitive (only
        Sandbox.terminate), so 'kill the running process' == tear the sandbox
        down, which also stops the command running inside it. Idempotent and a
        no-op when idle. Called by the shared signal handler; the happy-path /
        interruption teardown also lives in _run_phase's finally + the atexit
        hook + cleanup().
        """
        self._current_proc = None
        self._teardown_sandbox()

    # ------------------------------------------------------------------
    # Remote integrity check (mirrors parent error strings)
    # ------------------------------------------------------------------

    def _check_workspace_integrity(self) -> Optional[str]:
        """Verify the REMOTE .git + target files still exist after execution.

        Uses the same error strings as the parent so callers behave identically.
        """
        repo = self._remote_repo_dir
        git_dir = f"{repo}/.git"
        if not self._remote_exists(git_dir, is_dir=True):
            return f"CRITICAL: .git directory deleted: {git_dir}"
        target_files = self.config.get("target_files", [])
        for tf in target_files:
            tf_abs = tf if tf.startswith("/") else f"{repo}/{tf}"
            if not self._remote_exists(tf_abs, is_dir=False):
                return f"CRITICAL: target file deleted: {tf_abs}"
        return None

    def _remote_exists(self, path: str, is_dir: bool) -> bool:
        flag = "-d" if is_dir else "-e"
        res = self._sb_exec(f"test {flag} {self._shq(path)}", use_conda=False)
        return res.returncode == 0

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_sandbox(self):
        """Create the sandbox from the baked image if none is live; assert hash."""
        if self._sb is not None:
            return
        self._sb = self._sb_create()  # VERIFY (Sandbox.create from task image)
        if not self._tree_hash_checked:
            self._assert_tree_hash()
            self._tree_hash_checked = True

    def _assert_tree_hash(self):
        """Assert the remote working-tree hash matches the recorded build hash.

        Tree-hash (not commit SHA): commits embed timestamps. The manifest was
        written at build time by modal_app.images._record_tree_hash.
        """
        from modal_app import MANIFEST_DIR
        manifest_path = f"{self._remote_root}/{MANIFEST_DIR}/{self.task}.json"
        raw = self._sb_read(manifest_path)  # VERIFY
        try:
            recorded = json.loads(raw)["tree_hash"]
        except (ValueError, KeyError, TypeError):
            print(f"Warning: could not read build manifest at {manifest_path}; "
                  "skipping tree-hash assertion")
            return
        res = self._sb_exec("git rev-parse HEAD^{tree}", use_conda=False)
        actual = (res.stdout or "").strip()
        if actual and actual != recorded:
            raise RuntimeError(
                f"Remote tree-hash mismatch for task '{self.task}': "
                f"expected {recorded}, got {actual}. The baked image is stale "
                "or does not match the recorded build."
            )

    def _teardown_sandbox(self):
        if self._sb is None:
            return
        sb = self._sb
        self._sb = None
        try:
            self._sb_terminate(sb)  # VERIFY
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to terminate sandbox: {e}")

    def _atexit_terminate(self):
        """atexit backstop: terminate any lingering sandbox on interpreter exit."""
        try:
            self._teardown_sandbox()
        except Exception:  # noqa: BLE001
            pass

    def cleanup(self):
        """Terminate any lingering sandbox, then local git reset (parent)."""
        self._teardown_sandbox()
        # Drop the atexit hook for this (now cleaned-up) executor so the registry
        # does not accumulate dead bound methods across many executors in one run.
        atexit.unregister(self._atexit_terminate)
        super().cleanup()

    # ------------------------------------------------------------------
    # Per-eval input sync (push)
    # ------------------------------------------------------------------

    def _push_inputs(self):
        """Push current target_files bytes + the whole local ml_tasks/<task>/.

        Push happens BEFORE the command runs and overwrites the baked copy,
        guaranteeing the cp-ed harness scripts (train.py, etc.) are current;
        the repo/datasets are never pushed and checkpoints are never pulled.
        """
        # 5a. Push the whole local ml_tasks/<task>/ -> remote ml_tasks/<task>/.
        local_task_dir = osp.join("ml_tasks", self.task)
        remote_task_dir = f"{self._remote_root}/ml_tasks/{self.task}"
        if osp.isdir(local_task_dir):
            for root, _dirs, files in os.walk(local_task_dir):
                rel_root = osp.relpath(root, local_task_dir)
                for fn in files:
                    local_fp = osp.join(root, fn)
                    rel = fn if rel_root == "." else f"{rel_root}/{fn}"
                    remote_fp = f"{remote_task_dir}/{rel}"
                    with open(local_fp, "rb") as f:
                        self._sb_write(remote_fp, f.read())  # VERIFY

        # 5b. Push current target_files bytes -> remote repo_dir/<tf>.
        for tf in self.config.get("target_files", []):
            local_fp = tf if osp.isabs(tf) else osp.join(self.repo_dir, tf)
            if not osp.isfile(local_fp):
                continue
            remote_fp = tf if tf.startswith("/") else f"{self._remote_repo_dir}/{tf}"
            with open(local_fp, "rb") as f:
                self._sb_write(remote_fp, f.read())  # VERIFY

    # ------------------------------------------------------------------
    # Per-eval result sync (pull)
    # ------------------------------------------------------------------

    def _pull_results(self, output_path: str) -> bool:
        """Pull remote results_tmp/<phase>_info.json to LOCAL repo_dir/results_tmp/.

        Returns True if the file was pulled. The parent's _collect_results then
        reads the LOCAL copy unchanged.
        """
        remote_fp = f"{self._remote_repo_dir}/{output_path}"
        if not self._remote_exists(remote_fp, is_dir=False):
            return False
        data = self._sb_read_bytes(remote_fp)  # VERIFY
        local_fp = osp.join(osp.abspath(self.repo_dir), output_path)
        os.makedirs(osp.dirname(local_fp), exist_ok=True)
        with open(local_fp, "wb") as f:
            f.write(data)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shq(path: str) -> str:
        """Single-quote a path for safe interpolation into a remote bash -c."""
        return "'" + path.replace("'", "'\\''") + "'"

    # ==================================================================
    # Thin Modal SDK adapter — the ONLY place that touches the Modal SDK.
    # Every method here is marked # VERIFY (no Modal account to live-verify).
    # ==================================================================

    def _sb_create(self):
        """Create an ephemeral GPU sandbox from the task's baked image."""
        from modal_app import SANDBOX_TAG, app, task_gpu, task_image, task_volumes
        image = task_image(self.task)
        gpu = task_gpu(self.task)
        volumes = task_volumes(self.task)
        # Backstops against a leaked GPU sandbox: graceful SIGINT -> atexit /
        # _run_phase finally (terminates the sandbox); the per-eval wall-clock is
        # enforced server-side via Sandbox.exec(timeout=) (see _sb_spawn), which
        # bounds each command. For HARD kills (SIGKILL/OOM/host crash) that skip
        # atexit, the backstop is the manual `reap` command (provision.py::reap).
        # NOTE: no sandbox-level `timeout=` is passed here on purpose — the final
        # test runs unbounded (self.timeout is None), so a fixed sandbox TTL could
        # kill a legit long run. An operator who wants a hard server-side cap can
        # add `timeout=` below (see docs/MODAL.md). # VERIFY
        create_kwargs = dict(
            app=app, image=image, volumes=volumes,
            workdir=self._remote_repo_dir,
        )
        # CPU-only tasks (task_gpu returns None) are created WITHOUT a gpu= request
        # so no GPU is billed for an eval that never uses one.
        if gpu is not None:
            create_kwargs["gpu"] = gpu  # VERIFY (gpu= label)
        sb = modal_Sandbox_create(**create_kwargs)  # VERIFY (Sandbox.create signature)
        # Tag the sandbox so the reaper (provision.py::reap) can find ONLY this
        # app's orphans and never mass-terminate unrelated sandboxes. Defensive:
        # if the tagging API differs, degrade gracefully (eval still runs; the
        # reaper just can't see this sandbox by tag).
        try:
            sb.set_tags({**SANDBOX_TAG, "task": self.task})  # VERIFY (Sandbox.set_tags)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not tag sandbox (reaper filter degraded): {e}")
        return sb

    def _sb_spawn(self, argv: list, cwd: str):
        """Spawn *argv* in the sandbox with a scrubbed env + server-side timeout.

        Scrubbed env = a minimal env WITHOUT Modal-injected CUDA_VISIBLE_DEVICES
        / login-shell / extra CUDA paths, so remote device selection matches a
        clean local shell. A positive self.timeout is passed as ``timeout=`` so
        modal kills the process and raises ExecTimeoutError on expiry (modal 1.5
        has no per-process client kill); self.timeout None => unbounded (final
        test). Returns a remote process handle.
        """
        kwargs = dict(workdir=cwd, env=self._scrubbed_env())
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            kwargs["timeout"] = self.timeout  # VERIFY (Sandbox.exec timeout= server-side kill)
        # VERIFY (Sandbox.exec signature: argv, workdir, env, timeout)
        return self._sb.exec(*argv, **kwargs)

    def _sb_wait(self, proc):
        """Block until *proc* exits; return (returncode, stdout, stderr)."""
        proc.wait()  # VERIFY (process handle .wait())
        stdout = proc.stdout.read() if hasattr(proc, "stdout") else ""  # VERIFY
        stderr = proc.stderr.read() if hasattr(proc, "stderr") else ""  # VERIFY
        returncode = getattr(proc, "returncode", 0)  # VERIFY
        return returncode, stdout, stderr

    def _sb_write(self, remote_path: str, data: bytes):
        """Write *data* to *remote_path* inside the sandbox (mkdir -p parents)."""
        parent = remote_path.rsplit("/", 1)[0]
        self._sb.exec("bash", "-c", f"mkdir -p {self._shq(parent)}").wait()  # VERIFY
        with self._sb.open(remote_path, "wb") as f:  # VERIFY (Sandbox.open)
            f.write(data)

    def _sb_read(self, remote_path: str) -> str:
        """Read *remote_path* from the sandbox as text (empty string if absent)."""
        try:
            with self._sb.open(remote_path, "r") as f:  # VERIFY
                return f.read()
        except Exception:  # noqa: BLE001
            return ""

    def _sb_read_bytes(self, remote_path: str) -> bytes:
        """Read *remote_path* from the sandbox as bytes."""
        with self._sb.open(remote_path, "rb") as f:  # VERIFY
            return f.read()

    def _sb_terminate(self, sb):
        """Terminate the sandbox."""
        sb.terminate()  # VERIFY (Sandbox.terminate)

    def _scrubbed_env(self) -> dict:
        """Env for remote exec: INHERIT the orchestrator env, then strip the
        machine-specific / Modal-managed vars.

        Local `_run_command` (executor.py) runs Popen with NO env= and so
        inherits the full parent os.environ — including user vars some task eval
        scripts read at run time (e.g. open-unlearning fetches model/dataset from
        the HF Hub during eval: HF_TOKEN/HF_HOME/HF_ENDPOINT/HF_HUB_OFFLINE,
        http(s)_proxy, WANDB_*). Inheriting-then-stripping is far closer to local
        than a 4-var whitelist, while still removing vars that must NOT cross to
        the remote box (local paths, and the GPU pin Modal manages).
        """
        env = dict(os.environ)
        # Strip local-machine-specific / Modal-managed vars so remote device
        # selection and lib discovery are not perturbed (R2). conda run sets the
        # env's own PATH/LD_LIBRARY_PATH, and Modal assigns the GPU, so a locally
        # exported CUDA_VISIBLE_DEVICES must not leak. # VERIFY (confirm the exact
        # set Modal injects on a live account).
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH", "LD_PRELOAD",
            "PATH",  # overlaid below with the image's conda PATH
            "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CONDA_EXE",
            "CONDA_PYTHON_EXE", "_CE_CONDA", "_CE_M",
            "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
        ):
            env.pop(k, None)
        env["PATH"] = "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
        env.setdefault("HOME", "/root")
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_ALL", "C.UTF-8")
        return env


def modal_Sandbox_create(**kwargs):
    """Indirection so the Modal SDK symbol is resolved lazily (modal absent here).

    Imported and called only on the Modal path. Keeps the SDK import out of
    module-import scope so this file compiles/imports without ``modal``.
    """
    import modal  # VERIFY (Sandbox.create lives on modal.Sandbox)
    app = kwargs.pop("app")
    return modal.Sandbox.create(app=app, **kwargs)  # VERIFY (exact create signature)
