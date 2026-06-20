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
import time
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

    # Transient-error backoff / reconnect cadence (class defaults; instance-
    # overridable — tests shrink these).
    _RETRY_BACKOFF_START = 5.0       # first retry delay (s)
    _RETRY_BACKOFF_CAP = 60.0        # max backoff between retries (s)
    _RECONNECT_POLL_INTERVAL = 10.0  # (b1) gap between result-file polls (s)
    _RECONNECT_POLL_MAX_S = 7200.0   # (b1) poll cap when self.timeout is None
    # (b1/M2) the remote wrapper writes the eval's TRUE exit code here as its LAST
    # action, so a mid-eval reconnect recovers the real returncode (not just "a
    # result file exists"). In results_tmp/ (scratch; never pulled/parsed).
    _EXEC_RC_SENTINEL = "results_tmp/.fmlbench_exec_rc"
    # Sandbox MAX-LIFETIME (modal's Sandbox.create default is 300s = a 5-min cap, NOT
    # unbounded). We pass an explicit lifetime >= the eval budget so a long eval is
    # never killed mid-run (R2), and it doubles as a server-side orphan TTL backstop.
    _SANDBOX_LIFETIME_HEADROOM = 1800.0  # bounded eval: exec budget + this (s)
    _SANDBOX_MAX_LIFETIME = 86400.0      # unbounded final test cap (24h; config-overridable)

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
        self._sandbox_id = None      # remote sandbox id, for mid-eval reconnect (b1)
        self._app = None             # cached running App handle (modal.App.lookup)
        self._tree_hash_checked = False
        # Transient Modal-infra resilience (NOT eval failures): retry idempotent
        # ops + reconnect-poll the result on a mid-eval drop, up to this wall-clock
        # budget (default 30 min). Overridable per task via config.
        self._retry_budget_s = float(self.config.get("modal_retry_budget_seconds", 1800.0))
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

            # 6. Remote: clean results_tmp so we never pull stale output. Idempotent ->
            #    retry transient connection errors (parity with the other control ops).
            self._with_modal_retry(
                "clean-results",
                lambda: self._sb_exec(
                    f"rm -rf {self._shq(self._remote_repo_dir + '/results_tmp')}",
                    use_conda=False),
            )

            # 7. Run the main command (with remote .git protection mirroring
            #    _protect_git_dir). Fresh-container search evals don't need it,
            #    but the shared pre_test_val->final_test sandbox does.
            print(f"Running {command_key}: {command}")
            try:
                result = self._run_command(command)
            except self._transient_modal_errors() as e:
                # (b1) The orchestrator lost its connection mid-eval, but the
                # command keeps running on the remote sandbox. Reconnect by id and
                # poll for the result file instead of re-running (no wasted work).
                print(f"Connection lost during {command_key}; reconnecting and "
                      f"polling for the result (no re-run): {e}")
                result = self._reconnect_and_poll(output_path)

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
        # failed command still restores writability. As the LAST action, record the
        # TRUE exit code into the _EXEC_RC_SENTINEL file so a mid-eval reconnect
        # (_reconnect_and_poll) recovers the real returncode instead of inferring
        # success from result-file presence. The sentinel lives in results_tmp/
        # (scratch; never pulled/parsed) and is written AFTER the command, so it
        # cannot perturb the eval or its result. (M2)
        rc_dir = self._shq(self._remote_repo_dir + "/results_tmp")
        rc_file = self._shq(self._remote_repo_dir + "/" + self._EXEC_RC_SENTINEL)
        wrapped = (f"{protect}; ( {command} ); __rc=$?; {unprotect}; "
                   f"mkdir -p {rc_dir} 2>/dev/null; "
                   f"printf '%s' \"$__rc\" > {rc_file} 2>/dev/null; exit $__rc")
        return self._sb_exec(wrapped, use_conda=True)

    def _sb_exec(self, command: str, use_conda: bool = True) -> SubprocessResult:
        """Pinned exec contract: run *command* in the sandbox.

        For eval commands (use_conda=True) the argv is EXACTLY:
            ["conda","run","--no-capture-output","-n",conda_env,"bash","-c",command]
        cwd = remote repo_dir, scrubbed env (no Modal-injected
        CUDA_VISIBLE_DEVICES / login-shell / extra CUDA paths).

        Timeout: the wall-clock limit is enforced SERVER-SIDE by passing
        ``timeout=self.timeout`` to ``Sandbox.exec`` (see _sb_spawn). modal 1.5's
        ContainerProcess has NO per-process kill primitive, so a client-side timer
        cannot stop a running remote eval. On expiry modal 1.5's
        ContainerProcess.wait() SWALLOWS the ExecTimeoutError and returns
        ``returncode == -1`` (it does NOT raise — see its source); .read() MAY still
        raise ExecTimeoutError in some cases. So we map BOTH a rc==-1 (when a
        wall-clock is set, below) AND a raised ExecTimeoutError (the except) to the
        parent's byte-identical SubprocessResult(1, stderr=f"Timeout after
        {self.timeout} seconds"). self.timeout None => the exec is UNBOUNDED (mirrors
        local communicate(timeout=None) for the final test; no deadline => no rc==-1).
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
        # parent's byte-identical timeout result, (b) a TRANSIENT Modal infra error
        # PROPAGATES (callers retry idempotent ops / reconnect-poll the eval), and
        # (c) any other error is recorded as a failed eval (str(e)) so the agent
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
        except self._transient_modal_errors():
            # Transient Modal infra error (e.g. "Could not connect to the Modal
            # server"). PROPAGATE so callers can recover: idempotent control ops
            # retry via _with_modal_retry; the main eval reconnects + polls in
            # _run_phase (b1). NOT swallowed into a failure result here.
            raise
        except Exception as e:  # noqa: BLE001
            print(f"Error running remote command: {e}")
            return SubprocessResult(1, stderr=str(e))
        finally:
            self._current_proc = None

        # modal 1.5's ContainerProcess.wait() catches the server-side ExecTimeoutError
        # internally, sets returncode = -1, and returns (does NOT raise — so the
        # `except exec_timeout` above only fires on the rarer .read() path). When a
        # wall-clock is set, map that -1 sentinel to the parent's byte-identical
        # timeout result. A normal process never yields -1 (exit codes are 0-255;
        # signals are 128+sig); with self.timeout None no deadline is set so -1 cannot
        # occur. (M1)
        if returncode == -1 and isinstance(self.timeout, (int, float)) and self.timeout > 0:
            print(f"Command timed out after {self.timeout} seconds")
            return SubprocessResult(1, stderr=f"Timeout after {self.timeout} seconds")

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

    # ------------------------------------------------------------------
    # Transient-error resilience (Modal infra, NOT eval failures)
    # ------------------------------------------------------------------

    @staticmethod
    def _transient_modal_errors():
        """Exception types treated as TRANSIENT Modal infra errors — safe to retry
        / reconnect (e.g. "Could not connect to the Modal server", raised in
        modal/_utils/grpc_utils.py). Deliberately NARROW: NOT auth/not-found/
        invalid/timeout/nonzero-returncode, which are deterministic and fail fast.
        """
        errs = [ConnectionError]  # builtin (the grpc layer may raise this)
        try:
            from modal.exception import ConnectionError as _ModalConnErr  # VERIFY (transient)
            errs.append(_ModalConnErr)
        except Exception:  # noqa: BLE001  (modal absent off the modal path / in tests)
            pass
        return tuple(errs)

    def _with_modal_retry(self, op_name: str, fn):
        """Run *fn*, retrying ONLY transient Modal connection errors with
        exponential backoff until self._retry_budget_s elapses, then re-raise.

        For IDEMPOTENT ops ONLY (create, reads, existence checks, result pull) —
        never the main eval command (retrying that would re-run it). Non-transient
        errors propagate immediately (no retry). On the happy path fn runs once.
        """
        transient = self._transient_modal_errors()
        start = time.monotonic()
        delay = self._RETRY_BACKOFF_START
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except transient as e:
                elapsed = time.monotonic() - start
                if elapsed >= self._retry_budget_s:
                    print(f"Modal {op_name}: transient error after {attempt} attempts "
                          f"/ {elapsed:.0f}s; retry budget exhausted: {e}")
                    raise
                print(f"Modal {op_name}: transient error (attempt {attempt}, "
                      f"{elapsed:.0f}s elapsed); retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
                delay = min(delay * 2.0, self._RETRY_BACKOFF_CAP)

    def _reconnect_sandbox(self):
        """Reconnect to the still-running sandbox by id (overridable in tests).

        Modal sandboxes keep running independent of the client connection, so a
        dropped connection is recovered with Sandbox.from_id — no re-run.
        """
        import modal
        return modal.Sandbox.from_id(self._sandbox_id)  # VERIFY (from_id; live api 1.5.0)

    def _reconnect_and_poll(self, output_path: str) -> SubprocessResult:
        """(b1) After a transient drop DURING the eval, reconnect to the running
        sandbox and recover the eval's TRUE exit code — no re-run.

        The remote command keeps running and, as its LAST action, writes its exit
        code to the _EXEC_RC_SENTINEL file (see _run_command). We reconnect by id and
        poll for that sentinel; its presence means the eval finished and its content
        IS the real returncode, which we return verbatim — so a nonzero-exit eval is
        reported as a failure exactly like the normal path, NOT a false success from
        mere result-file presence (M2). Polls for up to ``self.timeout`` seconds
        MEASURED FROM RECONNECT (a 7200s cap when self.timeout is None — the final
        test passes no server-side exec timeout, so this cap is the poll's ceiling),
        tolerating further transient errors. Always finite; returns a failure result
        if the sentinel never appears or there is no sandbox id to reconnect to.
        """
        if not self._sandbox_id:
            return SubprocessResult(
                1, stderr="Connection lost during eval and no sandbox id to reconnect to")
        rc_file = f"{self._remote_repo_dir}/{self._EXEC_RC_SENTINEL}"
        budget = (self.timeout if isinstance(self.timeout, (int, float)) and self.timeout > 0
                  else self._RECONNECT_POLL_MAX_S)
        deadline = time.monotonic() + budget
        transient = self._transient_modal_errors()
        while time.monotonic() < deadline:
            try:
                self._sb = self._reconnect_sandbox()
                # The sentinel exists iff the command finished; its content is the rc.
                proc = self._sb.exec("bash", "-c", f"cat {self._shq(rc_file)} 2>/dev/null")  # VERIFY
                proc.wait()
                if getattr(proc, "returncode", 1) == 0:
                    raw = (proc.stdout.read() if hasattr(proc, "stdout") else "") or ""
                    rc_str = raw.strip()
                    if rc_str:
                        try:
                            rc = int(rc_str)          # robust: '--5'/garbage -> ValueError
                        except ValueError:
                            rc = 1                    # malformed sentinel -> treat as failure
                        print(f"Reconnected; eval finished (rc={rc}) — recovered "
                              "without re-running.")
                        # Streams are genuinely lost with the dropped connection; note that
                        # in stderr so a recovered nonzero-exit bug record isn't blank. (L-e)
                        stderr = ("" if rc == 0 else
                                  "(eval failed; stdout/stderr lost to a mid-eval connection "
                                  "drop — exit code recovered via the rc sentinel)")
                        return SubprocessResult(rc, stderr=stderr)
            except transient as e:
                print(f"Reconnect/poll transient error, will keep trying: {e}")
            time.sleep(self._RECONNECT_POLL_INTERVAL)
        return SubprocessResult(
            1, stderr="Connection lost during eval; result not found after reconnect/poll")

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
        # Idempotent check -> retry transient connection errors (a).
        res = self._with_modal_retry(
            "remote-exists",
            lambda: self._sb_exec(f"test {flag} {self._shq(path)}", use_conda=False),
        )
        return res.returncode == 0

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_sandbox(self):
        """Create the sandbox from the baked image if none is live; assert hash."""
        if self._sb is not None:
            return
        # Retry transient connection errors during creation (a); App.lookup +
        # Sandbox.create both go through here. Record the id for reconnect (b1).
        self._sb = self._with_modal_retry("sandbox-create", self._sb_create)  # VERIFY
        self._sandbox_id = getattr(self._sb, "object_id", None)
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
        raw = self._with_modal_retry("read-manifest", lambda: self._sb_read(manifest_path))  # VERIFY
        try:
            recorded = json.loads(raw)["tree_hash"]
        except (ValueError, KeyError, TypeError):
            print(f"Warning: could not read build manifest at {manifest_path}; "
                  "skipping tree-hash assertion")
            return
        res = self._with_modal_retry(
            "git-rev-parse",
            lambda: self._sb_exec("git rev-parse HEAD^{tree}", use_conda=False),
        )
        actual = (res.stdout or "").strip()
        if actual and actual != recorded:
            raise RuntimeError(
                f"Remote tree-hash mismatch for task '{self.task}': "
                f"expected {recorded}, got {actual}. The baked image is stale "
                "or does not match the recorded build."
            )

    def _teardown_sandbox(self):
        if self._sb is None:
            self._sandbox_id = None
            return
        sb = self._sb
        self._sb = None
        self._sandbox_id = None
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
        data = self._with_modal_retry("pull-results", lambda: self._sb_read_bytes(remote_fp))  # VERIFY
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
    # Each call is marked # VERIFY; the App.lookup -> Sandbox.create -> exec ->
    # wait -> filesystem.{write,read}_* -> set_tags -> terminate path was
    # live-confirmed on modal 1.5.0 (filesystem replaces the deprecated open()).
    # ==================================================================

    def _running_app(self):
        """A *running* App handle for ephemeral Sandbox.create from a plain process.

        A bare ``modal.App(name)`` (modal_app.app) is uninitialized and usable only
        under ``modal run`` / ``app.run()``; ``Sandbox.create`` from this plain
        orchestrator process rejects it ("App has not been initialized"), so we
        resolve a running handle by name (creating it if missing) and cache it.
        """
        if self._app is None:
            import modal
            from modal_app import APP_NAME
            # VERIFY (App.lookup create_if_missing; live-confirmed on modal 1.5.0)
            self._app = modal.App.lookup(APP_NAME, create_if_missing=True)
        return self._app

    def _sb_create(self):
        """Create an ephemeral GPU sandbox from the task's baked image."""
        from modal_app import SANDBOX_TAG, task_gpu, task_image, task_volumes
        app = self._running_app()
        image = task_image(self.task)
        gpu = task_gpu(self.task)
        volumes = task_volumes(self.task)
        # Sandbox MAX-LIFETIME (timeout=): modal's Sandbox.create default is 300s — a
        # 5-MINUTE CAP, NOT unbounded — which would kill any eval running >5 min
        # mid-run and break R2 (the local eval has no such cap). So pass an explicit
        # lifetime >= the eval budget: a bounded eval gets self.timeout + headroom (the
        # exec itself is still bounded server-side at self.timeout via
        # Sandbox.exec(timeout=) in _sb_spawn, so the sandbox only needs to outlive it);
        # the unbounded final test (self.timeout is None) gets a generous finite cap.
        # This ALSO acts as a server-side orphan TTL backstop for HARD kills
        # (SIGKILL/OOM/host crash) that skip the atexit/_run_phase-finally teardown —
        # complementing the manual `reap` (provision.py::reap). # VERIFY (timeout= = max
        # sandbox lifetime; default 300 is too low — pinned by the L3 contract test).
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            sb_timeout = int(self.timeout + self._SANDBOX_LIFETIME_HEADROOM)
        else:
            sb_timeout = int(self.config.get("modal_sandbox_max_lifetime_seconds",
                                             self._SANDBOX_MAX_LIFETIME))
        create_kwargs = dict(
            app=app, image=image, volumes=volumes,
            workdir=self._remote_repo_dir,
            timeout=sb_timeout,
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
            # Stamp a creation epoch into the tags: modal 1.5's Sandbox exposes no
            # created_at attribute, so the reaper (provision.py) reads age from this
            # tag via get_tags() instead. (H1)
            sb.set_tags({**SANDBOX_TAG, "task": self.task,
                         "created_at": str(int(time.time()))})  # VERIFY (Sandbox.set_tags)
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
        """Write *data* to *remote_path* inside the sandbox (mkdir -p parents).

        Uses the Sandbox.filesystem API; Sandbox.open/FileIO was deprecated
        2026-03-09. NOTE modal's write_bytes signature is (data, remote_path) —
        data FIRST (verified by the L3 contract test's arg-order pin).
        """
        fs = self._sb.filesystem  # VERIFY (Sandbox.filesystem property; modal 1.5.0)
        parent = remote_path.rsplit("/", 1)[0]
        if parent:
            fs.make_directory(parent, create_parents=True)  # VERIFY
        fs.write_bytes(data, remote_path)  # VERIFY (data, remote_path)

    def _sb_read(self, remote_path: str) -> str:
        """Read *remote_path* from the sandbox as text (empty string if absent)."""
        try:
            return self._sb.filesystem.read_text(remote_path)  # VERIFY (Sandbox.filesystem)
        except self._transient_modal_errors():
            raise  # transient infra -> let _with_modal_retry handle it
        except Exception:  # noqa: BLE001
            return ""  # genuinely missing / unreadable

    def _sb_read_bytes(self, remote_path: str) -> bytes:
        """Read *remote_path* from the sandbox as bytes."""
        return self._sb.filesystem.read_bytes(remote_path)  # VERIFY (Sandbox.filesystem)

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
            "HOME",  # local home (/home/<user>) doesn't exist on the sandbox; reset
                     # to /root below. HF_HOME/etc. are separate and NOT stripped. (L1)
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
