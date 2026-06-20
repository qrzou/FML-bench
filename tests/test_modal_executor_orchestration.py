"""
Layer-2 test: ModalExecutor orchestration, end-to-end, WITHOUT a Modal account.

The whole point of the Modal backend is that ONLY the val/test command is
dispatched to a remote sandbox; everything else (push inputs, pull results,
parse metric, sandbox lifecycle) is local plumbing. ModalExecutor funnels every
Modal SDK touch through 8 ``_sb_*`` adapters + ``modal_Sandbox_create``. We
exploit that seam: a FakeSandbox maps the remote root onto a LOCAL temp dir and
runs commands with REAL bash/git (stripping only the ``conda run -n <env>``
prefix and rewriting remote-absolute paths). So "remote execution" is genuine
subprocess + genuine files + genuine git — the ONLY fakery is the Modal boundary
itself.

What this DOES prove (offline):
  - the exact conda argv + the verbatim val/test command are issued (R2: same
    command runs);
  - target files + ml_tasks/<task>/ are pushed with identical bytes;
  - results_tmp/<phase>_info.json is pulled to the LOCAL repo_dir and parsed by
    the INHERITED parent code (so local-side processing is byte-identical);
  - sandbox lifecycle: create-once-and-reuse across pre_test_val->final_test
    (checkpoint handoff), teardown on every other phase, teardown on a raised
    error (which becomes a failure DICT, never a crash), tree-hash mismatch ->
    failure dict, timeout -> the parent's byte-identical timeout result.

What this does NOT prove (inherently live, see docs/MODAL.md):
  - that the REAL Modal SDK behaves like the fake (the # VERIFY surface);
  - that the remote GPU/env reproduces the metric byte-for-byte (R2 acceptance).

Run with stdlib unittest (no pytest, no modal needed):
    python tests/test_modal_executor_orchestration.py
"""
import os
import os.path as osp
import subprocess
import sys
import tempfile
import types
import unittest

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Stubbed-into-place modal_app constants. These MUST match the real
# modal_app/images.py values; test_modal_api_contract.py asserts they do (when
# modal is installed), closing the loop on this duplication.
STUB_REMOTE_ROOT = "/root/fml-bench"
STUB_MANIFEST_DIR = ".fmlbench_manifest"


# ----------------------------------------------------------------------------
# Fake Modal SDK objects (local-temp-dir-backed; real bash/git underneath).
# ----------------------------------------------------------------------------
class _Readable:
    """Minimal stand-in for a remote process stdout/stderr stream (.read())."""
    def __init__(self, s):
        self._s = s

    def read(self):
        return self._s


class FakeProc:
    """Stand-in for a Modal remote process handle (already run to completion)."""
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = _Readable(stdout)
        self.stderr = _Readable(stderr)
        self.terminated = False

    def wait(self):
        return self.returncode

    def terminate(self):
        self.terminated = True


class FakeSandbox:
    """Maps REMOTE_ROOT onto a local temp dir and runs commands with real bash.

    Only the Modal boundary is faked: the ``conda run --no-capture-output -n
    <env>`` prefix is stripped (and recorded), remote-absolute paths are
    rewritten to the temp dir, then a genuine subprocess runs.
    """
    def __init__(self, remote_root, local_root, record):
        self.remote_root = remote_root      # e.g. "/root/fml-bench"
        self.local_root = local_root        # temp dir standing in for it
        self.record = record

    def _map(self, s):
        if isinstance(s, str) and self.remote_root in s:
            return s.replace(self.remote_root, self.local_root)
        return s

    def exec(self, *argv, workdir=None, env=None, timeout=None):
        import shutil as _sh
        self.record["exec_argv"].append(list(argv))
        self.record["exec_timeouts"].append(timeout)
        real = list(argv)
        # Strip the conda wrapper exactly as the pinned exec contract builds it.
        if real[:4] == ["conda", "run", "--no-capture-output", "-n"] and len(real) >= 5:
            self.record["conda_envs"].append(real[4])
            real = real[5:]  # -> ["bash", "-c", <cmd>]
        real = [self._map(a) for a in real]
        if real and not osp.isabs(real[0]):
            real[0] = _sh.which(real[0]) or real[0]
        wd = self._map(workdir) if workdir else None
        if env is not None:
            self.record["last_env"] = dict(env)
        # Build a runnable env: keep whatever was passed (so we can assert on the
        # scrubbed env) but guarantee PATH can find bash/git/coreutils.
        run_env = dict(env) if env is not None else dict(os.environ)
        run_env["PATH"] = (run_env.get("PATH", "") + ":" + os.environ.get("PATH", "")).strip(":")
        cp = subprocess.run(real, cwd=wd, env=run_env, capture_output=True, text=True)
        return FakeProc(cp.returncode, cp.stdout, cp.stderr)

    def open(self, path, mode="r"):
        local = self._map(path)
        if "w" in mode:
            os.makedirs(osp.dirname(local), exist_ok=True)
        return open(local, mode)

    def set_tags(self, tags):
        self.record["tags"] = dict(tags)

    def terminate(self):
        self.record["terminated"] += 1


# ----------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------
class _ModalExecutorTestBase(unittest.TestCase):
    """Builds a hermetic local "remote world" and a ModalExecutor wired to a
    FakeSandbox over it."""

    TASK = "Demo_task"
    CONDA_ENV = "demoenv"

    def setUp(self):
        # 1. Stub modal_app so ModalExecutor.__init__ (which does
        #    `from modal_app import REMOTE_ROOT`) does not pull in the real
        #    package (whose top-level `import modal` would fail).
        self._saved_modal_app = sys.modules.get("modal_app")
        stub = types.ModuleType("modal_app")
        stub.REMOTE_ROOT = STUB_REMOTE_ROOT
        stub.MANIFEST_DIR = STUB_MANIFEST_DIR
        sys.modules["modal_app"] = stub
        self.addCleanup(self._restore_modal_app)

        # 2. Temp world; run from it so relative repo_dir / ml_tasks resolve here.
        self.tmp = tempfile.mkdtemp(prefix="modalexec_")
        self.addCleanup(self._rmtmp)
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self._orig_cwd)

        # 3. Local side: repo_dir is the runner's suffixed path; train.py present.
        self.repo_dir = osp.join("workspace", self.TASK + "__suf", "inner")
        os.makedirs(self.repo_dir, exist_ok=True)
        with open(osp.join(self.repo_dir, "train.py"), "w") as f:
            f.write("# local target file\nprint('train')\n")

        # 4. Local ml_tasks/<task>/ to be pushed.
        self.local_task_dir = osp.join("ml_tasks", self.TASK)
        os.makedirs(self.local_task_dir, exist_ok=True)
        with open(osp.join(self.local_task_dir, "harness.py"), "w") as f:
            f.write("# harness pushed each eval\n")

        # 5. Remote world: temp dir standing in for REMOTE_ROOT, with a git repo
        #    at the derived remote repo_dir + a tree-hash manifest.
        self.remote_root_local = osp.join(self.tmp, "remote")
        self.remote_repo = osp.join(self.remote_root_local, "workspace", self.TASK, "inner")
        os.makedirs(self.remote_repo, exist_ok=True)
        with open(osp.join(self.remote_repo, "train.py"), "w") as f:
            f.write("# baked target file\n")
        self._git(self.remote_repo, "init")
        self._git(self.remote_repo, "add", "-A")
        self._git(self.remote_repo, "-c", "user.email=t@t", "-c", "user.name=t",
                  "commit", "-m", "baseline")
        self.tree_hash = self._git(
            self.remote_repo, "rev-parse", "HEAD^{tree}", capture=True).strip()
        self._write_manifest(self.tree_hash)

        # 6. CUDA pin in the orchestrator env -> assert _scrubbed_env strips it.
        self._saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        self.addCleanup(self._restore_cuda)

        self.record = {"create": 0, "terminated": 0, "exec_argv": [],
                       "exec_timeouts": [], "conda_envs": [], "tags": None,
                       "last_env": None}

    # -- cleanup helpers --------------------------------------------------
    def _restore_modal_app(self):
        if self._saved_modal_app is not None:
            sys.modules["modal_app"] = self._saved_modal_app
        else:
            sys.modules.pop("modal_app", None)

    def _restore_cuda(self):
        if self._saved_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved_cuda

    def _rmtmp(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- small utilities --------------------------------------------------
    def _git(self, cwd, *args, capture=False):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode != 0 and not capture:
            self.fail(f"git {args} failed: {r.stderr}")
        return r.stdout if capture else None

    def _write_manifest(self, tree_hash):
        import json
        mdir = osp.join(self.remote_root_local, STUB_MANIFEST_DIR)
        os.makedirs(mdir, exist_ok=True)
        with open(osp.join(mdir, f"{self.TASK}.json"), "w") as f:
            json.dump({"task": self.TASK, "repo_dir": self.remote_repo,
                       "tree_hash": tree_hash}, f)

    def _make_executor(self, **config_overrides):
        import benchmark.modal_executor as me

        record = self.record
        remote_root_local = self.remote_root_local

        class FakeModalExecutor(me.ModalExecutor):
            def _sb_create(self_inner):
                record["create"] += 1
                sb = FakeSandbox(STUB_REMOTE_ROOT, remote_root_local, record)
                # mirror the real _sb_create's tag stamp so it is exercised
                sb.set_tags({"fml_bench": "fml-bench-eval", "task": self_inner.task})
                return sb

        config = {
            "repo_dir": self.repo_dir,
            "conda_env": self.CONDA_ENV,
            "metric": "acc",
            "target_files": ["train.py"],
            "val_command": ("mkdir -p results_tmp && "
                            "printf '{\"acc\": 0.75}' > results_tmp/val_info.json"),
            "test_command": ("mkdir -p results_tmp && "
                             "printf '{\"acc\": 0.80}' > results_tmp/test_info.json"),
        }
        config.update(config_overrides)
        ex = FakeModalExecutor(config, "agent", self.TASK, "exp", timeout=600)
        ex.workspace_dir = osp.join(self.tmp, "ws")
        os.makedirs(ex.workspace_dir, exist_ok=True)
        return ex


class TestHappyPath(_ModalExecutorTestBase):
    def test_val_runs_pushes_pulls_parses_and_tears_down(self):
        ex = self._make_executor()
        res = ex.run_val("0")

        # Result shape + parsed metric (parsed by INHERITED parent code).
        self.assertTrue(res["success"], res)
        self.assertEqual(res["primary_metric"], 0.75)
        self.assertEqual(res["results"], {"acc": 0.75})

        # The metric JSON was pulled to the LOCAL repo_dir for the parent to read.
        self.assertTrue(osp.isfile(osp.join(self.repo_dir, "results_tmp", "val_info.json")))

        # Sandbox created once and torn down (a plain val is not pre_test_val).
        self.assertEqual(self.record["create"], 1)
        self.assertEqual(self.record["terminated"], 1)
        self.assertIsNone(ex._sb)

    def test_exact_conda_argv_and_verbatim_command(self):
        ex = self._make_executor()
        ex.run_val("0")

        conda_execs = [a for a in self.record["exec_argv"]
                       if a[:4] == ["conda", "run", "--no-capture-output", "-n"]]
        self.assertTrue(conda_execs, "no conda-wrapped exec was issued")
        argv = conda_execs[0]
        # conda prefix is byte-identical to BenchmarkExecutor._run_command.
        self.assertEqual(argv[:6],
                         ["conda", "run", "--no-capture-output", "-n",
                          self.CONDA_ENV, "bash"])
        self.assertEqual(argv[6], "-c")
        # The verbatim val_command appears, wrapped only by the .git chmod guard.
        self.assertIn("( " + ex.config["val_command"] + " )", argv[7])

    def test_inputs_pushed_with_identical_bytes(self):
        ex = self._make_executor()
        ex.run_val("0")

        # target file
        remote_tf = osp.join(self.remote_repo, "train.py")
        with open(osp.join(self.repo_dir, "train.py"), "rb") as f:
            local_bytes = f.read()
        with open(remote_tf, "rb") as f:
            self.assertEqual(f.read(), local_bytes)

        # ml_tasks harness
        remote_harness = osp.join(self.remote_root_local, "ml_tasks", self.TASK, "harness.py")
        self.assertTrue(osp.isfile(remote_harness))
        with open(osp.join(self.local_task_dir, "harness.py"), "rb") as f:
            local_h = f.read()
        with open(remote_harness, "rb") as f:
            self.assertEqual(f.read(), local_h)

    def test_scrubbed_env_strips_cuda_visible_devices(self):
        ex = self._make_executor()
        ex.run_val("0")
        self.assertIsNotNone(self.record["last_env"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.record["last_env"])

    def test_sandbox_tagged_for_reaper(self):
        ex = self._make_executor()
        ex.run_val("0")
        self.assertEqual(self.record["tags"],
                         {"fml_bench": "fml-bench-eval", "task": self.TASK})

    def test_eval_exec_carries_server_side_timeout(self):
        # Option B: the eval is bounded by Sandbox.exec(timeout=), not a client
        # timer. The conda-wrapped eval exec must carry timeout == self.timeout.
        ex = self._make_executor()
        ex.run_val("0")
        idxs = [i for i, a in enumerate(self.record["exec_argv"])
                if a[:4] == ["conda", "run", "--no-capture-output", "-n"]]
        self.assertTrue(idxs, "no conda-wrapped exec was issued")
        self.assertEqual(self.record["exec_timeouts"][idxs[0]], ex.timeout)


class TestSharedSandboxHandoff(_ModalExecutorTestBase):
    def test_pre_test_val_then_final_test_reuses_one_sandbox(self):
        ex = self._make_executor(
            # pre_test_val writes a checkpoint; final_test must read it back
            # from the SAME remote fs (the shared-sandbox contract).
            val_command=("mkdir -p results_tmp && echo ck > ckpt.txt && "
                         "printf '{\"acc\": 0.50}' > results_tmp/val_info.json"),
            test_command=("test -f ckpt.txt && mkdir -p results_tmp && "
                          "printf '{\"acc\": 0.90}' > results_tmp/test_info.json"),
        )

        r1 = ex.run_val("pre_test_val")
        self.assertTrue(r1["success"], r1)
        self.assertEqual(self.record["create"], 1)
        self.assertEqual(self.record["terminated"], 0)   # kept alive for handoff
        self.assertIsNotNone(ex._sb)

        r2 = ex.run_test("final_test")
        self.assertTrue(r2["success"], r2)               # ckpt.txt was visible
        self.assertEqual(r2["primary_metric"], 0.90)
        self.assertEqual(self.record["create"], 1)       # REUSED, not recreated
        self.assertEqual(self.record["terminated"], 1)   # torn down after final
        self.assertIsNone(ex._sb)


class TestErrorContract(_ModalExecutorTestBase):
    def test_tree_hash_mismatch_becomes_failure_dict_not_raise(self):
        # Corrupt the manifest so the remote tree-hash assertion fails.
        self._write_manifest("deadbeef" * 5)
        ex = self._make_executor()

        res = ex.run_val("0")  # must NOT raise
        self.assertFalse(res["success"])
        self.assertIn("mismatch", res["error"].lower())
        self.assertEqual(self.record["create"], 1)
        self.assertEqual(self.record["terminated"], 1)   # torn down on error
        self.assertIsNone(ex._sb)

    def test_nonzero_returncode_saves_bug_record_and_fails(self):
        ex = self._make_executor(
            val_command="echo boom >&2; exit 3")  # writes no results, exits nonzero
        res = ex.run_val("0")
        self.assertFalse(res["success"])
        self.assertIsNone(res["primary_metric"])
        self.assertEqual(self.record["terminated"], 1)


class TestTimeoutContract(_ModalExecutorTestBase):
    def test_server_side_timeout_yields_parent_byte_identical_result(self):
        # Option B: modal enforces the wall-clock via Sandbox.exec(timeout=) and
        # raises ExecTimeoutError from .wait()/.read(); _sb_exec must catch it and
        # return the parent's byte-identical "Timeout after N seconds" result.
        import benchmark.modal_executor as me

        record = self.record
        remote_root_local = self.remote_root_local

        class _FakeExecTimeout(Exception):
            pass

        class TimeoutExecutor(me.ModalExecutor):
            def _sb_create(self_inner):
                record["create"] += 1
                return FakeSandbox(STUB_REMOTE_ROOT, remote_root_local, record)

            def _exec_timeout_exc(self_inner):
                return _FakeExecTimeout  # stands in for modal.exception.ExecTimeoutError

            def _sb_spawn(self_inner, argv, cwd):
                return object()

            def _sb_wait(self_inner, proc):
                raise _FakeExecTimeout("exec exceeded its server-side deadline")

        config = {"repo_dir": self.repo_dir, "conda_env": self.CONDA_ENV,
                  "val_command": "sleep 999", "metric": "acc"}
        ex = TimeoutExecutor(config, "agent", self.TASK, "exp", timeout=123)
        ex.workspace_dir = osp.join(self.tmp, "ws2")
        os.makedirs(ex.workspace_dir, exist_ok=True)
        ex._ensure_sandbox()

        res = ex._sb_exec("sleep 999", use_conda=False)
        self.assertEqual(res.returncode, 1)
        self.assertEqual(res.stderr, "Timeout after 123 seconds")

    def test_transient_error_becomes_failed_eval_not_crash(self):
        # A non-timeout exception from _sb_wait must degrade to a failed eval
        # (str(e)), mirroring the parent _run_command's generic except branch.
        import benchmark.modal_executor as me

        record = self.record
        remote_root_local = self.remote_root_local

        class FlakyExecutor(me.ModalExecutor):
            def _sb_create(self_inner):
                record["create"] += 1
                return FakeSandbox(STUB_REMOTE_ROOT, remote_root_local, record)

            def _sb_spawn(self_inner, argv, cwd):
                return object()

            def _sb_wait(self_inner, proc):
                raise RuntimeError("modal stream dropped")

        config = {"repo_dir": self.repo_dir, "conda_env": self.CONDA_ENV,
                  "val_command": "x", "metric": "acc"}
        ex = FlakyExecutor(config, "agent", self.TASK, "exp", timeout=123)
        ex.workspace_dir = osp.join(self.tmp, "ws3")
        os.makedirs(ex.workspace_dir, exist_ok=True)
        ex._ensure_sandbox()

        res = ex._sb_exec("x", use_conda=False)
        self.assertEqual(res.returncode, 1)
        self.assertIn("modal stream dropped", res.stderr)


class TestLocalProcessingParity(_ModalExecutorTestBase):
    def test_result_parsing_is_inherited_unchanged(self):
        from benchmark.executor import BenchmarkExecutor
        import benchmark.modal_executor as me

        # Same function object => Modal cannot have altered local-side parsing.
        self.assertIs(me.ModalExecutor._extract_primary_metric,
                      BenchmarkExecutor._extract_primary_metric)
        self.assertIs(me.ModalExecutor._collect_results,
                      BenchmarkExecutor._collect_results)
        self.assertIs(me.ModalExecutor._filter_results,
                      BenchmarkExecutor._filter_results)

    def test_nested_metric_averaging_matches_parent(self):
        from benchmark.executor import BenchmarkExecutor

        nested = {"d1": {"means": {"acc": 0.4}}, "d2": {"means": {"acc": 0.6}}}
        cfg = {"repo_dir": self.repo_dir, "conda_env": self.CONDA_ENV,
               "metric": "acc", "val_command": "x"}
        modal_ex = self._make_executor()
        plain = BenchmarkExecutor(cfg, "a", self.TASK, "exp")
        self.assertEqual(modal_ex._extract_primary_metric(nested),
                         plain._extract_primary_metric(nested))
        self.assertEqual(modal_ex._extract_primary_metric(nested), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
