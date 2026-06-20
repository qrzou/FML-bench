"""
Layer-3 test: the Modal SDK API contract our code assumes — verified OFFLINE.

Installing the ``modal`` pip package needs NO account and NO credit; importing
it and introspecting object signatures needs no login (only real RPC calls hit
Modal's servers). So we can cheaply kill the biggest slice of the ``# VERIFY``
risk — "wrong method name / wrong kwarg / wrong builder step" — without spending
anything. It does NOT prove runtime behavior (that is the live runbook in
docs/MODAL.md); it proves the surface we call against EXISTS and accepts the
arguments we pass.

IMPORTANT:
  - This layer SKIPS unless ``modal`` is importable. Run it in a SEPARATE venv:
        python -m venv /tmp/modalvenv && /tmp/modalvenv/bin/pip install modal
        /tmp/modalvenv/bin/python tests/test_modal_api_contract.py
  - Do NOT install modal into the env that runs the OTHER test files:
    test_executor_factory.py asserts modal is ABSENT (that absence is part of
    the R1 contract), so it would start failing if modal were importable there.
  - Pin/record the modal version you tested against (printed below) into
    docs/MODAL.md, since the SDK surface can drift between versions.
"""
import importlib
import importlib.util
import inspect
import os.path as osp
import sys
import unittest

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HAVE_MODAL = importlib.util.find_spec("modal") is not None
_SKIP = "modal not installed; run this layer in a venv with `pip install modal`"

# Must mirror tests/test_modal_executor_orchestration.py's stubbed constants and
# modal_app/gpu_map.py — this is where that duplication is proven consistent.
EXPECT_REMOTE_ROOT = "/root/fml-bench"
EXPECT_MANIFEST_DIR = ".fmlbench_manifest"
EXPECT_DEFAULT_GPU = "A100"


def _accepts(test, fn, names):
    """Assert callable *fn* accepts each kwarg in *names* (or has **kwargs).

    Modal wraps objects with `synchronicity`, so signature introspection can
    fail; if it does we cannot make a hard claim and skip that check rather than
    emit a false failure.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        test.skipTest(f"signature of {fn!r} not introspectable (wrapper)")
        return
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    for n in names:
        test.assertTrue(has_var_kw or n in params,
                        f"{fn!r} does not accept kwarg '{n}' (params={list(params)})")


@unittest.skipUnless(HAVE_MODAL, _SKIP)
class TestModalSDKContract(unittest.TestCase):
    def setUp(self):
        import modal
        self.modal = modal

    def test_version_recorded(self):
        ver = getattr(self.modal, "__version__", None)
        self.assertTrue(ver, "modal.__version__ missing")
        # Surfaced so the operator pins the tested version into docs/MODAL.md.
        print(f"\n[layer3] tested against modal=={ver}")

    def test_app_constructible(self):
        app = self.modal.App("fml-bench-eval-contract-test")
        self.assertIsInstance(app, self.modal.App)

    def test_app_lookup_accepts_create_if_missing(self):
        # ModalExecutor._running_app() resolves a RUNNING app via
        # App.lookup(APP_NAME, create_if_missing=True) — a bare App() is
        # uninitialized and Sandbox.create rejects it (live-caught in the smoke).
        _accepts(self, self.modal.App.lookup, ["create_if_missing"])

    def test_sandbox_has_methods_we_call(self):
        Sandbox = self.modal.Sandbox
        for name in ("create", "exec", "open", "set_tags", "get_tags",
                     "terminate", "list"):
            self.assertTrue(hasattr(Sandbox, name),
                            f"modal.Sandbox has no '{name}' (we call it)")

    def test_sandbox_create_accepts_our_kwargs(self):
        # We call modal.Sandbox.create(app=, image=, volumes=, workdir=, gpu=).
        _accepts(self, self.modal.Sandbox.create,
                 ["app", "image", "volumes", "workdir", "gpu"])

    def test_sandbox_list_accepts_tags_filter(self):
        # reap() calls modal.Sandbox.list(tags=SANDBOX_TAG).
        _accepts(self, self.modal.Sandbox.list, ["tags"])

    def test_image_builder_methods_exist(self):
        Image = self.modal.Image
        self.assertTrue(hasattr(Image, "from_registry"))
        img = Image.from_registry("continuumio/miniconda3")
        for name in ("apt_install", "pip_install", "add_local_dir",
                     "run_function", "env", "workdir"):
            self.assertTrue(hasattr(img, name),
                            f"modal.Image has no '{name}' (we chain it)")

    def test_add_local_dir_accepts_our_kwargs(self):
        img = self.modal.Image.from_registry("continuumio/miniconda3")
        _accepts(self, img.add_local_dir, ["remote_path", "copy", "ignore"])

    def test_run_function_accepts_args_and_gpu(self):
        img = self.modal.Image.from_registry("continuumio/miniconda3")
        _accepts(self, img.run_function, ["args", "gpu"])


@unittest.skipUnless(HAVE_MODAL, _SKIP)
class TestModalAppRealValues(unittest.TestCase):
    """With modal present, import the REAL modal_app and assert the values the
    other layers assume by constant."""

    def setUp(self):
        # Force a clean real import (Layer-2 may have left a stub behind when run
        # in the same process via discovery).
        for m in list(sys.modules):
            if m == "modal_app" or m.startswith("modal_app."):
                del sys.modules[m]
        import modal
        self.modal = modal
        self.modal_app = importlib.import_module("modal_app")

    def test_constants_match_other_layers(self):
        self.assertEqual(self.modal_app.REMOTE_ROOT, EXPECT_REMOTE_ROOT)
        self.assertEqual(self.modal_app.MANIFEST_DIR, EXPECT_MANIFEST_DIR)
        self.assertEqual(self.modal_app.SANDBOX_TAG, {"fml_bench": "fml-bench-eval"})
        self.assertEqual(self.modal_app.APP_NAME, "fml-bench-eval")

    def test_app_is_modal_app_instance(self):
        self.assertIsInstance(self.modal_app.app, self.modal.App)

    def test_task_volumes_default_empty(self):
        self.assertEqual(self.modal_app.task_volumes("anything"), {})

    def test_gpu_map_a100_default_and_cpu_only(self):
        from modal_app import gpu_map
        self.assertEqual(gpu_map.DEFAULT_GPU, EXPECT_DEFAULT_GPU)
        # Unlisted GPU task -> A100 default.
        self.assertEqual(self.modal_app.task_gpu("Unlearning_open_unlearning"), "A100")
        # CPU-only tasks -> no GPU requested (causalml is CPU-only by project choice).
        self.assertIsNone(self.modal_app.task_gpu("Fairness_fairlearn"))
        self.assertIsNone(self.modal_app.task_gpu("Causality_gcastle"))
        self.assertIsNone(self.modal_app.task_gpu("Causality_causalml"))


@unittest.skipUnless(HAVE_MODAL, _SKIP)
class TestProcessAndTimeoutContract(unittest.TestCase):
    """Locks the process-handle / timeout contract Option B relies on.

    Discovered via introspection (modal 1.5 ContainerProcess has NO per-process
    kill); see ModalExecutor._sb_spawn / _sb_exec / _exec_timeout_exc /
    kill_running_process.
    """

    def test_exec_accepts_server_side_timeout(self):
        # _sb_spawn passes timeout= (and workdir=, env=) to Sandbox.exec.
        import modal
        _accepts(self, modal.Sandbox.exec, ["timeout", "workdir", "env"])

    def test_exec_timeout_error_importable_where_we_catch_it(self):
        # _exec_timeout_exc() does `from modal.exception import ExecTimeoutError`.
        from modal.exception import ExecTimeoutError
        self.assertTrue(issubclass(ExecTimeoutError, BaseException))

    def test_container_process_has_wait_and_streams(self):
        # _sb_wait reads proc.wait(), proc.stdout.read(), proc.stderr.read(),
        # proc.returncode.
        from modal.container_process import ContainerProcess
        for attr in ("wait", "poll", "returncode", "stdout", "stderr"):
            self.assertTrue(hasattr(ContainerProcess, attr),
                            f"ContainerProcess missing '{attr}' (_sb_wait uses it)")

    def test_container_process_has_no_per_process_kill(self):
        # Tripwire: the REASON kill_running_process tears the whole sandbox down
        # is that modal 1.5 exposes no per-process kill. If a future modal adds
        # one, this fails -> revisit kill_running_process to kill just the proc.
        from modal.container_process import ContainerProcess
        for attr in ("terminate", "kill", "signal", "send_signal"):
            self.assertFalse(
                hasattr(ContainerProcess, attr),
                f"ContainerProcess now has '{attr}'; revisit kill_running_process "
                "(it could kill the process instead of the whole sandbox)",
            )

    def test_stream_reader_has_read(self):
        from modal.io_streams import StreamReader
        self.assertTrue(hasattr(StreamReader, "read"))

    def test_sandbox_terminate_is_the_kill_primitive(self):
        # The only kill available: Sandbox.terminate (used by _teardown_sandbox,
        # kill_running_process, reap).
        import modal
        self.assertTrue(hasattr(modal.Sandbox, "terminate"))

    def test_connection_error_importable(self):
        # _transient_modal_errors() retries/reconnects on this (the observed
        # "Could not connect to the Modal server").
        from modal.exception import ConnectionError as ModalConnErr
        self.assertTrue(issubclass(ModalConnErr, BaseException))

    def test_sandbox_from_id_exists_for_reconnect(self):
        # (b1) _reconnect_sandbox() reconnects to a running sandbox by id.
        import modal
        self.assertTrue(hasattr(modal.Sandbox, "from_id"))
        _accepts(self, modal.Sandbox.from_id, ["sandbox_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
