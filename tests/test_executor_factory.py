"""
Tests for benchmark.executor_factory.make_executor.

These verify the eval-backend selection contract (the local path must never
import modal). They pass whether or not modal is installed:
  - local / unset backend returns a plain BenchmarkExecutor (not ModalExecutor)
  - "modal" backend triggers the lazy import: ImportError when modal is ABSENT
    (proving the local path is independent of modal), else a ModalExecutor
  - importing the factory module does not import modal

Runnable with stdlib unittest (pytest is not available):
    python tests/test_executor_factory.py
"""
import os
import os.path as osp
import sys
import unittest

# Ensure the project root is importable regardless of cwd.
PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _minimal_config():
    # BenchmarkExecutor.__init__ only stores args and reads config["repo_dir"]
    # and config["conda_env"]; a minimal dict suffices.
    return {"repo_dir": "x", "conda_env": "y"}


class TestMakeExecutor(unittest.TestCase):
    def test_local_returns_benchmark_executor(self):
        from benchmark.executor_factory import make_executor
        from benchmark.executor import BenchmarkExecutor

        ex = make_executor(_minimal_config(), "agent", "bench", "exp",
                           eval_backend="local")
        self.assertIsInstance(ex, BenchmarkExecutor)
        # It must be a plain BenchmarkExecutor, not a ModalExecutor subclass.
        self.assertEqual(type(ex), BenchmarkExecutor)

    def test_default_returns_benchmark_executor(self):
        from benchmark.executor_factory import make_executor
        from benchmark.executor import BenchmarkExecutor

        ex = make_executor(_minimal_config(), "agent", "bench", "exp")
        self.assertIsInstance(ex, BenchmarkExecutor)
        self.assertEqual(type(ex), BenchmarkExecutor)

    def test_modal_backend_imports_modal_executor(self):
        import importlib.util
        from benchmark.executor_factory import make_executor

        # A new-format config so ModalExecutor.__init__ passes its config check
        # and reaches the lazy modal_app import (which imports modal).
        config = _minimal_config()
        config["val_command"] = "echo val"

        if importlib.util.find_spec("modal") is None:
            # modal ABSENT (the recommended env for the R1 layers): the lazy
            # `from benchmark.modal_executor import ModalExecutor` must fail, proving
            # the modal path is opt-in and the local path is independent of modal.
            with self.assertRaises((ImportError, ModuleNotFoundError)):
                make_executor(config, "agent", "bench", "exp", eval_backend="modal")
        else:
            # modal INSTALLED (e.g. the operator's live env): the lazy import
            # succeeds and a ModalExecutor is returned. The factory does the right
            # thing either way — assert it so this test passes in BOTH envs.
            ex = make_executor(config, "agent", "bench", "exp", eval_backend="modal")
            self.assertEqual(type(ex).__name__, "ModalExecutor")

    def test_factory_import_does_not_import_modal(self):
        # Drop any cached factory module so the import is exercised fresh.
        for mod in ("benchmark.executor_factory", "modal"):
            sys.modules.pop(mod, None)

        import benchmark.executor_factory  # noqa: F401

        self.assertNotIn("modal", sys.modules)


if __name__ == "__main__":
    unittest.main()
