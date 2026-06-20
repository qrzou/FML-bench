"""
Tests for benchmark.executor_factory.make_executor.

These verify the eval-backend selection contract WITHOUT the modal package
installed (the local path must never import modal):
  - local / unset backend returns a plain BenchmarkExecutor (not ModalExecutor)
  - "modal" backend triggers the lazy import (ImportError here, since modal
    is not installed), proving the local path is independent of modal
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

    def test_modal_backend_raises_without_modal(self):
        from benchmark.executor_factory import make_executor

        # A new-format config so ModalExecutor.__init__ passes its config check
        # and reaches the lazy modal_app import (which imports modal).
        config = _minimal_config()
        config["val_command"] = "echo val"
        # modal is not installed here, so the lazy import must fail.
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            make_executor(config, "agent", "bench", "exp",
                          eval_backend="modal")

    def test_factory_import_does_not_import_modal(self):
        # Drop any cached factory module so the import is exercised fresh.
        for mod in ("benchmark.executor_factory", "modal"):
            sys.modules.pop(mod, None)

        import benchmark.executor_factory  # noqa: F401

        self.assertNotIn("modal", sys.modules)


if __name__ == "__main__":
    unittest.main()
