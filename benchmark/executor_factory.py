"""
Executor factory: selects the eval backend (local vs. Modal).

Keeps Modal SDK imports off the local path. When eval_backend != "modal",
behavior is exactly as today (returns a plain BenchmarkExecutor) and the
modal package is never imported.
"""
from benchmark.executor import BenchmarkExecutor


def make_executor(*args, eval_backend="local", **kwargs):
    if eval_backend == "modal":
        from benchmark.modal_executor import ModalExecutor  # lazy: modal never imported on the local path
        return ModalExecutor(*args, **kwargs)
    if eval_backend != "local":
        # Fail loud on a misconfigured backend (e.g. a typo or bad
        # FMLBENCH_EVAL_BACKEND value) instead of silently running locally.
        raise ValueError(
            f"Unknown eval_backend {eval_backend!r}; expected 'local' or 'modal'."
        )
    return BenchmarkExecutor(*args, **kwargs)
