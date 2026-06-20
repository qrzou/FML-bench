"""
Layer-1 test: the LOCAL eval path is byte-for-byte unaffected by the Modal work
(requirement R1). No modal package, no network.

Three guards:
  A. The files that define the original local behavior are unchanged vs the
     branch's fork point (merge-base with main). If any of these drift, the
     "running the original README way is identical" guarantee is broken.
  B. With ``modal`` BANNED from import, the factory's local path still builds a
     plain BenchmarkExecutor and ``modal`` never enters sys.modules — proven in
     a fresh subprocess so nothing else can have imported it.
  C. Every agent is wired through make_executor with an eval_backend hook (a
     static source check; importing all agents would drag in heavy optional
     deps unrelated to this contract).

Run with stdlib unittest:
    python tests/test_local_path_unaffected.py
"""
import os.path as osp
import subprocess
import sys
import unittest

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Files that encode the original local behavior. R2's "WITH == WITHOUT Modal"
# rests on these being untouched; R1's "local path identical" rests on it too.
PROTECTED_FILES = [
    "benchmark/executor.py",
    "agents/base.py",
    "agents/code_editor.py",
    "agents/llm.py",
    "setup.py",
]

# The 7 agents that construct executors, and must route through the factory.
AGENT_FILES = [
    "agents/adaptivesearch/agent.py",
    "agents/ai_scientist_v2/agent.py",
    "agents/aide/agent.py",
    "agents/aira_mcts/agent.py",
    "agents/autoresearch/agent.py",
    "agents/openevolve/agent.py",
    "agents/theaiscientist/theaiscientist.py",
]


def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)


class TestProtectedFilesUnchanged(unittest.TestCase):
    """Guard A: protected files identical to the fork point."""

    @classmethod
    def setUpClass(cls):
        # Baseline = merge-base(HEAD, <nearest base branch>). public_modal was
        # forked from `public`, so prefer it; fall back through main, then to
        # HEAD (which still catches uncommitted edits to the protected files).
        cls.base = None
        cls.base_ref = None
        for ref in ("public", "origin/public", "main", "origin/main"):
            mb = _git("merge-base", "HEAD", ref)
            if mb.returncode == 0 and mb.stdout.strip():
                cls.base = mb.stdout.strip()
                cls.base_ref = ref
                break
        if cls.base is None:
            head = _git("rev-parse", "HEAD")
            cls.base = head.stdout.strip() if head.returncode == 0 else None
            cls.base_ref = "HEAD"

    def test_protected_files_byte_identical_to_fork_point(self):
        if not self.base:
            self.skipTest("no git base ref (merge-base with main / HEAD) available")
        changed = []
        for path in PROTECTED_FILES:
            # Compare the WORKING TREE (committed + uncommitted) to the baseline.
            diff = _git("diff", "--quiet", self.base, "--", path)
            if diff.returncode != 0:
                changed.append(path)
        self.assertEqual(
            changed, [],
            f"Protected (local-path) files changed vs {self.base[:10]}: {changed}. "
            "R1 requires these stay byte-identical.",
        )


class TestLocalPathDoesNotImportModal(unittest.TestCase):
    """Guard B: the factory's local path never imports modal, even when modal
    is forcibly unavailable."""

    def test_local_backend_builds_without_modal(self):
        script = r"""
import sys
class _BanModal:
    def find_spec(self, name, path=None, target=None):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("modal is banned in the R1 local-path test")
        return None
sys.meta_path.insert(0, _BanModal())

from benchmark.executor_factory import make_executor
from benchmark.executor import BenchmarkExecutor

cfg = {"repo_dir": "x", "conda_env": "y"}
for backend in ("local", None):
    kw = {} if backend is None else {"eval_backend": backend}
    ex = make_executor(cfg, "a", "b", "c", **kw)
    assert type(ex) is BenchmarkExecutor, ("wrong type for backend=%r: %r" %
                                           (backend, type(ex)))
assert "modal" not in sys.modules, "local path imported modal!"
print("R1_OK")
"""
        r = subprocess.run([sys.executable, "-c", script],
                           cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0,
                         f"subprocess failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}")
        self.assertIn("R1_OK", r.stdout)


class TestAgentsWiredThroughFactory(unittest.TestCase):
    """Guard C: every agent routes executor construction through the factory
    with an eval_backend hook (static source check)."""

    def test_all_agents_reference_make_executor_and_eval_backend(self):
        missing = []
        for rel in AGENT_FILES:
            path = osp.join(PROJECT_ROOT, rel)
            self.assertTrue(osp.isfile(path), f"missing agent file: {rel}")
            with open(path, "r") as f:
                src = f.read()
            if "make_executor" not in src or "eval_backend" not in src:
                missing.append(rel)
        self.assertEqual(
            missing, [],
            f"agents not wired through make_executor(eval_backend=...): {missing}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
