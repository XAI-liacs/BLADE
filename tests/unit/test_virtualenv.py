from pathlib import Path

from iohblade.problem import Problem
from iohblade.solution import Solution


class DummyDepProblem(Problem):
    def __init__(self, pkg_path):
        super().__init__(dependencies=[str(pkg_path)])

    def get_prompt(self):
        return ""

    def evaluate(self, s):
        import mypkg  # type: ignore

        s.set_scores(len(mypkg.hello()))
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}


def test_dependencies_installed_in_virtualenv(tmp_path, monkeypatch):
    # ensure only essential dependency is installed in the evaluation env
    monkeypatch.setattr(
        "iohblade.problem.BASE_DEPENDENCIES",
        ["cloudpickle", "numpy", "joblib", "ioh"],
        raising=False,
    )

    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "setup.py").write_text(
        "from setuptools import setup; setup(name='mypkg', version='0.0.0')"
    )
    (pkg_dir / "mypkg").mkdir()
    (pkg_dir / "mypkg" / "__init__.py").write_text("def hello():\n    return 'hi'\n")

    problem = DummyDepProblem(pkg_dir)
    sol = Solution()
    result = problem(sol)

    assert result.fitness == 2, result.feedback
    # after the call, the dep should not leak back to the outer environment
    import importlib

    assert importlib.util.find_spec("mypkg") is None
