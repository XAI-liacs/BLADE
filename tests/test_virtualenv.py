import importlib.util
import sys
import types
from pathlib import Path

# Minimal stubs for dependencies required by problem/solution modules
np_stub = types.ModuleType("numpy")
np_stub.Inf = float("inf")
sys.modules["numpy"] = np_stub

joblib_stub = types.ModuleType("joblib")
externals_stub = types.ModuleType("externals")
loky_stub = types.ModuleType("loky")


def _dummy_executor(*args, **kwargs):
    pass


loky_stub.get_reusable_executor = _dummy_executor
externals_stub.loky = loky_stub
joblib_stub.externals = externals_stub
sys.modules["joblib"] = joblib_stub
sys.modules["joblib.externals"] = externals_stub
sys.modules["joblib.externals.loky"] = loky_stub

ioh_stub = types.ModuleType("ioh")


class DummyLogInfo:
    pass


logger_mod = types.ModuleType("logger")


class AbstractLogger:
    pass


logger_mod.AbstractLogger = AbstractLogger
ioh_stub.LogInfo = DummyLogInfo
ioh_stub.logger = logger_mod
sys.modules["ioh"] = ioh_stub

cs = types.ModuleType("ConfigSpace")
cs.ConfigurationSpace = type("ConfigurationSpace", (), {})
sys.modules["ConfigSpace"] = cs

tc = types.ModuleType("tokencost")
for fn in [
    "calculate_completion_cost",
    "calculate_prompt_cost",
    "count_message_tokens",
    "count_string_tokens",
]:
    setattr(tc, fn, lambda *a, **k: 0)
sys.modules["tokencost"] = tc

# Set up a lightweight iohblade package without running its __init__
repo_root = Path(__file__).resolve().parents[1]
iohblade_pkg = types.ModuleType("iohblade")
iohblade_pkg.__path__ = [str(repo_root / "iohblade")]
sys.modules["iohblade"] = iohblade_pkg

spec_solution = importlib.util.spec_from_file_location(
    "iohblade.solution", repo_root / "iohblade" / "solution.py"
)
solution_mod = importlib.util.module_from_spec(spec_solution)
sys.modules["iohblade.solution"] = solution_mod
spec_solution.loader.exec_module(solution_mod)
iohblade_pkg.solution = solution_mod

spec_problem = importlib.util.spec_from_file_location(
    "iohblade.problem", repo_root / "iohblade" / "problem.py"
)
problem_mod = importlib.util.module_from_spec(spec_problem)
sys.modules["iohblade.problem"] = problem_mod
spec_problem.loader.exec_module(problem_mod)
iohblade_pkg.problem = problem_mod

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
    # Avoid installing heavy base packages
    monkeypatch.setattr("iohblade.problem.BASE_DEPENDENCIES", [])

    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "setup.py").write_text(
        "from setuptools import setup; setup(name='mypkg', version='0.0.0')"
    )
    package_code = "def hello():\n    return 'hi'\n"
    (pkg_dir / "mypkg").mkdir()
    (pkg_dir / "mypkg" / "__init__.py").write_text(package_code)

    assert importlib.util.find_spec("mypkg") is None

    problem = DummyDepProblem(pkg_dir)
    sol = Solution()
    result = problem(sol)

    assert result.fitness == 2
    assert importlib.util.find_spec("mypkg") is None
