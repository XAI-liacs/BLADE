"""Regression tests for the smaller core and benchmark bugs of issue #209.

Each section names the item of the issue it covers.
"""

import copy
import pickle
import sys
import types

import numpy as np
import pytest

import iohblade.problem as problem_module
from iohblade.mcts_node import MCTS_Node
from iohblade.problem import (
    WrappedProblem,
    evaluate_in_subprocess,
    simplify_subprocess_error,
    wrap_problem,
)
from iohblade.solution import Solution

# The AutoML and photonics fixtures stub optional modules (openml, smac,
# PyMoosh, pyGDM2) that may be missing or broken in this environment.
from .test_benchmark_scores_bbob import automl_module  # noqa: F401
from .test_benchmark_scores_bbob import photonics_module  # noqa: F401

# --------------------------------------------------------------------------
# 1. Solution.set_scores
# --------------------------------------------------------------------------


def _raise_in_candidate(code, name="Candidate"):
    """Run ``code`` like a benchmark does and return the raised exception."""
    ns = {"np": np}
    try:
        exec(compile(code, name, "exec"), ns, ns)
    except Exception as e:  # noqa: BLE001 - we want the exception itself
        return e
    raise AssertionError("code did not raise")


def test_set_scores_has_a_single_definition():
    import inspect

    import iohblade.solution as solution_module

    source = inspect.getsource(solution_module.Solution)
    assert source.count("def set_scores(") == 1


def test_set_scores_error_raised_outside_the_candidate_code():
    # A validity check of the benchmark itself, not candidate code (raised
    # explicitly: pytest rewrites assert statements in test modules).
    try:
        raise AssertionError("colouring is invalid")
    except AssertionError as e:
        err = e
    s = Solution(code="x = 1\n", name="Candidate")
    s.set_scores(-np.inf, "bad", err)
    assert s.error == "AssertionError: colouring is invalid"
    assert "On line" not in s.error


def test_set_scores_error_without_traceback():
    s = Solution(code="x = 1\n")
    s.set_scores(-np.inf, "bad", ValueError("not a number"))
    assert s.error == "ValueError: not a number"


def test_set_scores_error_raised_inside_numpy_points_at_candidate_line():
    code = "import numpy as np\nnp.linalg.inv(np.zeros((2, 2)))\n"
    err = _raise_in_candidate(code)
    # the last frame is in numpy, the candidate frame is further up
    import traceback

    assert traceback.extract_tb(err.__traceback__)[-1].filename != "Candidate"
    s = Solution(code=code, name="Candidate")
    s.set_scores(-np.inf, "bad", err)
    assert s.error.startswith("LinAlgError: Singular matrix")
    assert "On line 2: np.linalg.inv(np.zeros((2, 2)))" in s.error


def test_set_scores_error_in_candidate_code_with_string_filename():
    code = "a = 1\nb = a / 0\n"
    err = _raise_in_candidate(code, name="<string>")
    s = Solution(code=code, name="")
    s.set_scores(-np.inf, "bad", err)
    assert s.error.startswith("ZeroDivisionError: division by zero")
    assert "On line 2: b = a / 0" in s.error


def test_set_scores_clears_a_stale_error():
    s = Solution()
    s.set_scores(-np.inf, "bad", ValueError("boom"))
    assert s.error
    s.set_scores(1.0, "good")
    assert s.error == ""
    s.set_scores(-np.inf, "bad", "a message")
    assert s.error == "a message"
    s.set_scores(2.0, "good", "")
    assert s.error == ""
    s.set_scores(2.0, "good", None)
    assert s.error == ""


# --------------------------------------------------------------------------
# 2. simplify_subprocess_error and errors after the script finished
# --------------------------------------------------------------------------

_NUMPY_STDERR = """Traceback (most recent call last):
  File "/tmp/env/abc_run_eval.py", line 40, in <module>
    result=problem.evaluate(solution)
  File "/repo/iohblade/benchmarks/foo.py", line 120, in evaluate
    exec(code, ns)
  File "<string>", line 3, in <module>
  File "<string>", line 7, in __call__
  File "/env/numpy/linalg/_linalg.py", line 104, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
"""


def test_simplify_error_uses_last_candidate_frame():
    code = "\n".join(f"line{i}" for i in range(1, 10))
    sol = Solution(code=code, name="Algo")
    msg = simplify_subprocess_error(_NUMPY_STDERR, sol)
    assert "line 7, in __call__" in msg
    assert "On line: line7" in msg
    assert "numpy.linalg.LinAlgError: Singular matrix" in msg


def test_simplify_error_uses_solution_name_as_filename():
    stderr = _NUMPY_STDERR.replace('"<string>"', '"Algo"')
    sol = Solution(code="a\nb\nc\nd\ne\nf\ng\n", name="Algo")
    assert "On line: g" in simplify_subprocess_error(stderr, sol)


def test_simplify_error_without_candidate_frame_omits_the_line():
    stderr = """Traceback (most recent call last):
  File "/tmp/env/abc_run_eval.py", line 2, in <module>
    result=problem.evaluate(solution)
  File "/env/numpy/linalg/_linalg.py", line 1, in inv
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
"""
    sol = Solution(code="def __call__(self):\n    pass\n", name="Algo")
    msg = simplify_subprocess_error(stderr, sol)
    assert msg == "numpy.linalg.LinAlgError: Singular matrix"
    assert "def __call__" not in msg


def test_simplify_error_empty_and_unparsable():
    assert simplify_subprocess_error("") == "Unknown error."
    assert simplify_subprocess_error("Killed\n") == "Killed"


class _Conn:
    def __init__(self):
        self.sent = []

    def send(self, item):
        self.sent.append(item)

    def close(self):
        pass


def test_error_after_the_script_is_not_mapped_on_candidate_code(monkeypatch):
    from .harness_problems import GoodProblem

    problem = GoodProblem()
    problem._ensure_env()
    problem.eval_timeout = 120

    def broken_load(f):
        raise EOFError("Ran out of input")

    # The evaluation script succeeds, reading its result fails.
    monkeypatch.setattr(problem_module.cloudpickle, "load", broken_load)
    conn = _Conn()
    code = "\n".join(f"candidate line {i}" for i in range(1, 400))
    evaluate_in_subprocess(problem, conn, Solution(code=code, name="Algo"))
    assert conn.sent == [
        {"error": "EOFError: Ran out of input", "stdout": "", "stderr": ""}
    ]


# --------------------------------------------------------------------------
# 3. aoc_logger(scale_log=False)
# --------------------------------------------------------------------------


def test_aoc_logger_without_log_scale():
    from iohblade.utils import aoc_logger, correct_aoc

    log = aoc_logger(10, lower=0.0, upper=10.0, scale_log=False)
    assert log.transform(5.0) == 5.0
    log(types.SimpleNamespace(evaluations=1, raw_y_best=5.0))
    assert log.aoc == pytest.approx(0.5)
    func = types.SimpleNamespace(
        state=types.SimpleNamespace(
            current_best_internal=types.SimpleNamespace(y=2.0), evaluations=1
        )
    )
    # 1 logged evaluation at 0.5, 9 remaining evaluations at 2/10
    assert correct_aoc(func, log, 10) == pytest.approx(1 - (0.5 + 9 * 0.2) / 10)
    pickle.loads(pickle.dumps(log.transform))  # no lambda


def test_aoc_logger_log_scale_unchanged():
    from iohblade.utils import aoc_logger

    log = aoc_logger(10)
    assert log.transform(100.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 4. WrappedProblem signatures and timeout default
# --------------------------------------------------------------------------


def _score(solution, value):
    solution.set_scores(value, "ok")
    return solution


def test_wrapped_problem_solution_with_optional_argument():
    def evaluate(solution, logger=None):
        assert logger is None
        return _score(solution, 1.0)

    p = wrap_problem(evaluate)
    assert p.evaluate(Solution()).fitness == 1.0


def test_wrapped_problem_self_and_problem_signatures():
    def with_self(self, solution):
        assert isinstance(self, WrappedProblem)
        return _score(solution, 2.0)

    def with_problem(problem, solution, extra=3.0):
        assert isinstance(problem, WrappedProblem)
        return _score(solution, extra)

    def only_solution(solution):
        return _score(solution, 4.0)

    assert wrap_problem(with_self).evaluate(Solution()).fitness == 2.0
    assert wrap_problem(with_problem).evaluate(Solution()).fitness == 3.0
    assert wrap_problem(only_solution).evaluate(Solution()).fitness == 4.0


def test_wrapped_problem_timeout_defaults_match():
    p1 = wrap_problem(lambda solution: solution)
    p2 = WrappedProblem(lambda solution: solution)
    assert p1.eval_timeout == p2.eval_timeout == 6000


# --------------------------------------------------------------------------
# 5. Solution defaults and round trips
# --------------------------------------------------------------------------


def test_solution_parent_ids_default_is_not_shared():
    a = Solution()
    b = Solution()
    a.parent_ids.append("x")
    assert b.parent_ids == []
    assert Solution().parent_ids == []


def test_solution_dict_round_trip_keeps_task_prompt():
    s = Solution(code="x = 1", name="A", task_prompt="Solve it", parent_ids=["p"])
    s.set_scores(1.5, "fine", "")
    restored = Solution()
    restored.from_dict(s.to_dict())
    assert restored.task_prompt == "Solve it"
    assert restored.to_dict() == s.to_dict()


def test_solution_pickle_round_trip_keeps_task_prompt():
    s = Solution(code="x = 1", name="A", task_prompt="Solve it", parent_ids=["p"])
    s.set_scores(1.5, "fine")
    restored = pickle.loads(pickle.dumps(s))
    assert restored.task_prompt == "Solve it"
    assert restored.to_dict() == s.to_dict()


# --------------------------------------------------------------------------
# 6. MCTS_Node pickling and copying
# --------------------------------------------------------------------------


def _tree():
    root = MCTS_Node(Solution(code="root"), "root")
    child = MCTS_Node(Solution(code="child"), "i1", depth=1, visit=3, Q=0.5)
    grandchild = MCTS_Node(Solution(code="gc"), "m1", depth=2, Q=0.25)
    root.add_child(child)
    child.add_child(grandchild)
    return root, child, grandchild


def _check_tree(root):
    assert root.approach == "root" and root.parent is None
    (child,) = root.children
    assert child.parent is root
    assert (child.approach, child.depth, child.visit, child.Q) == ("i1", 1, 3, 0.5)
    assert child.code == "child"
    assert child.parent_ids == [root.id]
    (grandchild,) = child.children
    assert grandchild.parent is child
    assert (grandchild.approach, grandchild.depth, grandchild.Q) == ("m1", 2, 0.25)
    assert grandchild.children == []
    assert isinstance(grandchild, MCTS_Node)


@pytest.mark.parametrize(
    "clone",
    [
        copy.deepcopy,
        lambda n: pickle.loads(pickle.dumps(n)),
    ],
    ids=["deepcopy", "pickle"],
)
def test_mcts_tree_survives_copy(clone):
    root, child, _ = _tree()
    new_root = clone(root)
    assert new_root is not root
    _check_tree(new_root)
    # copying a node in the middle of the tree keeps its links both ways
    new_child = clone(child)
    assert new_child.parent.children[0] is new_child
    assert new_child.parent.id == root.id


def test_mcts_node_parent_ids_is_a_list():
    from iohblade.plots import safe_parse_parent_ids

    root, child, _ = _tree()
    assert root.parent_ids == []
    assert child.parent_ids == [root.id]
    node = MCTS_Node(Solution(parent_ids=["a", "b"]), "e1")
    assert node.parent_ids == ["b"]
    assert safe_parse_parent_ids(child.parent_ids) == [root.id]
    assert safe_parse_parent_ids(root.id) == [root.id]  # old logs


# --------------------------------------------------------------------------
# 7. Kerneltuner test() and cache path
# --------------------------------------------------------------------------


def test_kerneltuner_test_and_cache_dir(monkeypatch, tmp_path):
    from iohblade.benchmarks.kerneltuner import kerneltuner as kt_module

    captured = {}

    def fake_generate(name, path, strategies, applications, gpus, **kwargs):
        captured["override"] = kwargs["override"]
        captured["applications"] = applications
        return path / "experiment.json"

    monkeypatch.setattr(kt_module, "generate_experiment_file", fake_generate)
    monkeypatch.setattr(
        kt_module, "get_strategy_scores", lambda path: {"s": {"score": 0.75}}
    )
    cache_dir = str(tmp_path / "cache") + "/"
    problem = kt_module.Kerneltuner(
        gpus=["A100"], kernels=["gemm"], cache_dir=cache_dir, imports="import os\n"
    )
    problem.logger_dir = str(tmp_path)
    sol = Solution(
        code="class MyTuner:\n    def __init__(self):\n        pass\n",
        name="MyTuner",
    )
    result = problem.test(sol)
    assert result.fitness == 0.75
    regex = captured["override"]["experimental_groups_defaults"][
        "pattern_for_full_search_space_filenames"
    ]["regex"]
    assert regex.startswith(cache_dir)
    assert regex.endswith("cachefiles/${applications}/${gpus}_T4.json")
    assert "/data/neocortex" not in regex
    assert problem.imports == "import os\n"


# --------------------------------------------------------------------------
# 8. imports passed through to Problem
# --------------------------------------------------------------------------


def test_bbob_problems_keep_imports():
    from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX
    from iohblade.benchmarks.BBOB.mabbob import MA_BBOB

    assert BBOB_SBOX(imports="import scipy\n").imports == "import scipy\n"
    assert MA_BBOB(imports="import scipy\n").imports == "import scipy\n"
    # the benchmark's own default is used when none is given
    assert "import ioh" in BBOB_SBOX().imports
    assert "import ioh" in MA_BBOB().imports


def test_photonics_keeps_imports(photonics_module):  # noqa: F811
    p = photonics_module.Photonics(None, imports="import scipy\n")
    assert p.imports == "import scipy\n"
    assert "import ioh" in photonics_module.Photonics(None).imports


# --------------------------------------------------------------------------
# 9. AutoML.get_config without an OpenML task
# --------------------------------------------------------------------------


def test_automl_get_config_without_task(automl_module):  # noqa: F811
    problem = automl_module.AutoML(openml_task_id=None)
    config = problem.get_config()
    assert config["config"]["task_id"] is None
    assert config["config"]["task_type"] is None


def test_automl_get_config_with_task(automl_module):  # noqa: F811
    problem = automl_module.AutoML(openml_task_id=None)
    problem.task = types.SimpleNamespace(
        task_type="Supervised Classification", class_labels=["a", "b"]
    )
    config = problem.get_config()
    assert config["config"]["task_type"] == "classification"
    assert config["tags"][-2:] == ["a", "b"]


# --------------------------------------------------------------------------
# 10. TSP best known distance
# --------------------------------------------------------------------------

_TSP_SOLVER = """
class Solver:
    def __init__(self, customers):
        self.customers = customers

    def __call__(self):
        return [c[0] for c in self.customers]
"""


def test_tsp_best_known_is_updated_and_reported():
    from iohblade.benchmarks.logistics import TravelingSalesmanProblem

    tsp = TravelingSalesmanProblem("A-n53-k7")
    sol = tsp.evaluate(Solution(code=_TSP_SOLVER, name="Solver"))
    assert np.isfinite(sol.fitness)
    assert tsp.best_known == sol.fitness
    assert "inf" not in sol.feedback
    assert f"best known distance is {sol.fitness}" in sol.feedback


def test_tsp_feedback_omits_unknown_best():
    from iohblade.benchmarks.logistics import TravelingSalesmanProblem

    tsp = TravelingSalesmanProblem("A-n53-k7")
    bad = tsp.evaluate(Solution(code="x = 1", name="Solver"))
    assert bad.fitness == float("inf")
    assert tsp.best_known == float("inf")
    assert "inf" not in bad.feedback.replace("info", "")


def test_tsp_best_known_survives_subprocess_evaluation(monkeypatch):
    from iohblade.benchmarks.logistics import TravelingSalesmanProblem

    fitnesses = iter([120.0, float("inf"), 100.0, 130.0])

    def fake_call(self, solution, logger=None):
        # Like the real evaluation: the subprocess works on a copy.
        result = copy.deepcopy(solution)
        result.set_scores(next(fitnesses), "")
        return result

    monkeypatch.setattr(problem_module.Problem, "__call__", fake_call)
    tsp = TravelingSalesmanProblem("A-n53-k7")
    seen = []
    for _ in range(4):
        tsp(Solution())
        seen.append(tsp.best_known)
    assert seen == [120.0, 120.0, 100.0, 100.0]


# --------------------------------------------------------------------------
# 11. / 12. Steiner tree name and graph colouring prompt
# --------------------------------------------------------------------------


def test_steiner_tree_name_uses_benchmark_id():
    from iohblade.benchmarks.combinatorics import EuclidianSteinerTree

    assert EuclidianSteinerTree(10).name == "EuclidianSteinerTree-n10"
    assert EuclidianSteinerTree(20).name == "EuclidianSteinerTree-n20"


def test_graph_coloring_prompt_gives_the_number_of_nodes():
    from iohblade.benchmarks.combinatorics import GraphColoring

    gc = GraphColoring("1")
    n = len(gc.nodes)
    assert f"A list of nodes id'd 1...{n}." in gc.task_prompt
    assert str(gc.nodes) not in gc.task_prompt


# --------------------------------------------------------------------------
# 13. misc.ast print_results / process_file_paths
# --------------------------------------------------------------------------


def test_process_file_paths_prints_stats(tmp_path, capsys):
    from iohblade.misc.ast import analyze_graph, print_results, process_file_paths

    path = tmp_path / "algo.py"
    path.write_text(
        "def f(x):\n"
        "    if x > 0:\n"
        "        return x * 2\n"
        "    return -x\n"
        "\n"
        "class A:\n"
        "    def g(self):\n"
        "        return [f(i) for i in range(3)]\n"
    )
    results = process_file_paths([str(path)], visualize=False)
    out = capsys.readouterr().out
    assert len(results) == 1
    assert f"Statistics for file: {path}" in out
    for key in ("Nodes", "Edges", "Max Depth", "Mean Clustering"):
        assert f"{key}:" in out
        assert key in results[0]
    print_results({"Nodes": 3}, "snippet")
    assert "Nodes: 3" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 14. prepare_namespace
# --------------------------------------------------------------------------


def test_prepare_namespace_skips_star_and_relative_imports():
    from iohblade.misc.prepare_namespace import prepare_namespace

    code = "from math import *\nfrom . import sibling\nfrom .pkg import thing\n"
    assert prepare_namespace(code, ["numpy"]) == {}


@pytest.mark.parametrize(
    "requirement",
    ["numpy==2.0", "numpy<3", "numpy>=2", "numpy~=2.1", "numpy!=1.0", "numpy[extra]"],
)
def test_prepare_namespace_strips_version_specifiers(requirement):
    from iohblade.misc.prepare_namespace import prepare_namespace

    ns = prepare_namespace("import numpy as np\n", [requirement])
    assert ns["np"] is np


def test_prepare_namespace_still_blocks_other_modules():
    from iohblade.misc.prepare_namespace import prepare_namespace

    ns = prepare_namespace("import json\nfrom numpy import linalg\n", ["scipy>=1"])
    assert ns["json"] is object
    assert ns["linalg"] is object
    ns = prepare_namespace("from numpy import linalg\n", ["numpy>=2"])
    assert ns["linalg"] is np.linalg


def test_mcts_simulate_does_not_send_the_tree_to_the_problem():
    import pickle

    from iohblade.mcts_node import MCTS_Node
    from iohblade.methods.mcts_ahd.mcts import MCTS

    sizes = []

    class _Recording:
        minimisation = False

        def __call__(self, solution, logger=None):
            sizes.append(len(pickle.dumps(solution)))
            assert solution.parent is None and solution.children == []
            return solution.set_scores(1.0)

    mcts = MCTS.__new__(MCTS)
    mcts.problem = _Recording()
    mcts.eval_remain = 5
    mcts.maximisation = True
    mcts.q_min = mcts.q_max = None
    root = MCTS_Node(Solution(code="x"), approach="root")
    mcts.best_solution = root
    root.fitness = float("-inf")
    parent = MCTS_Node(Solution(code="p" * 10000), approach="m1")
    root.add_child(parent)
    leaf = MCTS_Node(Solution(code="y"), approach="m1")
    parent.add_child(leaf)

    evaluated = mcts.simulate(leaf)
    assert sizes and sizes[0] < 5000  # not the 10 kB parent
    assert evaluated.parent is parent  # tree links restored
    assert evaluated.Q == 1.0
