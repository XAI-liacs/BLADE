"""Multi-objective benchmarks (pymoo ZDT/DTLZ/WFG), issue #257."""

import math

import numpy as np
import pytest

from iohblade.benchmarks.multiobjective import (
    MultiObjectiveBenchmark,
    MultiObjectiveFunction,
    anytime_hypervolume,
    mo_dtlz,
    mo_quick,
    mo_wfg,
    mo_zdt,
)
from iohblade.benchmarks.multiobjective.mo_benchmark import (
    MO_DEPENDENCIES,
    _pymoo_problem,
    function_spec,
)
from iohblade.config import ConfigError, load_plan
from iohblade.solution import Solution
from iohblade.utils import OverBudgetException

RANDOM = """
import numpy as np

class RandomSearch:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        X = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        return func(X)
"""


# ---------------------------------------------------------------- specs


def test_function_spec_defaults_and_errors():
    assert function_spec("zdt1") == {"name": "zdt1"}
    assert function_spec("DTLZ2") == {"name": "dtlz2", "n_obj": 3}
    assert function_spec({"name": "dtlz2", "n_obj": 2}) == {"name": "dtlz2", "n_obj": 2}
    assert function_spec("wfg4") == {"name": "wfg4", "n_var": 10, "n_obj": 2}
    with pytest.raises(ValueError, match="Unknown multi-objective function"):
        function_spec("sphere")
    with pytest.raises(ValueError, match="needs a 'name'"):
        function_spec({"n_obj": 2})


# ---------------------------------------------------------------- func


def _func(name="zdt1", budget=10):
    func, _ = _func_and_record(name, budget)
    return func


def _func_and_record(name="zdt1", budget=10):
    return MultiObjectiveFunction.create(_pymoo_problem(function_spec(name)), budget)


def test_function_single_and_batch_calls_are_recorded():
    func, record = _func_and_record("zdt1", budget=10)
    assert func.meta_data.n_variables == 30 and func.meta_data.n_objectives == 2
    f = func(np.full(30, 0.5))
    assert f.shape == (2,)
    F = func(np.random.uniform(0, 1, (4, 30)))
    assert F.shape == (4, 2)
    assert func.state.evaluations == 5
    assert record.evaluations == 5
    assert np.allclose(record.objectives()[0], f)


def test_function_clips_to_bounds_and_checks_input():
    func = _func("zdt1", budget=10)
    inside = func(np.zeros(30))
    outside = func(np.full(30, -5.0))  # clipped to the lower bound 0
    assert np.allclose(inside, outside)
    with pytest.raises(ValueError, match="dimension 30"):
        func(np.zeros(3))
    with pytest.raises(ValueError, match="NaN or inf"):
        func(np.full(30, np.nan))


def test_function_enforces_the_budget_and_keeps_what_fits():
    func, record = _func_and_record("zdt1", budget=5)
    func(np.random.uniform(0, 1, (3, 30)))
    with pytest.raises(OverBudgetException):
        func(np.random.uniform(0, 1, (4, 30)))  # only 2 more fit
    assert func.state.evaluations == 5 and len(record.objectives()) == 5
    with pytest.raises(OverBudgetException):
        func(np.zeros(30))
    assert func.state.evaluations == 5


def test_function_hides_the_problem():
    func = _func()
    assert not hasattr(func, "problem")
    assert not hasattr(func, "pareto_front")


def test_candidate_cannot_reach_the_scoring_record():
    func, record = _func_and_record("zdt1", budget=10)
    # No attribute holds the record (review finding on #258).
    assert not any(hasattr(func, a) for a in ("_archive", "_record", "_count"))
    with pytest.raises(AttributeError):
        func._archive = []  # __slots__: no new attributes
    F = func(np.full(30, 0.5))
    F[:] = -100.0  # the returned array is a copy
    assert np.all(record.objectives()[0] > -100.0)
    copy = record.objectives()
    copy[:] = -100.0  # objectives() returns a copy too
    assert np.all(record.objectives()[0] > -100.0)


def test_tampering_attempt_does_not_change_the_score():
    cheat = (
        "import numpy as np\n"
        "class Cheat:\n"
        "    def __init__(self, budget, dim):\n"
        "        self.budget, self.dim = budget, dim\n"
        "    def __call__(self, func):\n"
        "        func(np.full(self.dim, 0.5))\n"
        "        for name in ('_archive', '_record'):\n"
        "            try:\n"
        "                getattr(func, name).extend([np.zeros(2)] * 1000)\n"
        "            except AttributeError:\n"
        "                pass\n"
    )
    problem = MultiObjectiveBenchmark(["zdt1"], ["zdt2"], budget_factor=5, runs=1)
    solution = problem.evaluate(Solution(code=cheat, name="Cheat"))
    run = solution.metadata["performance_data"][0]
    assert run["evaluations"] == 1
    assert run["final_hv"] < 0.5


# ---------------------------------------------------------------- scoring


def test_pareto_front_scores_about_one():
    front = _pymoo_problem(function_spec("zdt1")).pareto_front()
    result = anytime_hypervolume(front, front, budget=len(front), checkpoints=1)
    assert result["final_hv"] == pytest.approx(1.0)
    assert result["igd_plus"] == pytest.approx(0.0, abs=1e-12)
    assert result["distance"] == 0.0
    # The whole front within the first checkpoint (budget/50 evaluations):
    # the area is close to 1. Found only halfway, the area is about halved.
    early = np.vstack([front, np.full((4900, 2), 5.0)])
    assert anytime_hypervolume(early, front, budget=5000)["area"] > 0.99
    late = np.vstack([np.full((2500, 2), 0.95), front, np.full((2400, 2), 5.0)])
    assert 0.5 < anytime_hypervolume(late, front, budget=5000)["area"] < 0.8


def test_points_outside_the_box_give_negative_but_ordered_scores():
    front = _pymoo_problem(function_spec("zdt1")).pareto_front()
    near = np.full((10, 2), 1.5)  # normalised distance ~0.57 from the box
    far = np.full((10, 2), 10.0)
    near_score = anytime_hypervolume(near, front, budget=10)["area"]
    far_score = anytime_hypervolume(far, front, budget=10)["area"]
    assert -1.0 < far_score < near_score < 0.0
    assert anytime_hypervolume(np.empty((0, 2)), front, budget=10)["area"] == -1.0


def test_better_front_scores_higher():
    front = _pymoo_problem(function_spec("zdt1")).pareto_front()
    worse = front + 0.2
    better = front + 0.05
    assert (
        anytime_hypervolume(better, front, budget=100)["area"]
        > anytime_hypervolume(worse, front, budget=100)["area"]
    )


# ---------------------------------------------------------------- problem


def test_evaluate_scores_and_reports():
    problem = MultiObjectiveBenchmark(["zdt1"], ["zdt2"], budget_factor=5, runs=2)
    solution = problem.evaluate(Solution(code=RANDOM, name="RandomSearch"))
    assert -1.0 < solution.fitness <= 1.0
    data = solution.metadata["performance_data"]
    assert [r["seed"] for r in data] == [0, 1]
    # budget = budget_factor * n_var = 5 * 30 for ZDT1
    assert all(r["budget"] == 150 and r["evaluations"] == 150 for r in data)
    assert "zdt1" in solution.feedback and "IGD+" in solution.feedback
    tested = problem.test(Solution(code=RANDOM, name="RandomSearch"))
    assert {r["function"] for r in tested.metadata["performance_data"]} == {"zdt2"}


def test_evaluate_is_reproducible_per_seed():
    problem = MultiObjectiveBenchmark(["zdt1"], ["zdt2"], budget_factor=3, runs=1)
    a = problem.evaluate(Solution(code=RANDOM, name="RandomSearch")).fitness
    b = problem.evaluate(Solution(code=RANDOM, name="RandomSearch")).fitness
    assert a == b


def test_evaluate_errors():
    problem = MultiObjectiveBenchmark(["zdt1"], ["zdt2"], budget_factor=2, runs=1)
    lazy = "class Lazy:\n    def __init__(self, budget, dim):\n        pass\n    def __call__(self, func):\n        return None\n"
    with pytest.raises(ValueError, match="never called func"):
        problem.evaluate(Solution(code=lazy, name="Lazy"))
    with pytest.raises(ValueError, match="not found"):
        problem.evaluate(Solution(code="x = 1", name="Missing"))


def test_over_budget_candidates_are_scored_on_what_fits():
    greedy = RANDOM.replace("(self.budget, self.dim)", "(self.budget * 2, self.dim)")
    problem = MultiObjectiveBenchmark(["zdt1"], ["zdt2"], budget_factor=2, runs=1)
    solution = problem.evaluate(Solution(code=greedy, name="RandomSearch"))
    assert solution.metadata["performance_data"][0]["evaluations"] == 2 * 30
    assert math.isfinite(solution.fitness)


def test_constructor_validation():
    with pytest.raises(ValueError, match="Unknown multi-objective"):
        MultiObjectiveBenchmark(["sphere"])
    with pytest.raises(ValueError, match="'budget_factor' must be an integer"):
        MultiObjectiveBenchmark(budget_factor=0)
    with pytest.raises(Exception):
        MultiObjectiveBenchmark([{"name": "dtlz2", "n_obj": "three"}])


def test_config_tags_dependencies_and_suites():
    from iohblade.tags import ObjectiveType, PrimaryCategories

    problem = MultiObjectiveBenchmark()
    config = problem.get_config()
    assert PrimaryCategories.BBO in config["tags"]
    assert ObjectiveType.MULTI_OBJECTIVE in config["tags"]
    # The fitness is a scalar (the score), so no per-objective keys.
    assert "keys" not in config["config"]
    assert all(dep in problem.dependencies for dep in MO_DEPENDENCIES)
    assert problem.minimisation is False
    sizes = {make.__name__: make() for make in (mo_quick, mo_zdt, mo_dtlz, mo_wfg)}
    assert [p.name for ps in sizes.values() for p in ps] == [
        "MO-quick",
        "MO-ZDT",
        "MO-DTLZ",
        "MO-WFG",
    ]
    (dtlz,) = sizes["mo_dtlz"]
    assert all(spec["n_obj"] == 3 for spec in dtlz.training_instances)
    assert len(dtlz.test_instances) == 3


# ---------------------------------------------------------------- YAML

BASE = {
    "experiment": {"output_dir": "results/mo", "runs": 1, "budget": 5},
    "llms": {"a": {"provider": "dummy"}},
    "methods": [{"type": "random_search", "llm": "a"}],
}


def test_yaml_problem_and_suite():
    plan = load_plan(
        {
            **BASE,
            "problems": [
                {
                    "type": "multiobjective",
                    "name": "mine",
                    "params": {
                        "training_instances": ["zdt1", {"name": "dtlz2", "n_obj": 3}],
                        "test_instances": ["wfg4"],
                        "budget_factor": 100,
                    },
                },
                {"suite": "mo_quick"},
            ],
        },
        env={},
    )
    names = [p.name for p in plan.problems]
    assert names == ["mine", "MO-quick"]
    assert plan.problems[0].budget_factor == 100


def test_yaml_reports_an_unknown_function():
    with pytest.raises(ConfigError) as exc:
        load_plan(
            {
                **BASE,
                "problems": [
                    {
                        "type": "multiobjective",
                        "params": {"training_instances": ["zdt9"]},
                    }
                ],
            },
            env={},
        )
    assert "zdt9" in "\n".join(str(i) for i in exc.value.issues)


def test_default_budget_and_timeout_scale_with_dimension():
    problem = MultiObjectiveBenchmark(["zdt1"], [{"name": "dtlz2", "n_obj": 3}])
    assert problem.budget_factor == 2000
    # 3 runs * 2000 * 30 variables (ZDT1) = 180 000 evaluations, 2 ms each
    assert problem.eval_timeout == 360
    assert MultiObjectiveBenchmark(["zdt1"], budget_factor=1).eval_timeout == 300
    assert MultiObjectiveBenchmark(eval_timeout=42).eval_timeout == 42
