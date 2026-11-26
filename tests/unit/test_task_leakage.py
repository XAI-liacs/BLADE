"""Benchmarks must not tell the LLM (or its code) which benchmark it is on."""

import ioh
import numpy as np
import pytest

from iohblade.benchmarks.blackbox import BlackBoxFunction
from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX
from iohblade.benchmarks.BBOB.mabbob import MA_BBOB
from iohblade.benchmarks.geometry.heilbronn_triangle import HeilbronnTriangle
from iohblade.benchmarks.logistics.tsp import TravelingSalesmanProblem
from iohblade.benchmarks.number_theory.get_num_theory_problems import (
    get_sum_vs_difference_problem,
)
from iohblade.solution import Solution
from iohblade.utils import OverBudgetException, aoc_logger


def _bbob_problem(fid=3, iid=2, dim=5):
    return ioh.get_problem(
        fid, instance=iid, dimension=dim, problem_class=ioh.ProblemClass.BBOB
    )


def test_blackbox_hides_optimum_and_identity():
    func = BlackBoxFunction(_bbob_problem())
    for name in ("optimum", "reset", "detach_logger", "attach_logger"):
        assert not hasattr(func, name)
    assert not hasattr(func.meta_data, "problem_id")
    assert not hasattr(func.meta_data, "name")
    assert not hasattr(func.meta_data, "instance")
    assert not hasattr(func.state, "optimum_found")
    with pytest.raises(AttributeError):
        func.extra = 1  # __slots__: nothing can be attached either


def test_blackbox_behaves_like_the_problem():
    problem = _bbob_problem()
    func = BlackBoxFunction(problem)
    x = np.zeros(5)
    assert func(x) == pytest.approx(problem(x))
    assert func.meta_data.n_variables == 5
    np.testing.assert_allclose(func.bounds.lb, -5.0)
    np.testing.assert_allclose(func.bounds.ub, 5.0)
    assert func.state.evaluations == 2
    func.bounds.lb[:] = 0  # changing the copy leaves the problem alone
    np.testing.assert_allclose(problem.bounds.lb, -5.0)


def test_blackbox_keeps_budget_logger():
    problem = _bbob_problem()
    # Keep a reference: ioh does not keep the logger alive.
    budget = aoc_logger(3, triggers=[ioh.logger.trigger.ALWAYS])
    problem.attach_logger(budget)
    func = BlackBoxFunction(problem)
    for _ in range(3):
        func(np.zeros(5))
    with pytest.raises(OverBudgetException):
        func(np.zeros(5))


CHEAT = """
import numpy as np

class Cheat:
    def __init__(self, budget, dim):
        self.budget = budget

    def __call__(self, func):
        x = np.array(func.optimum.x)
        return func(x), x
"""


@pytest.mark.parametrize(
    "problem",
    [
        BBOB_SBOX(
            training_instances=[(3, 1)],
            test_instances=[(3, 2)],
            dims=[2],
            budget_factor=50,
            problem_type=ioh.ProblemClass.BBOB,
        ),
        MA_BBOB(training_instances=[0], test_instances=[1], dims=[2], budget_factor=50),
    ],
    ids=["bbob", "ma_bbob"],
)
def test_reading_the_optimum_fails(problem):
    with pytest.raises(AttributeError):
        problem.evaluate(Solution(code=CHEAT, name="Cheat"))


def test_bbob_prompts_do_not_name_the_suite():
    for group in range(1, 6):
        prompt = BBOB_SBOX(
            specific_group=group, problem_type=ioh.ProblemClass.BBOB
        ).get_prompt()
        assert "BBOB" not in prompt
        assert "Separable Functions" not in prompt
        assert "adequate global structure" not in prompt
    prompt = BBOB_SBOX(specific_fid=3).get_prompt()
    assert "f3" not in prompt and "Separable Rastrigin" in prompt
    prompt = MA_BBOB().get_prompt()
    assert "BBOB" not in prompt and "Many Affine" not in prompt


def test_math_prompts_do_not_cite_the_benchmark():
    prompt = get_sum_vs_difference_problem()[0].get_prompt()
    assert "C6" not in prompt and "Gyarmati" not in prompt
    tsp = TravelingSalesmanProblem("A-n53-k7").get_prompt()
    assert "Vehicle Routing" not in tsp and "A-n53-k7" not in tsp


POINTS = """
import numpy as np

class Points:
    def __init__(self, n_points):
        self.n_points = n_points

    def __call__(self):
        rng = np.random.default_rng(0)
        u = rng.random((self.n_points, 2))
        u[u.sum(axis=1) > 1] = 1 - u[u.sum(axis=1) > 1]
        return None, np.column_stack([u[:, 0], 2 * u[:, 1]])
"""


def test_best_known_only_in_feedback_when_enabled():
    problem = HeilbronnTriangle(n_points=11, best_known=0.0365)
    hidden = problem.evaluate(Solution(code=POINTS, name="Points"))
    assert "0.0365" not in hidden.feedback
    assert "best known" not in hidden.feedback

    problem.show_best_known = True
    shown = problem.evaluate(Solution(code=POINTS, name="Points"))
    assert "best known: 0.0365" in shown.feedback
    assert shown.fitness == hidden.fitness
