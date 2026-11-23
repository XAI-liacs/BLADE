"""Tests for the ToyMultiObjective benchmark problem."""

import math

import pytest

from iohblade.benchmarks.toy_multiobjective import ToyMultiObjective
from iohblade.fitness import Fitness
from iohblade.solution import Solution

_GOOD_CODE = """\
import numpy as np


class BiSphereSearcher:
    \"\"\"Simple grid searcher for the bi-sphere toy problem.\"\"\"

    def run(self, objective, budget: int = 100):
        for x in np.linspace(-1.0, 3.0, budget):
            objective(x)
"""

_BROKEN_CODE = "this is not valid python ]["

_MISSING_CLASS_CODE = """\
def run(objective, budget=100):
    objective(0.0)
"""

_NO_EVALUATION_CODE = """\
class BiSphereSearcher:
    def run(self, objective, budget=100):
        return 0.0, 0.0  # reports values without evaluating anything
"""


@pytest.fixture
def problem():
    return ToyMultiObjective(budget=50)


def test_evaluate_returns_fitness_object(problem):
    sol = Solution(code=_GOOD_CODE)
    result = problem.evaluate(sol)
    assert isinstance(result.fitness, Fitness)


def test_evaluate_good_code_valid_values(problem):
    sol = Solution(code=_GOOD_CODE)
    result = problem.evaluate(sol)
    f = result.fitness
    # Negated: values should be <= 0
    assert f["f1"] <= 0.0
    assert f["f2"] <= 0.0
    # f1 objective: best x^2 with x in [-1,3] → close to 0
    # (linspace with budget=50 may not land exactly on 0)
    assert f["f1"] >= -0.1  # should be very close to 0 (negated)


def _is_failure(fitness):
    # Failures score the worst value (-inf) on both maximised objectives.
    return fitness["f1"] == -math.inf and fitness["f2"] == -math.inf


def test_evaluate_broken_code_is_failure(problem):
    sol = Solution(code=_BROKEN_CODE)
    result = problem.evaluate(sol)
    assert _is_failure(result.fitness)


def test_evaluate_missing_class_is_failure(problem):
    sol = Solution(code=_MISSING_CLASS_CODE)
    result = problem.evaluate(sol)
    assert _is_failure(result.fitness)
    assert "BiSphereSearcher" in result.feedback


def test_evaluate_without_evaluations_is_failure(problem):
    """Values returned by run() are ignored; only evaluated points count."""
    sol = Solution(code=_NO_EVALUATION_CODE)
    result = problem.evaluate(sol)
    assert _is_failure(result.fitness)


def test_test_method_same_as_evaluate(problem):
    """test() should behave identically to evaluate() for this toy problem."""
    sol = Solution(code=_GOOD_CODE)
    r1 = problem.evaluate(sol)
    r2 = problem.test(sol)
    assert r1.fitness == r2.fitness


def test_to_dict(problem):
    d = problem.to_dict()
    assert d["name"] == "ToyMultiObjective"
    assert d["eval_budget"] == 50


def test_fitness_objectives_present(problem):
    sol = Solution(code=_GOOD_CODE)
    result = problem.evaluate(sol)
    assert "f1" in result.fitness.keys()
    assert "f2" in result.fitness.keys()


def test_fitness_pareto_trade_off(problem):
    """A solution closer to x=0 should have better f1 but worse f2."""
    code_near_zero = """\
class BiSphereSearcher:
    def run(self, objective, budget=100):
        objective(0.03)  # x ~ 0
"""
    code_near_two = """\
class BiSphereSearcher:
    def run(self, objective, budget=100):
        objective(1.97)  # x ~ 2
"""
    r_zero = problem.evaluate(Solution(code=code_near_zero))
    r_two = problem.evaluate(Solution(code=code_near_two))
    # r_zero has better f1 (less negative means closer to 0)
    assert r_zero.fitness["f1"] > r_two.fitness["f1"]
    # r_two has better f2
    assert r_two.fitness["f2"] > r_zero.fitness["f2"]
