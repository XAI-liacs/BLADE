"""Regression tests for #201: the optimisation direction is respected.

Failed evaluations must score the worst value for the problem's direction,
and methods must never pick a failed (non-finite) solution as their best.
"""

import math
from types import SimpleNamespace

import pytest

from iohblade.fitness import (
    Fitness,
    from_minimised,
    is_better,
    to_minimised,
    worst_fitness,
)
from iohblade.mcts_node import MCTS_Node
from iohblade.methods.mcts_ahd.mcts import MCTS
from iohblade.methods.mcts_ahd.prompts import MCTS_Prompts
from iohblade.methods.moeh_method.prompts import MoEH_Prompts
from iohblade.methods.random_search import RandomSearch
from iohblade.problem import Problem
from iohblade.solution import Solution


class _Scripted(Problem):
    """A problem that returns a fixed sequence of fitness values."""

    def __init__(self, scores, minimisation=False):
        super().__init__(name="scripted")
        self.scores = list(scores)
        self.minimisation = minimisation
        self.task_prompt = "task"

    def __call__(self, solution, logger=None):
        score = self.scores.pop(0)
        if score is None:  # a failed evaluation, as Problem.__call__ reports it
            return solution.set_scores(self.failure_fitness(), "failed", "failed")
        return solution.set_scores(score, f"score {score}")

    def evaluate(self, solution):
        return solution

    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self):
        return {"name": self.name, "minimisation": self.minimisation, "config": {}}


class _SequentialLLM:
    model = "scripted"

    def __init__(self):
        self.count = 0

    def sample_solution(self, *args, **kwargs):
        self.count += 1
        return Solution(code=f"class A{self.count}: pass", name=f"A{self.count}")


# --- Problem failure score -------------------------------------------------


@pytest.mark.parametrize(
    "minimisation, expected", [(True, math.inf), (False, -math.inf)]
)
def test_failure_fitness_is_worst_for_direction(minimisation, expected):
    problem = _Scripted([], minimisation=minimisation)
    assert problem.failure_fitness() == expected
    assert worst_fitness(minimisation) == expected


def test_budget_exhausted_scores_worst_for_minimisation():
    problem = _Scripted([], minimisation=True)
    problem.logger = SimpleNamespace(budget_exhausted=lambda: True)
    solution = Problem.__call__(problem, Solution(code="x = 1"))
    assert solution.fitness == math.inf
    assert solution.feedback == "Budget is exhausted."


# --- comparisons -----------------------------------------------------------


def test_is_better_never_prefers_failures():
    for bad in (math.inf, -math.inf, math.nan, None):
        assert not is_better(bad, 1.0, minimisation=True)
        assert not is_better(bad, 1.0, minimisation=False)
        assert is_better(1.0, bad, minimisation=True)
        assert is_better(1.0, bad, minimisation=False)
    assert is_better(1.0, 2.0, minimisation=True)
    assert not is_better(2.0, 1.0, minimisation=True)
    assert is_better(2.0, 1.0, minimisation=False)
    assert not is_better(1.0, 1.0, minimisation=False)  # ties keep the incumbent


def test_is_better_multi_objective_uses_dominance():
    a = Fitness({"x": 1.0, "y": 1.0})
    b = Fitness({"x": 2.0, "y": 2.0})
    c = Fitness({"x": 0.0, "y": 3.0})
    assert is_better(a, b, minimisation=True)
    assert is_better(b, a, minimisation=False)
    assert not is_better(a, c, minimisation=True)  # incomparable
    assert not is_better(Fitness({"x": math.inf, "y": 0.0}), b, minimisation=True)


@pytest.mark.parametrize("minimisation", [True, False])
def test_minimised_objective_roundtrip(minimisation):
    assert from_minimised(to_minimised(-3.5, minimisation), minimisation) == -3.5
    assert to_minimised(math.nan, minimisation) == math.inf
    assert to_minimised(worst_fitness(minimisation), minimisation) == math.inf
    assert from_minimised(math.inf, minimisation) == worst_fitness(minimisation)
    assert to_minimised(5.0, minimisation) == (5.0 if minimisation else -5.0)


# --- methods ---------------------------------------------------------------


@pytest.mark.parametrize(
    "minimisation, scores, expected",
    [
        (True, [5.0, None, 3.0, 3.0], 3.0),
        (False, [5.0, None, 7.0, 7.0], 7.0),
        (True, [None, 4.0], 4.0),
    ],
)
def test_random_search_ignores_failures(minimisation, scores, expected):
    problem = _Scripted(scores, minimisation=minimisation)
    best = RandomSearch(_SequentialLLM(), budget=len(scores))(problem)
    assert best.fitness == expected
    # ties keep the first solution with that fitness
    assert best.name == f"A{scores.index(expected) + 1}"


def test_lhns_best_is_not_replaced_by_failure():
    from iohblade.methods.lhns.lhns import LHNS

    problem = _Scripted([], minimisation=True)
    lhns = LHNS(problem, _SequentialLLM(), "vns", minimisation=True)
    good = Solution().set_scores(5.0)
    lhns._log_best_solution(good)
    lhns._log_best_solution(Solution().set_scores(math.inf, "", "timeout"))
    lhns._log_best_solution(Solution().set_scores(-math.inf, "", "crash"))
    assert lhns.best_solution is good
    better = Solution().set_scores(2.0)
    lhns._log_best_solution(better)
    assert lhns.best_solution is better


@pytest.mark.parametrize("minimisation", [True, False])
def test_eoh_adapter_maps_failures_to_worst(minimisation):
    from iohblade.methods.eoh import _BladeProblemAdapter

    problem = _Scripted([None, 4.0], minimisation=minimisation)
    adapter = _BladeProblemAdapter(problem)
    assert adapter.evaluate("class A: pass") == math.inf  # EoH minimises
    assert adapter.evaluate("class A: pass") == (4.0 if minimisation else -4.0)


class _FakeReEvo:
    function_evals = 0

    def mark_invalid_individual(self, individual, message):
        individual["exec_success"] = False
        individual["obj"] = math.inf
        individual["traceback_msg"] = message
        return individual


@pytest.mark.parametrize("minimisation", [True, False])
def test_reevo_population_objectives(minimisation):
    from iohblade.methods.reevo import ReEvo

    problem = _Scripted([4.0, None], minimisation=minimisation)
    population = [
        {"code": "class A: pass"},
        {"code": "class B: pass"},
        {"code": None},
    ]
    result = ReEvo(_SequentialLLM(), 3)._eval_population(
        _FakeReEvo(), population, problem
    )
    assert result[0]["exec_success"]
    assert result[0]["obj"] == (4.0 if minimisation else -4.0)
    assert not result[1]["exec_success"] and result[1]["obj"] == math.inf
    # No code: invalid, with a string as code (ReEvo reads it later).
    assert not result[2]["exec_success"] and result[2]["obj"] == math.inf
    assert result[2]["code"] == ""


# --- MCTS-AHD --------------------------------------------------------------


def _mcts(maximisation):
    problem = _Scripted([], minimisation=not maximisation)
    return MCTS(_SequentialLLM(), problem, budget=10, maximisation=maximisation)


def _child(parent, q):
    node = MCTS_Node(Solution(), approach="m1")
    node.Q = q
    node.fitness = q if q is not None else math.nan
    node.visit = 1
    parent.add_child(node)
    return node


@pytest.mark.parametrize("maximisation, expected", [(True, 9.0), (False, 1.0)])
def test_mcts_backpropagation_uses_best_child(maximisation, expected):
    mcts = _mcts(maximisation)
    parent = _child(mcts.root, 5.0)
    leaf = _child(parent, 1.0)
    _child(parent, 9.0)
    mcts.discount_factor = 1.0
    mcts.backpropogate(leaf)
    assert parent.Q == expected


def test_mcts_uct_treats_zero_as_a_score_and_prefers_better_nodes():
    for maximisation in (True, False):
        mcts = _mcts(maximisation)
        mcts.q_min, mcts.q_max = 0.0, 10.0
        mcts.eval_remain = 0  # no exploration bonus
        low = _child(mcts.root, 0.0)
        high = _child(mcts.root, 10.0)
        failed = _child(mcts.root, None)
        assert mcts.uct(failed) == -math.inf
        assert mcts.uct(low) != -math.inf  # 0.0 is a real score
        best = max(mcts.root.children, key=mcts.uct)
        assert best is (high if maximisation else low)


def test_mcts_node_fitness_comparison_follows_direction():
    a = MCTS_Node(Solution(), approach="m1")
    b = MCTS_Node(Solution(), approach="m1")
    a.Q, b.Q = 1.0, 2.0
    assert a.is_less_fit_than(b, maximisation=True)
    assert not a.is_less_fit_than(b, maximisation=False)
    assert b.is_less_fit_than(a, maximisation=False)


def test_mcts_simulate_never_keeps_failure_as_best():
    mcts = _mcts(maximisation=False)
    mcts.problem = _Scripted([3.0, None], minimisation=True)
    good = mcts.simulate(_child(mcts.root, None))
    mcts.simulate(_child(mcts.root, None))
    assert mcts.best_solution is good


# --- prompts ---------------------------------------------------------------


def test_prompts_state_the_direction():
    node = MCTS_Node(Solution(code="x=1"), approach="m1")
    node.fitness = 1.0
    for maximisation, word in ((True, "higher"), (False, "lower")):
        for prompt in (
            MCTS_Prompts.get_prompt_e1("t", "e", "f", [node], maximisation),
            MCTS_Prompts.get_prompt_e2("t", "e", "f", [node, node], maximisation),
            MCTS_Prompts.get_prompt_s1("t", "e", "f", [node], maximisation),
            MoEH_Prompts.get_prompt_e2("e", "f", [node, node], maximisation),
        ):
            assert f"{word} is better" in prompt
            assert "lower than" not in prompt
