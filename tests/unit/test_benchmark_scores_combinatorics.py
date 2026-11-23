"""Regression tests for benchmark scoring bugs (GitHub issue #203).

Covers the Euclidean Steiner tree, Erdős minimum overlap, graph colouring
and toy multi-objective benchmarks: invalid outputs must get the worst
fitness, and scores must reflect what the harness itself computed.
"""

import math
import textwrap

import pytest

from iohblade.benchmarks.combinatorics.erdos_min_overlap import ErdosMinOverlap
from iohblade.benchmarks.combinatorics.euclidian_steiner_tree import (
    EuclidianSteinerTree,
    Location,
)
from iohblade.benchmarks.combinatorics.graph_coloring import GraphColoring
from iohblade.benchmarks.toy_multiobjective.toy_multiobjective import ToyMultiObjective
from iohblade.solution import Solution

# ---------------------------------------------------------------- Steiner


@pytest.fixture
def triangle_steiner():
    problem = EuclidianSteinerTree(20)
    problem.points = {
        0: [
            Location(0, 0.0, 0.0),
            Location(1, 1.0, 0.0),
            Location(2, 0.5, math.sqrt(3) / 2),
        ]
    }
    return problem


def _steiner(points_expr: str) -> Solution:
    code = textwrap.dedent(f"""
        class Steiner:
            def __init__(self, points, tolerance):
                self.points = points

            def __call__(self):
                return {points_expr}
        """)
    return Solution(code=code, name="Steiner")


def test_steiner_centroid_improves(triangle_steiner):
    # The centroid of an equilateral triangle is its Fermat point.
    result = triangle_steiner.evaluate(_steiner(f"[[0.5, {math.sqrt(3) / 6!r}]]"))
    assert not result.error, result.feedback
    assert triangle_steiner.minimisation
    assert result.fitness < 1
    assert result.fitness == pytest.approx(math.sqrt(3) / 2)


def test_steiner_far_point_scores_worse_than_no_points(triangle_steiner):
    none = triangle_steiner.evaluate(_steiner("[]"))
    far = triangle_steiner.evaluate(_steiner("[[1e9, 1e9]]"))
    assert not none.error and not far.error
    assert none.fitness == pytest.approx(1.0)
    assert far.fitness > 1
    assert far.fitness > none.fitness  # minimisation: higher is worse


@pytest.mark.parametrize("bad", ["float('nan')", "float('inf')"])
def test_steiner_non_finite_point_is_invalid(triangle_steiner, bad):
    result = triangle_steiner.evaluate(_steiner(f"[[{bad}, 0.5]]"))
    assert result.error
    assert result.fitness == math.inf


# ------------------------------------------------------------------ Erdős


def test_erdos_nan_is_invalid():
    problem = ErdosMinOverlap()
    code = textwrap.dedent(f"""
        class ErdosCandidate:
            def __call__(self):
                return [float("nan")] * {problem.n_bins}
        """)
    result = problem.evaluate(Solution(code=code, name="ErdosCandidate"))
    assert result.fitness == problem.failure_fitness() == math.inf
    assert result.error


def test_erdos_baseline_still_valid():
    problem = ErdosMinOverlap()
    code = textwrap.dedent(f"""
        class ErdosCandidate:
            def __call__(self):
                return [0.5] * {problem.n_bins}
        """)
    result = problem.evaluate(Solution(code=code, name="ErdosCandidate"))
    assert not result.error
    assert math.isfinite(result.fitness) and result.fitness > 0.38


# --------------------------------------------------------- Graph colouring


def _coloring(body: str) -> Solution:
    code = "class GraphColoring:\n"
    code += "    def __init__(self, nodes, edges):\n"
    code += "        self.nodes = nodes\n"
    code += "        self.edges = edges\n\n"
    code += "    def __call__(self):\n"
    code += textwrap.indent(textwrap.dedent(body), " " * 8)
    return Solution(code=code, name="GraphColoring")


@pytest.mark.parametrize(
    "body",
    [
        # One shared NaN object: passes `!=` on every edge in the old code.
        "nan = float('nan')\nreturn {n: nan for n in self.nodes}\n",
        "return {n: float('nan') for n in self.nodes}\n",
        # A missing node.
        "return {n: i + 1 for i, n in enumerate(self.nodes[1:])}\n",
        # Booleans are ints in Python but not colours.
        "return {n: i % 2 == 0 for i, n in enumerate(self.nodes)}\n",
        # Colours must be >= 1.
        "return {n: -i for i, n in enumerate(self.nodes)}\n",
        # Floats are not colours either.
        "return {n: float(i + 1) for i, n in enumerate(self.nodes)}\n",
        "return [1] * len(self.nodes)\n",
    ],
    ids=["shared-nan", "nan", "missing-node", "bool", "zero-neg", "float", "list"],
)
def test_coloring_invalid_outputs(body):
    problem = GraphColoring("1")
    result = problem.evaluate(_coloring(body))
    assert result.error, result.feedback
    assert result.fitness == math.inf


def test_coloring_valid_greedy_scores_number_of_colours():
    problem = GraphColoring("1")
    body = """
        import numpy as np
        neighbours = {n: set() for n in self.nodes}
        for u, v in self.edges:
            neighbours[u].add(v)
            neighbours[v].add(u)
        colours = {}
        for n in self.nodes:
            used = {colours[m] for m in neighbours[n] if m in colours}
            c = 1
            while c in used:
                c += 1
            colours[n] = np.int64(c)  # numpy integers are accepted
        return colours
        """
    result = problem.evaluate(_coloring(body))
    assert not result.error, result.feedback
    # Recompute the greedy colouring here to know the expected count.
    neighbours = {n: set() for n in problem.nodes}
    for u, v in problem.edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    colours = {}
    for n in problem.nodes:
        used = {colours[m] for m in neighbours[n] if m in colours}
        c = 1
        while c in used:
            c += 1
        colours[n] = c
    assert result.fitness == float(len(set(colours.values())))


def test_coloring_conflict_is_invalid():
    problem = GraphColoring("1")
    result = problem.evaluate(_coloring("return {n: 1 for n in self.nodes}\n"))
    assert result.error
    assert result.fitness == math.inf


# ------------------------------------------------------- Toy multi-objective


def _toy(body: str) -> Solution:
    code = "class BiSphereSearcher:\n"
    code += "    def run(self, objective, budget=100):\n"
    code += textwrap.indent(textwrap.dedent(body), " " * 8)
    return Solution(code=code, name="BiSphereSearcher")


def _values(fitness):
    return fitness["f1"], fitness["f2"]


REAL_SEARCH = """
    for i in range(budget):
        objective(-1.0 + 4.0 * i / (budget - 1))
    """


def test_toy_fake_result_cannot_beat_real_search():
    problem = ToyMultiObjective(budget=20)
    fake = problem.evaluate(_toy("return (-1e9, -1e9)\n"))
    real = problem.evaluate(_toy(REAL_SEARCH))
    assert not real.error, real.feedback
    assert all(math.isfinite(v) for v in _values(real.fitness))
    # Maximisation of negated objectives: the real search must be better.
    assert real.fitness > fake.fitness
    assert fake.error
    assert _values(fake.fitness) == (-math.inf, -math.inf)


def test_toy_score_comes_from_observed_values():
    problem = ToyMultiObjective(budget=5)
    # Only x = 1 is evaluated, whatever run() returns.
    result = problem.evaluate(_toy("objective(1.0)\nreturn (0.0, 0.0)\n"))
    assert not result.error
    assert _values(result.fitness) == (-1.0, -1.0)


def test_toy_budget_is_enforced():
    problem = ToyMultiObjective(budget=5)
    # Evaluates x=0 and x=2 only after the budget: they must not count.
    body = """
        for _ in range(budget):
            objective(1.0)
        objective(0.0)
        objective(2.0)
        """
    result = problem.evaluate(_toy(body))
    assert not result.error
    assert _values(result.fitness) == (-1.0, -1.0)
    assert "more than 5" in result.feedback


def test_toy_swallowed_budget_exception_does_not_help():
    problem = ToyMultiObjective(budget=3)
    body = """
        for x in [1.0, 1.0, 1.0, 0.0, 2.0]:
            try:
                objective(x)
            except Exception:
                pass
        """
    result = problem.evaluate(_toy(body))
    assert not result.error
    assert _values(result.fitness) == (-1.0, -1.0)
    assert "not scored" in result.feedback


def test_toy_x_is_clipped_to_domain():
    problem = ToyMultiObjective(budget=5)
    # x = 100 is clipped to 3: f1 = 9, f2 = 1.
    result = problem.evaluate(_toy("objective(100.0)\n"))
    assert _values(result.fitness) == (-9.0, -1.0)


def test_toy_nan_x_is_invalid():
    problem = ToyMultiObjective(budget=5)
    result = problem.evaluate(_toy("objective(float('nan'))\n"))
    assert result.error
    assert _values(result.fitness) == (-math.inf, -math.inf)
