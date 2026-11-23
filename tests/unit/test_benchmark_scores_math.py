"""Regression tests for scoring bugs in the maths benchmarks (issue #203).

Each test feeds a hand-made candidate to a benchmark's ``evaluate`` (in
process, no virtualenv) and checks that invalid output gets the worst
fitness and that valid output gets the mathematically correct score.
"""

import math
import textwrap

import numpy as np
import pytest

from iohblade.benchmarks.analysis.auto_correlation_ineq1 import AutoCorrIneq1
from iohblade.benchmarks.analysis.auto_correlation_ineq2 import AutoCorrIneq2
from iohblade.benchmarks.analysis.auto_correlation_ineq3 import AutoCorrIneq3
from iohblade.benchmarks.geometry.heilbronn_triangle import HeilbronnTriangle
from iohblade.benchmarks.geometry.min_max_distance_ratio import MinMaxMinDistanceRatio
from iohblade.benchmarks.matrix_multiplication.constants import MATMUL_RANKS
from iohblade.benchmarks.matrix_multiplication.problem_sets import default_problems
from iohblade.benchmarks.matrix_multiplication.task_specification import (
    MatMulTensorDecomposition,
)
from iohblade.benchmarks.packing.hexagon_packing import HexagonPacking
from iohblade.benchmarks.packing.rectangle_packing import RectanglePacking
from iohblade.benchmarks.packing.unit_square_packing import UnitSquarePacking
from iohblade.solution import Solution


def _evaluate(problem, code, name="Candidate"):
    solution = problem.evaluate(Solution(code=textwrap.dedent(code), name=name))
    return solution


def _returns(value_code, init="def __init__(self, *args, **kwargs):\n        pass"):
    """Code of a class ``Candidate`` whose call returns ``value_code``."""
    return f"""
import numpy as np

class Candidate:
    {init}

    def __call__(self):
        return {value_code}
"""


# ---------------------------------------------------------------- NaN inputs


def test_unit_square_packing_rejects_nan_circles():
    problem = UnitSquarePacking(26, best_known=2.635)
    result = _evaluate(problem, _returns("np.array([[np.nan, np.nan, 0.5]] * 26)"))
    assert result.fitness == -math.inf
    assert "non-finite" in result.feedback.lower()


def test_rectangle_packing_rejects_nan_dimensions():
    problem = RectanglePacking()
    code = _returns("[[10 * i + 4, 4, 4.0] for i in range(21)], np.nan, np.nan")
    result = _evaluate(problem, code)
    assert result.fitness == -math.inf
    assert "non-finite" in result.feedback


def test_rectangle_packing_rejects_nan_circles():
    problem = RectanglePacking()
    code = _returns("[[np.nan, np.nan, 0.1]] * 21, 1.0, 1.0")
    result = _evaluate(problem, code)
    assert result.fitness == -math.inf
    assert "non-finite" in result.feedback


@pytest.mark.parametrize(
    "problem_cls, n_bins",
    [(AutoCorrIneq1, 600), (AutoCorrIneq2, 50), (AutoCorrIneq3, 400)],
)
def test_auto_correlation_rejects_nan(problem_cls, n_bins):
    problem = problem_cls()
    result = _evaluate(problem, _returns(f"[float('nan')] * {n_bins}", init="pass"))
    assert result.fitness == problem.failure_fitness()
    assert "finite" in result.feedback


def test_auto_correlation_1_rejects_inf_values():
    problem = AutoCorrIneq1()
    result = _evaluate(problem, _returns("[float('inf')] * 600", init="pass"))
    assert result.fitness == math.inf


def test_matrix_multiplication_rejects_nan():
    problem = MatMulTensorDecomposition(2, 2, 2, grid=0.5, rank=7)
    result = _evaluate(problem, _returns("[float('nan')] * (12 * 7)"))
    assert result.fitness == math.inf
    assert "non-finite" in result.feedback


# --------------------------------------------------------- hexagon packing


def _exact_side_lp(A):
    """Independent check: smallest enclosing outer hexagon by linear program.

    Outer hexagon: vertices at 30 + k*60 degrees, i.e. edge normals at k*60
    degrees, apothem s*sqrt(3)/2; every inner vertex v must satisfy
    n_k . (v - c) <= s*sqrt(3)/2.
    """
    from scipy.optimize import linprog

    verts = []
    for x, y, th in A:
        for k in range(6):
            a = th + k * math.pi / 3
            verts.append((x + math.cos(a), y + math.sin(a)))
    rows, rhs = [], []
    for k in range(6):
        nx, ny = math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)
        for vx, vy in verts:
            # n.v - n.c - s*sqrt(3)/2 <= 0
            rows.append([-nx, -ny, -math.sqrt(3) / 2])
            rhs.append(-(nx * vx + ny * vy))
    res = linprog(
        c=[0, 0, 1],
        A_ub=rows,
        b_ub=rhs,
        bounds=[(None, None)] * 3,
        method="highs",
    )
    assert res.success
    return res.x[2]


def _brute_force_side(A, steps=81, span=0.5):
    """Independent check: grid search over outer-hexagon centres."""
    verts = []
    for x, y, th in A:
        for k in range(6):
            a = th + k * math.pi / 3
            verts.append((x + math.cos(a), y + math.sin(a)))
    V = np.array(verts)
    normals = np.array(
        [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
    )
    centre = V.mean(axis=0)
    best = math.inf
    for _ in range(4):  # refine the grid around the best centre
        best_c = centre
        for dx in np.linspace(-span, span, steps):
            for dy in np.linspace(-span, span, steps):
                c = centre + np.array([dx, dy])
                apothem = ((V - c) @ normals.T).max()
                side = 2 * apothem / math.sqrt(3)
                if side < best:
                    best, best_c = side, c
        centre, span = best_c, span * 4 / steps
    return best


# Three flat-topped unit hexagons around a common vertex (a triangle).
THREE_IN_TRIANGLE = [
    [0.0, 0.0, 0.0],
    [1.5, math.sqrt(3) / 2, 0.0],
    [1.5, -math.sqrt(3) / 2, 0.0],
]


def test_hexagon_packing_three_in_triangle_is_exact():
    problem = HexagonPacking(3)
    result = _evaluate(problem, _returns(f"np.array({THREE_IN_TRIANGLE!r})"))
    assert not result.error, result.feedback
    expected = 4 / math.sqrt(3)  # 2.3094..., the old evaluator said 2.0207
    assert result.fitness == pytest.approx(expected, abs=1e-12)
    assert result.fitness == pytest.approx(_exact_side_lp(THREE_IN_TRIANGLE), 1e-7)
    assert result.fitness == pytest.approx(
        _brute_force_side(THREE_IN_TRIANGLE), abs=1e-4
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        # One hexagon aligned with the outer hexagon fits exactly.
        ([[0.3, -1.7, math.pi / 6]], 1.0),
        # One flat-topped hexagon in the pointy-topped outer hexagon.
        ([[5.0, 2.0, 0.0]], 2 / math.sqrt(3)),
        # Two aligned hexagons sharing an edge.
        ([[0.0, 0.0, math.pi / 6], [math.sqrt(3), 0.0, math.pi / 6]], 2.0),
    ],
)
def test_hexagon_packing_tight_configurations(config, expected):
    problem = HexagonPacking(len(config))
    result = _evaluate(problem, _returns(f"np.array({config!r})"))
    assert not result.error, result.feedback
    assert result.fitness == pytest.approx(expected, abs=1e-12)
    assert result.fitness == pytest.approx(_exact_side_lp(config), 1e-7)


def test_hexagon_packing_random_configurations_match_lp():
    problem = HexagonPacking(4)
    rng = np.random.default_rng(1)
    for _ in range(5):
        A = np.column_stack(
            [
                3.0 * np.arange(4) + rng.random(4),
                5.0 * rng.random(4),
                rng.random(4) * math.pi,
            ]
        )
        V = np.vstack([problem._unit_hex_vertices((x, y), t) for x, y, t in A])
        side = problem._outer_side_from_vertices(V)
        assert side == pytest.approx(_exact_side_lp(A.tolist()), rel=1e-7)


def test_hexagon_prompt_states_outer_orientation():
    prompt = HexagonPacking(11).task_prompt
    assert "with side s, and theta 0" not in prompt
    assert "pi/6 + k*pi/3" in prompt


# --------------------------------------------------- min/max distance ratio


def _ratio_13_layout():
    """16 triangular-lattice points with (max/min distance)^2 = 13.

    The radius-2 hexagon of 19 lattice points without three alternate
    corners: min distance 1, max distance |3*e1 + e2| = sqrt(13).
    """
    removed = {(2, 0), (-2, 2), (0, -2)}
    axial = [
        (i, j)
        for i in range(-2, 3)
        for j in range(-2, 3)
        if abs(i + j) <= 2 and (i, j) not in removed
    ]
    return [(i + 0.5 * j, math.sqrt(3) / 2 * j) for i, j in axial]


def test_min_max_distance_ratio_far_from_origin():
    points = np.array(_ratio_13_layout())
    assert len(points) == 16
    problem = MinMaxMinDistanceRatio(16, 2, best_known=12.889266)

    result = _evaluate(problem, _returns(f"{points.tolist()!r}"))
    assert result.fitness == pytest.approx(13.0, rel=1e-12)

    shifted = points * 0.01 + np.array([5.8e5, 4.6e5])
    result = _evaluate(problem, _returns(f"{shifted.tolist()!r}"))
    assert not result.error, result.feedback
    assert result.fitness == pytest.approx(13.0, rel=1e-6)


# --------------------------------------------------- auto-correlation 2 / 3


@pytest.mark.parametrize("k", [1, 7, 50, 200])
def test_auto_correlation_2_indicator_is_two_thirds(k):
    # For f = indicator of an interval, f*f is a triangle and
    # ||g||_2^2 / (||g||_1 ||g||_inf) = 2/3 exactly.
    problem = AutoCorrIneq2()
    result = _evaluate(problem, _returns(f"[1.0] * {k}", init="pass"))
    assert result.fitness == pytest.approx(2 / 3, abs=1e-9)


def test_auto_correlation_2_matches_fine_quadrature():
    f = np.random.default_rng(0).random(50)
    problem = AutoCorrIneq2()
    result = _evaluate(problem, _returns(f"{f.tolist()!r}", init="pass"))
    # Reference: sample the piecewise-constant f finely and integrate.
    refine = 200
    fine = np.repeat(f, refine)
    h = problem.dx / refine
    g = h * np.convolve(fine, fine)
    expected = (h * np.sum(g * g)) / (h * np.sum(g) * np.max(g))
    assert result.fitness == pytest.approx(expected, rel=1e-3)


def test_auto_correlation_3_rejects_wrong_length():
    problem = AutoCorrIneq3()
    result = _evaluate(problem, _returns("[1.0] * 4000", init="pass"))
    assert result.fitness == math.inf
    assert "length" in result.feedback
    result = _evaluate(problem, _returns("[1.0] * 400", init="pass"))
    assert result.fitness == pytest.approx(2.0)


# --------------------------------------------------------- heilbronn


def test_heilbronn_triangle_scale_invariant():
    triangle = np.array([[0.0, 0.0], [3.0, 0.5], [1.0, 2.5]])
    rng = np.random.default_rng(3)
    w = rng.dirichlet([1, 1, 1], size=6)
    points = w @ triangle
    problem = HeilbronnTriangle(6)

    scores = []
    for scale in (1.0, 2.0):
        code = _returns(
            f"(np.array({(scale * triangle).tolist()!r}), "
            f"np.array({(scale * points).tolist()!r}))"
        )
        result = _evaluate(problem, code)
        assert not result.error, result.feedback
        scores.append(result.fitness)
    assert scores[0] > 0
    assert scores[1] == pytest.approx(scores[0], rel=1e-9)


# --------------------------------------------------- matrix multiplication


def test_matrix_multiplication_targets_use_table():
    for problem in default_problems():
        n, m, p = problem.n, problem.m, problem.p
        assert problem.rank <= n * m * p
        info = MATMUL_RANKS.get((n, m, p))
        if info is not None and info["best_known"] < n * m * p:
            assert problem.rank == info["best_known"], (n, m, p)
    by_dims = {(q.n, q.m, q.p): q.rank for q in default_problems()}
    assert by_dims[(2, 3, 3)] == 15


def test_matrix_multiplication_uses_named_class():
    # Naive rank-8 decomposition of <2,2,2>: one term per (i, j, k).
    problem = MatMulTensorDecomposition(2, 2, 2, grid=0.5, rank=8)
    code = """
import numpy as np

class Helper:
    def __call__(self):
        return [0.0]

class Solver:
    def __call__(self):
        n = m = p = 2
        F1 = np.zeros((n * m, 8))
        F2 = np.zeros((m * p, 8))
        F3 = np.zeros((p * n, 8))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    F1[i * m + j, r] = 1
                    F2[j * p + k, r] = 1
                    F3[k * n + i, r] = 1
                    r += 1
        return [F1, F2, F3]
"""
    result = _evaluate(problem, code, name="Solver")
    assert result.fitness == 0.0, result.feedback
