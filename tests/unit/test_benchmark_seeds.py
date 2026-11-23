"""Regression tests for best-known seeds and reference data (issue #204).

Each seed is evaluated through the problem's own ``evaluate()`` (in process)
the way the ``use_best`` / ``get_best`` factories use it, and must give the
score that the AlphaEvolve results reproduce. Further tests cover the
evaluator fixes that made the seeds consistent.
"""

import math
import textwrap

import numpy as np
import pytest
from numpy.polynomial.hermite import hermval

from iohblade.benchmarks.analysis.auto_correlation_ineq1 import AutoCorrIneq1
from iohblade.benchmarks.analysis.auto_correlation_ineq3 import AutoCorrIneq3
from iohblade.benchmarks.analysis.get_analysis_problems import (
    ac3_best,
    get_analysis_problems,
)
from iohblade.benchmarks.combinatorics.erdos_min_overlap import ErdosMinOverlap
from iohblade.benchmarks.combinatorics.get_erdos_problem import (
    best_solution as erdos_seed,
)
from iohblade.benchmarks.combinatorics.get_erdos_problem import (
    best_solution_half as erdos_half,
)
from iohblade.benchmarks.combinatorics.get_erdos_problem import (
    get_combinatorics_problems,
)
from iohblade.benchmarks.fourier.get_fourier_problem import get_fourier_problems
from iohblade.benchmarks.fourier.uncertainty_inequality import UncertaintyInequality
from iohblade.benchmarks.geometry.get_geometry_problems import (
    best_min_max_min_distance_2D,
    get_min_max_dist_ratio_problem,
)
from iohblade.benchmarks.geometry.min_max_distance_ratio import MinMaxMinDistanceRatio
from iohblade.benchmarks.matrix_multiplication.constants import MATMUL_RANKS
from iohblade.benchmarks.matrix_multiplication.problem_sets import (
    _max_rank_minus_1,
    _target_rank,
    default_problems,
)
from iohblade.benchmarks.number_theory.get_num_theory_problems import (
    best_solution as svd_seed,
)
from iohblade.benchmarks.number_theory.get_num_theory_problems import (
    get_sum_vs_difference_problem,
)
from iohblade.benchmarks.number_theory.sums_vs_differences import SumDifference
from iohblade.benchmarks.packing.get_packing_problems import (
    get_hexagon_packing_problems,
)
from iohblade.solution import Solution

# A candidate that returns the seed it is given. The keyword and positional
# arguments differ per problem, so accept anything and look for the seed.
SEED_CODE = """
import numpy as np

class {name}:
    def __init__(self, *args, best_known_configuration=None, **kwargs):
        if best_known_configuration is None and args:
            best_known_configuration = args[-1]
        self.seed = best_known_configuration

    def __call__(self):
        return self.seed
"""


def _evaluate(problem, code, name="Candidate"):
    return problem.evaluate(Solution(code=textwrap.dedent(code), name=name))


def _evaluate_seed(problem, name):
    return _evaluate(problem, SEED_CODE.format(name=name), name=name)


def _returns(value_code, name="Candidate"):
    return f"""
import numpy as np

class {name}:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self):
        return {value_code}
"""


def _assert_valid(result, expected):
    assert not result.error, result.feedback
    assert math.isfinite(result.fitness), result.feedback
    assert result.fitness == pytest.approx(expected, rel=1e-6), result.feedback


# ------------------------------------------------------ 1. auto-correlation


def test_auto_correlation_1_seed_reaches_best_known():
    problem = get_analysis_problems(use_best=True)[0]
    result = _evaluate_seed(problem, "AutoCorrCandidate_1")
    _assert_valid(result, 1.505294)
    assert result.fitness == pytest.approx(problem.best_known, abs=1e-4)


def test_auto_correlation_2_seed_still_scores():
    problem = get_analysis_problems(use_best=True)[1]
    result = _evaluate_seed(problem, "AutoCorrCandidate_2")
    _assert_valid(result, 0.8962799)


def test_auto_correlation_seed_is_the_configuration_not_the_score():
    problem = AutoCorrIneq1(best_solution=[1.0] * 600)
    code = """
class Candidate:
    def __init__(self, best_known_configuration=None):
        assert isinstance(best_known_configuration, list)
        self.seed = best_known_configuration

    def __call__(self):
        return self.seed
"""
    result = _evaluate(problem, code)
    assert not result.error, result.feedback
    # A box function: max(f*f) = N * dx^2, (∫f)^2 = (N dx)^2, ratio = 1/(N dx).
    assert result.fitness == pytest.approx(1 / 0.5)


def test_auto_correlation_1_exec_error_feedback_is_a_message():
    problem = AutoCorrIneq1()
    result = _evaluate(problem, "class Candidate:\n    pass\n", name="Missing")
    assert result.fitness == math.inf
    assert isinstance(result.feedback, str)
    assert result.feedback.startswith("exec-error")
    assert result.error


# ----------------------------------------------------- 2. hexagon packing


@pytest.mark.parametrize(
    "index, n_hex, expected, best_known",
    [(0, 11, 3.930075, 3.930092), (1, 12, 3.941896, 3.941912)],
)
def test_hexagon_packing_seeds(index, n_hex, expected, best_known):
    problem = get_hexagon_packing_problems(get_best=True)[index]
    assert problem.n_hex == n_hex
    assert problem.best_known == best_known
    result = _evaluate_seed(problem, "HexagonPackingCandidate")
    # evaluate() reports overlaps as an invalid (infinite) score.
    _assert_valid(result, expected)
    # The evaluator computes the exact minimal outer hexagon, so the seed
    # scores slightly below the notebook's reported side length.
    assert expected <= result.fitness <= best_known


def test_hexagon_seeds_are_converted_to_radians():
    problem = get_hexagon_packing_problems(get_best=True)[0]
    angles = np.array([row[2] for row in problem.best_solution])
    assert np.all(np.abs(angles) < 2 * math.pi + 1)


# ------------------------------------------------- 3. sums vs differences


def test_sums_vs_differences_seed():
    problem = get_sum_vs_difference_problem(use_best=True)[0]
    assert len(svd_seed) == 2003 and max(svd_seed) == 1_040_200
    assert problem.max_set_size >= len(svd_seed)
    assert problem.max_value == 2**21
    assert problem.best_score == pytest.approx(1.147989)
    result = _evaluate_seed(problem, "SumDiffCandidate")
    _assert_valid(result, 1.147989)


def test_sums_vs_differences_prompt_states_limits():
    problem = SumDifference()
    assert str(2**21) in problem.task_prompt
    assert str(2**21) in problem.example_prompt


@pytest.mark.parametrize(
    "value_code, reason",
    [
        ("[1, 2, 3]", "0 must be in U"),
        ("[0, 2**21 + 1]", "exceeds 2097152"),
        ("list(range(2049))", "|U|>2048"),
    ],
)
def test_sums_vs_differences_invalid_feedback_is_the_reason(value_code, reason):
    problem = SumDifference()
    result = _evaluate(problem, _returns(value_code))
    assert result.fitness == -math.inf
    assert reason in result.feedback
    assert "is not defined" not in result.feedback
    assert reason in str(result.error)


def test_sums_vs_differences_accepts_values_up_to_limit():
    problem = SumDifference()
    result = _evaluate(problem, _returns("[0, 1, 2**21]"))
    assert not result.error, result.feedback
    assert math.isfinite(result.fitness)


# ------------------------------------------------------------- 4. Erdős


def test_erdos_seed_is_notebook_construction():
    assert len(erdos_half) == 48
    assert len(erdos_seed) == 95
    assert erdos_seed == list(np.concatenate((erdos_half[:-1], erdos_half[::-1])))


def test_erdos_seed():
    problem = get_combinatorics_problems(use_best=True)[0]
    result = _evaluate_seed(problem, "ErdosCandidate")
    _assert_valid(result, 0.3809230)
    assert result.fitness == pytest.approx(problem.best_known, abs=2e-6)


@pytest.mark.parametrize("n", [1, 7, 95, 800, 5000])
def test_erdos_accepts_any_length(n):
    problem = ErdosMinOverlap()
    result = _evaluate(problem, _returns(f"[0.5] * {n}"), name="Candidate")
    assert not result.error, result.feedback
    # f = 1/2 on [-1, 1]: the overlap at shift 0 is ∫ f (1 - f) = 1/2.
    assert result.fitness == pytest.approx(0.5)


def test_erdos_resolution_does_not_change_the_score():
    problem = ErdosMinOverlap()
    coarse = _evaluate(problem, _returns(f"{list(erdos_seed)!r}"))
    fine = _evaluate(problem, _returns(f"list(np.repeat({list(erdos_seed)!r}, 50))"))
    assert not coarse.error and not fine.error
    assert fine.fitness == pytest.approx(coarse.fitness, rel=1e-9)


@pytest.mark.parametrize(
    "value_code",
    ["[]", "[0.5] * 100_001", "[1.5] * 2 + [-0.5] * 2", "[0.25] * 8"],
)
def test_erdos_still_rejects_invalid(value_code):
    problem = ErdosMinOverlap()
    result = _evaluate(problem, _returns(value_code))
    assert result.fitness == problem.failure_fitness()
    assert result.error


def test_erdos_sup_overlap_fft_matches_direct():
    problem = ErdosMinOverlap()
    rng = np.random.default_rng(1)
    f = rng.random(5000)
    g = 1.0 - f
    dx = 2.0 / f.size
    direct = dx * float(np.max(np.correlate(g, f, mode="full")))
    assert problem._sup_overlap(f, g, dx) == pytest.approx(direct, rel=1e-10)


# ------------------------------------------------------------ 5. Fourier


def test_fourier_seed():
    problem = get_fourier_problems(use_best=True)[0]
    assert problem.best_known == 0.3521
    result = _evaluate_seed(problem, "FourierCandidate")
    _assert_valid(result, 0.3520991)


def test_fourier_added_coefficient_makes_p0_zero():
    problem = UncertaintyInequality()
    rng = np.random.default_rng(2)
    for _ in range(5):
        c = rng.normal(size=3)
        full = problem._complete_coefficients(c)
        assert full.size == 4
        assert full[3] == pytest.approx(-(c[0] + 12 * c[1] + 1680 * c[2]) / 665280)
        hcoef = problem._build_hcoef(full)
        assert hermval(0.0, hcoef) == pytest.approx(0.0, abs=1e-12)


def test_fourier_hermite_values_at_zero():
    # H_0(0), H_4(0), H_8(0), H_12(0) of the physicists' Hermite polynomials.
    values = [hermval(0.0, np.eye(13)[n]) for n in (0, 4, 8, 12)]
    assert values == [1.0, 12.0, 1680.0, 665280.0]


def test_fourier_score_is_sign_invariant_and_no_longer_needs_p0_negative():
    problem = UncertaintyInequality()
    seed = get_fourier_problems(use_best=True)[0].best_known_configuration
    # The seed has P_3(0) = c0 + 12 c1 + 1680 c2 > 0 (the old evaluator
    # required < 0); with the added H_12 term both signs give the same score.
    assert seed[0] + 12 * seed[1] + 1680 * seed[2] > 0
    plus = _evaluate(problem, _returns(repr(list(seed))))
    minus = _evaluate(problem, _returns(repr([-v for v in seed])))
    _assert_valid(plus, 0.3520991)
    _assert_valid(minus, 0.3520991)


def test_fourier_rejects_bad_coefficients():
    problem = UncertaintyInequality()
    for value_code in ("[0.0, 0.0, 0.0]", "[1.0, 2.0]", "[np.nan, 0.0, 1.0]"):
        result = _evaluate(problem, _returns(value_code))
        assert result.fitness == math.inf, value_code
        assert result.error


# ---------------------------------------------- 6. auto-correlation 3 (C3)


def test_auto_correlation_3_seed():
    problem = get_analysis_problems(use_best=True)[2]
    result = _evaluate_seed(problem, "AutoCorreCandidate_3")
    _assert_valid(result, 1.455643)
    assert result.fitness == pytest.approx(problem.best_known, abs=1e-4)


def test_auto_correlation_3_takes_abs_outside_the_max():
    problem = AutoCorrIneq3()
    f = np.array(ac3_best)
    dx = 0.5 / f.size
    g = dx * np.convolve(f, f)
    integral = dx * f.sum()
    # For this seed max|g| (C3') is larger than |max g| (C3).
    assert np.max(np.abs(g)) > abs(np.max(g))
    result = _evaluate(problem, _returns(repr(list(f))))
    assert result.fitness == pytest.approx(abs(np.max(g)) / integral**2)
    assert "|max_t (f*f)(t)|" in problem.task_prompt


# ------------------------------------------------- 7. min/max distance


@pytest.mark.parametrize("index, expected", [(0, 12.889266112), (1, 4.165849767)])
def test_min_max_distance_seeds(index, expected):
    problem = get_min_max_dist_ratio_problem(use_best=True)[index]
    assert problem.best_known == pytest.approx(expected, rel=1e-9)
    result = _evaluate_seed(problem, "MinMaxDistanceSolver")
    _assert_valid(result, expected)


def test_min_max_distance_guard_drops_wrong_length_seed():
    problem = MinMaxMinDistanceRatio(
        16, 2, best_solution=list(best_min_max_min_distance_2D)[:15]
    )
    assert problem.best_solution is None


def test_min_max_distance_guard_drops_wrong_dimension_seed():
    problem = MinMaxMinDistanceRatio(
        16, 3, best_solution=list(best_min_max_min_distance_2D)
    )
    assert problem.best_solution is None


def test_min_max_distance_guard_keeps_good_seed():
    seed = list(best_min_max_min_distance_2D)
    assert MinMaxMinDistanceRatio(16, 2, best_solution=seed).best_solution == seed


# ------------------------------------------- 8. matrix multiplication


# AlphaEvolve paper, Table 3: (best known, AlphaEvolve) for corrected rows.
CORRECTED_MATMUL = {
    (2, 4, 4): (26, 26),
    (2, 4, 7): (46, 45),
    (2, 5, 6): (48, 47),
    (3, 3, 5): (36, 36),
    (3, 5, 5): (58, 58),
    (4, 4, 4): (49, 48),
    (4, 4, 6): (73, 73),
    (4, 4, 7): (87, 85),
    (4, 5, 5): (76, 76),
    (4, 4, 9): (104, 108),
    (6, 6, 6): (153, 156),
}


@pytest.mark.parametrize("key, ranks", sorted(CORRECTED_MATMUL.items()))
def test_matmul_table_matches_alphaevolve_table_3(key, ranks):
    info = MATMUL_RANKS[key]
    assert (info["best_known"], info["alphaevolve_best"]) == ranks


def test_matmul_table_entries_are_consistent():
    for key, info in MATMUL_RANKS.items():
        n, m, p = key
        assert key == tuple(sorted(key))
        assert "alphaevolve_best" in info, key
        assert 0 < info["best_known"] <= n * m * p - 1, key
        assert 0 < info["alphaevolve_best"] <= n * m * p - 1, key


def test_matmul_default_problems_are_in_table_and_target_best_known():
    for problem in default_problems():
        key = tuple(sorted((problem.n, problem.m, problem.p)))
        assert key in MATMUL_RANKS, key
        assert problem.rank == MATMUL_RANKS[key]["best_known"], key


def test_matmul_target_rank_alias():
    assert _max_rank_minus_1 is _target_rank
    assert _target_rank(4, 4, 4) == 49
    assert _target_rank(5, 4, 4) == _target_rank(4, 4, 5)
    # Not in the table: one below the naive rank.
    assert _target_rank(7, 7, 7) == 7 * 7 * 7 - 1
