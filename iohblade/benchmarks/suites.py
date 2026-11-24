"""Default benchmark suites: named sets of problems with fixed settings.

Each suite is a function that returns a list of problems, registered in
:mod:`iohblade.registry` so it can be used in a YAML experiment::

    problems:
      - suite: bbob            # the five BBOB function groups
        params: {dim: 10, budget_factor: 2000}

Suites:

* ``bbob``: one problem per BBOB function group (5 problems).
* ``ma_bbob``: MA-BBOB (1 problem).
* ``continuous_bbo``: ``bbob`` and ``ma_bbob`` (6 problems).
* ``alphaevolve_math``: the AlphaEvolve mathematical problems (17 problems).
* ``combinatorial``: Steiner trees, graph colouring, TSP and VRP (6 problems).
* ``quick``: a small cross-domain set for trying out a setup (5 problems).
* ``full``: ``continuous_bbo``, ``alphaevolve_math``, ``combinatorial``,
  circle packing and spherical codes (31 problems; optionally the 54
  matrix multiplication problems).
"""

from __future__ import annotations

import math
from typing import Any

from iohblade.problem import Problem

# BBOB function ids per function group (Hansen et al., 2009).
BBOB_GROUPS: dict[int, list[int]] = {
    1: [1, 2, 3, 4, 5],  # separable
    2: [6, 7, 8, 9],  # low or moderate conditioning
    3: [10, 11, 12, 13, 14],  # high conditioning, unimodal
    4: [15, 16, 17, 18, 19],  # multi-modal, adequate global structure
    5: [20, 21, 22, 23, 24],  # multi-modal, weak global structure
}

#: Seconds per function evaluation used for the default timeout of the
#: continuous suites (generous: LLM-written optimisers are often slow).
SECONDS_PER_EVALUATION = 1e-3
MIN_EVAL_TIMEOUT = 120

STEINER_INSTANCES = (50, 100, 250)
ROUTING_INSTANCE = "A-n53-k7"
GRAPH_COLORING_INSTANCE = "1"


def _check_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer (got {value!r}).")


def _timeout(eval_timeout: int | None, runs: int, dim: int, budget_factor: int) -> int:
    """``eval_timeout``, or a default from the number of function
    evaluations of one evaluation (at least ``MIN_EVAL_TIMEOUT``)."""
    if eval_timeout is not None:
        if (
            not isinstance(eval_timeout, (int, float))
            or isinstance(eval_timeout, bool)
            or eval_timeout <= 0
        ):
            raise ValueError(
                f"eval_timeout must be a positive number of seconds "
                f"(got {eval_timeout!r})."
            )
        return eval_timeout
    evaluations = runs * dim * budget_factor
    return max(MIN_EVAL_TIMEOUT, math.ceil(evaluations * SECONDS_PER_EVALUATION))


def bbob(
    dim: int = 10,
    budget_factor: int = 2000,
    groups: list[int] | None = None,
    n_train: int = 5,
    n_test: int = 10,
    eval_timeout: int | None = None,
) -> list[Problem]:
    """One BBOB problem per function group.

    Each problem uses the functions of its group, instances ``1..n_train``
    for training and the next ``n_test`` instances for testing.

    Args:
        dim: Dimension of the functions.
        budget_factor: Function evaluations per run = ``budget_factor * dim``.
        groups: The function groups (1-5) to include; all by default.
        n_train: Training instances per function (instances 1..n_train).
        n_test: Test instances per function (the next n_test instances).
        eval_timeout: Seconds before an evaluation is stopped. By default
            ``SECONDS_PER_EVALUATION`` per function evaluation of the
            training runs, at least 120 s (500 s for a 5-function group
            in 10-D with budget factor 2000).

    Returns:
        list[Problem]: ``BBOB_SBOX`` problems named ``BBOB-g<group>-<dim>d``.
    """
    import ioh

    from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX

    for name, value in (
        ("dim", dim),
        ("budget_factor", budget_factor),
        ("n_train", n_train),
        ("n_test", n_test),
    ):
        _check_positive_int(name, value)
    groups = list(BBOB_GROUPS) if groups is None else list(groups)
    unknown = [g for g in groups if g not in BBOB_GROUPS]
    if unknown or not groups:
        raise ValueError(
            f"groups must be a non-empty list of BBOB function groups 1-5 "
            f"(got {groups!r})."
        )
    train_iids = range(1, n_train + 1)
    test_iids = range(n_train + 1, n_train + n_test + 1)
    problems = []
    for group in groups:
        fids = BBOB_GROUPS[group]
        problems.append(
            BBOB_SBOX(
                training_instances=[(f, i) for f in fids for i in train_iids],
                test_instances=[(f, i) for f in fids for i in test_iids],
                name=f"BBOB-g{group}-{dim}d",
                dims=[dim],
                budget_factor=budget_factor,
                specific_group=group,
                problem_type=ioh.ProblemClass.BBOB,
                eval_timeout=_timeout(
                    eval_timeout, len(fids) * n_train, dim, budget_factor
                ),
            )
        )
    return problems


def ma_bbob(
    dim: int = 10,
    budget_factor: int = 2000,
    n_train: int = 25,
    n_test: int = 50,
    eval_timeout: int | None = None,
) -> list[Problem]:
    """MA-BBOB with instances ``0..n_train-1`` for training and the next
    ``n_test`` instances for testing.

    Args:
        dim: Dimension of the functions.
        budget_factor: Function evaluations per run = ``budget_factor * dim``.
        n_train: Number of training instances.
        n_test: Number of test instances (after the training instances).
        eval_timeout: Seconds before an evaluation is stopped; by default
            from the number of function evaluations, as for :func:`bbob`.

    Returns:
        list[Problem]: One ``MA_BBOB`` problem named ``MA_BBOB-<dim>d``.
    """
    from iohblade.benchmarks.BBOB.mabbob import MA_BBOB

    for name, value in (
        ("dim", dim),
        ("budget_factor", budget_factor),
        ("n_train", n_train),
        ("n_test", n_test),
    ):
        _check_positive_int(name, value)
    if n_train + n_test > 1000:
        raise ValueError(
            "MA-BBOB has 1000 instances: n_train + n_test must be <= 1000 "
            f"(got {n_train + n_test})."
        )
    return [
        MA_BBOB(
            training_instances=list(range(0, n_train)),
            test_instances=list(range(n_train, n_train + n_test)),
            name=f"MA_BBOB-{dim}d",
            dims=[dim],
            budget_factor=budget_factor,
            eval_timeout=_timeout(eval_timeout, n_train, dim, budget_factor),
        )
    ]


def continuous_bbo(
    dim: int = 10,
    budget_factor: int = 2000,
    eval_timeout: int | None = None,
) -> list[Problem]:
    """The five BBOB function groups and MA-BBOB, with the default
    instances of :func:`bbob` and :func:`ma_bbob`.

    Args:
        dim: Dimension of all functions.
        budget_factor: Function evaluations per run = ``budget_factor * dim``.
        eval_timeout: Seconds before an evaluation is stopped; by default
            from the number of function evaluations of each problem.
    """
    return bbob(dim, budget_factor, eval_timeout=eval_timeout) + ma_bbob(
        dim, budget_factor, eval_timeout=eval_timeout
    )


def alphaevolve_math(use_best: bool = False, eval_timeout: int = 600) -> list[Problem]:
    """The AlphaEvolve mathematical problems: auto-correlation inequalities,
    Erdős minimum overlap, the uncertainty inequality, hexagon, rectangle
    and unit square packing, Heilbronn problems, the kissing number in 11-D,
    min/max distance ratios and sums versus differences.

    Args:
        use_best: Show the best known solution as the example in the prompt.
        eval_timeout: Seconds before an evaluation is stopped.
    """
    from iohblade.benchmarks.analysis.get_analysis_problems import get_analysis_problems
    from iohblade.benchmarks.combinatorics.get_erdos_problem import (
        get_combinatorics_problems,
    )
    from iohblade.benchmarks.fourier.get_fourier_problem import get_fourier_problems
    from iohblade.benchmarks.geometry.get_geometry_problems import (
        get_heilbronn_convex_region_problems,
        get_heilbronn_triangle_problems,
        get_kissing_number_11D_problems,
        get_min_max_dist_ratio_problem,
    )
    from iohblade.benchmarks.number_theory.get_num_theory_problems import (
        get_sum_vs_difference_problem,
    )
    from iohblade.benchmarks.packing.get_packing_problems import (
        get_hexagon_packing_problems,
        get_rectangle_packing_problems,
        get_square_packing_problems,
    )

    factories = (
        get_analysis_problems,
        get_combinatorics_problems,
        get_fourier_problems,
        get_hexagon_packing_problems,
        get_rectangle_packing_problems,
        get_square_packing_problems,
        get_heilbronn_triangle_problems,
        get_heilbronn_convex_region_problems,
        get_kissing_number_11D_problems,
        get_min_max_dist_ratio_problem,
        get_sum_vs_difference_problem,
    )
    problems: list[Problem] = []
    for factory in factories:
        problems += factory(use_best=use_best, eval_timeout=eval_timeout)
    return problems


def combinatorial(eval_timeout: int = 300) -> list[Problem]:
    """Euclidean Steiner trees (50, 100 and 250 points), graph colouring,
    TSP and capacitated VRP.

    Args:
        eval_timeout: Seconds before an evaluation is stopped.
    """
    from iohblade.benchmarks.combinatorics.euclidian_steiner_tree import (
        EuclidianSteinerTree,
    )
    from iohblade.benchmarks.combinatorics.graph_coloring import GraphColoring
    from iohblade.benchmarks.logistics.tsp import TravelingSalesmanProblem
    from iohblade.benchmarks.logistics.vrp import VehicleRoutingProblem

    problems: list[Problem] = [
        EuclidianSteinerTree(n, eval_timeout=eval_timeout) for n in STEINER_INSTANCES
    ]
    problems.append(GraphColoring(GRAPH_COLORING_INSTANCE, eval_timeout=eval_timeout))
    problems.append(
        TravelingSalesmanProblem(ROUTING_INSTANCE, eval_timeout=eval_timeout)
    )
    problems.append(VehicleRoutingProblem(ROUTING_INSTANCE, eval_timeout=eval_timeout))
    return problems


def quick(dim: int = 5, budget_factor: int = 1000) -> list[Problem]:
    """A small set across domains, for trying out a method or a setup:
    BBOB groups 1 (separable) and 4 (multi-modal, adequate structure),
    circle packing, the Heilbronn triangle problem and the TSP.

    Args:
        dim: Dimension of the BBOB functions.
        budget_factor: Function evaluations per BBOB run =
            ``budget_factor * dim``.
    """
    from iohblade.benchmarks.geometry.get_geometry_problems import (
        get_heilbronn_triangle_problems,
    )
    from iohblade.benchmarks.logistics.tsp import TravelingSalesmanProblem
    from iohblade.benchmarks.packing.circles import CirclePacking

    return (
        bbob(dim, budget_factor, groups=[1, 4])
        + [CirclePacking(eval_timeout=300)]
        + get_heilbronn_triangle_problems(eval_timeout=300)
        + [TravelingSalesmanProblem(ROUTING_INSTANCE, eval_timeout=300)]
    )


def full(
    dim: int = 10,
    budget_factor: int = 2000,
    use_best: bool = False,
    include_matrix_multiplication: bool = False,
) -> list[Problem]:
    """All default suites: :func:`continuous_bbo`, :func:`alphaevolve_math`
    and :func:`combinatorial`, plus circle packing and spherical codes.

    Args:
        dim: Dimension of the continuous problems.
        budget_factor: Function evaluations per continuous run =
            ``budget_factor * dim``.
        use_best: Show the best known solutions in the math prompts.
        include_matrix_multiplication: Also add the 54 AlphaEvolve matrix
            multiplication problems.
    """
    from iohblade.benchmarks.geometry.spherical_code import SphericalCode
    from iohblade.benchmarks.packing.circles import CirclePacking

    problems = (
        continuous_bbo(dim, budget_factor)
        + alphaevolve_math(use_best=use_best)
        + combinatorial()
        + [CirclePacking(), SphericalCode()]
    )
    if include_matrix_multiplication:
        from iohblade.benchmarks.matrix_multiplication.problem_sets import (
            default_problems,
        )

        problems += default_problems()
    return problems
