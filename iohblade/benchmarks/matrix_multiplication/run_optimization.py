"""
Decremental rank search helper for matrix-multiplication.

This module exposes a callable utility `decremental_rank_search` that performs the
rank loop but delegates single-run execution to a `run_fn` callback to avoid
coupling to the main driver and to prevent circular imports.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple


def decremental_rank_search(
    spec,
    args,
    minimization: bool,
    tol: float = 1e-12,
    run_fn: Optional[Callable[..., Tuple[object, float, int]]] = None,
) -> Tuple[object, float, int]:
    """
    Run decremental rank search for a single matrix-multiplication spec.

    Parameters:
        spec: ProblemSpec-like object with fields n, m, p, grid, rank
        args: argparse-like object carrying provider, model, parents, budget, etc.
        minimization: Whether the objective is minimization
        tol: Exactness tolerance (<= tol counts as exact)
        run_fn: Callable that runs a single optimization and returns
                (best_solution, best_fitness, used_budget)

    Returns:
        (best_solution, best_fitness, total_used_budget)
    """
    if run_fn is None:
        raise ValueError("decremental_rank_search requires a run_fn to execute a run")

    # If the spec does not carry a rank, just run once
    if not hasattr(spec, "rank"):
        return run_fn(
            spec=spec,
            provider=args.provider,
            model=args.model,
            n_parents=args.n_parents,
            n_offspring=args.n_offspring,
            minimization=minimization,
            experiment_name=args.benchmark,
            elitism=args.elitism,
            HPO=args.hpo,
            budget=args.budget,
            max_workers=args.max_workers,
            parallel_backend=args.parallel_backend,
        )

    from .task_specification import MatMulTensorDecomposition

    def _with_rank(s, r):
        return MatMulTensorDecomposition(n=s.n, m=s.m, p=s.p, grid=s.grid, rank=r)

    remaining_budget = args.budget
    if remaining_budget is None:
        raise ValueError(
            "A finite evaluation budget is required for decremental rank search (set --budget)."
        )

    current_rank = int(spec.rank)

    # If searching only rank==1, there's nothing to decrement; run once.
    if current_rank <= 1:
        return run_fn(
            spec=spec,
            provider=args.provider,
            model=args.model,
            n_parents=args.n_parents,
            n_offspring=args.n_offspring,
            minimization=minimization,
            experiment_name=args.benchmark,
            elitism=args.elitism,
            HPO=args.hpo,
            budget=remaining_budget,
            max_workers=args.max_workers,
            parallel_backend=args.parallel_backend,
        )
    best_solution = None
    best_fitness: float = float("inf") if minimization else float("-inf")
    total_used = 0

    while current_rank >= 1 and remaining_budget > 0:
        this_spec = _with_rank(spec, current_rank)
        best_solution, best_fitness, used = run_fn(
            spec=this_spec,
            provider=args.provider,
            model=args.model,
            n_parents=args.n_parents,
            n_offspring=args.n_offspring,
            minimization=minimization,
            experiment_name=args.benchmark,
            elitism=args.elitism,
            HPO=args.hpo,
            budget=remaining_budget,
            max_workers=args.max_workers,
            parallel_backend=args.parallel_backend,
        )
        total_used += used
        remaining_budget -= used

        # Exact success triggers decrement
        if best_fitness <= tol:
            current_rank -= 1
        else:
            # keep trying same rank until budget exhausted
            if remaining_budget <= 0:
                break

    return best_solution, best_fitness, total_used
