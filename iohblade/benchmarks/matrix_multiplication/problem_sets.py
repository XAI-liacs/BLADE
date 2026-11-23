from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from iohblade.benchmarks.matrix_multiplication.task_specification import (
    MatMulTensorDecomposition,
)

from .constants import MATMUL_RANKS

try:
    # Import at runtime so tests can instantiate specs
    from .task_specification import MatMulTensorDecomposition
except Exception:  # pragma: no cover - fallback for type checking only
    if TYPE_CHECKING:
        from benchmarks.matrix_multiplication.task_specification import (
            MatMulTensorDecomposition,
        )


def _target_rank(n: int, m: int, p: int) -> int:
    """Return the target rank for the <n,m,p> matrix-multiplication tensor.

    The target is the literature best-known rank from
    ``constants.MATMUL_RANKS`` (not that rank minus one), which is keyed by
    the sorted dimensions (the rank of <n,m,p> does not depend on the order
    of n, m and p). Without a table entry it falls back to ``n*m*p - 1``,
    one below the naive rank, and a table entry is never allowed to exceed
    that fallback.
    """
    naive_minus_1 = n * m * p - 1
    key: Tuple[int, int, int] = tuple(sorted((n, m, p)))  # type: ignore[assignment]
    info = MATMUL_RANKS.get(key)
    if info and "best_known" in info:
        return min(int(info["best_known"]), naive_minus_1)
    return naive_minus_1


# Backwards-compatible alias: the old name suggested ``best_known - 1``, but
# the function has always returned the best-known rank itself.
_max_rank_minus_1 = _target_rank


def default_problems() -> List[MatMulTensorDecomposition]:
    """
    Full benchmark suite from AlphaEvolve (Table 3), m <= n <= p.
    grid=0.5 permits integer and half-integer entries.
    """
    grid = 0.5
    sizes = [
        # 2,2,*
        (2, 2, 2),
        (2, 2, 3),
        (2, 2, 4),
        (2, 2, 5),
        (2, 2, 6),
        (2, 2, 7),
        (2, 2, 8),
        (2, 2, 9),
        (2, 2, 10),
        (2, 2, 11),
        (2, 2, 12),
        (2, 2, 13),
        (2, 2, 14),
        (2, 2, 15),
        (2, 2, 16),
        # 2,3,*
        (2, 3, 3),
        (2, 3, 4),
        (2, 3, 5),
        (2, 3, 6),
        (2, 3, 7),
        (2, 3, 8),
        (2, 3, 9),
        (2, 3, 10),
        # 2,4,*
        (2, 4, 4),
        (2, 4, 5),
        (2, 4, 6),
        (2, 4, 7),
        (2, 4, 8),
        # 2,5,*
        (2, 5, 5),
        (2, 5, 6),
        # 3,3,*
        (3, 3, 3),
        (3, 3, 4),
        (3, 3, 5),
        (3, 3, 6),
        (3, 3, 7),
        (3, 3, 8),
        # 3,4,*
        (3, 4, 4),
        (3, 4, 5),
        (3, 4, 6),
        (3, 4, 7),
        (3, 4, 8),
        # 3,5,*
        (3, 5, 5),
        (3, 5, 6),
        (3, 5, 7),
        # 4,4,*
        (4, 4, 4),
        (4, 4, 5),
        (4, 4, 6),
        (4, 4, 7),
        (4, 4, 8),
        (4, 4, 9),
        # 4,5,*
        (4, 5, 5),
        (4, 5, 6),
        # 5,5,5 and 6,6,6
        (5, 5, 5),
        (6, 6, 6),
    ]

    problems = [
        MatMulTensorDecomposition(n, m, p, grid=grid, rank=_target_rank(n, m, p))
        for (n, m, p) in sizes
    ]
    print(
        f"Running full benchmark suite: {len(problems)} matrix-multiplication problems"
    )

    return problems
