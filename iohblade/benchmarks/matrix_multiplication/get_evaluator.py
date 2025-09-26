from __future__ import annotations

from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np

from iohblade.solution import Solution


# Type hint only - not imported at runtime
if TYPE_CHECKING:
    from benchmarks.matrix_multiplication.task_specification import (
        MatMulTensorDecompSpec,
    )

from .verify import verify_tensor_decomposition


def build_matmul_tensor(n: int, m: int, p: int) -> np.ndarray:
    """
    Build the (n*m)×(m*p)×(p*n) array representing the <n,m,p> mat-mult tensor.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def get_evaluator(
    spec: "MatMulTensorDecompSpec",
) -> Callable[[Any, list[str], Optional[Any]], Solution]:
    """
    Return an evaluation function for LLaMEA that measures the quality of a tensor decomposition.

    The function evaluates how well a candidate solution approximates the matrix multiplication
    tensor via low-rank decomposition.

    Args:
        spec: prompts.MatMulTensorDecompSpec object containing problem specifications

    Returns:
        A callable function that:
        - Accepts a solution (code or Solution object) and an optional logger
        - Executes the code to get a solution vector
        - Reshapes the vector into 3 factor matrices
        - Quantizes entries to the specified grid
        - Reconstructs an approximation via tensor operations
        - Measures the Frobenius norm of the error
        - Returns (feedback, quality=error, debug_info)
    """
    # n, m, p are the dimensions of the matrix multiplication problem <n,m,p>
    # r is the target rank for the decomposition
    # grid is the quantization step size (all values must be multiples of this)
    n, m, p, grid = spec.n, spec.m, spec.p, spec.grid
    rank = spec.rank
    if rank is None:
        raise ValueError("Rank must be specified when creating evaluate")

    # Pre-compute the dimensions of the three factor matrices
    # nm = n*m: dimension of the first mode of the tensor (and rows of F1)
    # mp = m*p: dimension of the second mode of the tensor (and rows of F2)
    # pn = p*n: dimension of the third mode of the tensor (and rows of F3)
    nm, mp, pn = n * m, m * p, p * n

    # Calculate the total number of variables in the solution vector
    # Each factor matrix has size (mode × rank), so total variables is:
    # (nm*r) + (mp*r) + (pn*r) = (nm + mp + pn) * r
    total_vars = (nm + mp + pn) * rank

    # Calculate the target matrix multiplication tensor to approximate
    t_target = build_matmul_tensor(n, m, p)

    def evaluate(individual: Solution, allowed_libraries: list[str], explogger=None):
        """
        Evaluate a tensor-decomposition individual.

        LLaMEA passes in a Solution object; we must:
        1. turn individual.code (or the list itself) into a NumPy vector
        2. compute the reconstruction error
        3. write .fitness / .description / .debug on the object
        4. return the same object

        # explogger arg because Llamea expects it, but we don't use it here.

        """
        code = individual.code
        algorithm_name = individual.name

        safe_globals = prepare_namespace(code, allowed_libraries)

        # --- 1. execute candidate code ------------------------------------
        # safe_globals  – read-only “whitelist” of modules/functions exposed
        # local_ns      – receives every symbol the candidate defines; we do
        #                 not leak these bindings into the global interpreter

        try:
            local_ns: dict = {}
            exec(code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)

            # get the first class defined by the candidate
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            # Call the class to get an instance, which should return factor_matrics / decomposition
            # If the class has a __call__ method, it will be executed
            decomposition = cls()()
        except Exception as e:
            individual.set_scores(
                float("inf"),
                f"exec-error {e}",
                "exec-failed",
            )
            return individual

        # ------------------------------------------------------------------ #
        # 1a.  normalise decomposition  → flattened NumPy decomposition `x`
        # ------------------------------------------------------------------ #
        if isinstance(decomposition, (list, tuple)) and len(decomposition) == 3:
            # assume [F1, F2, F3]  → flatten
            try:
                f1, f2, f3 = (np.asarray(m, dtype=np.float64) for m in decomposition)
                x = np.concatenate([f1.ravel(), f2.ravel(), f3.ravel()])
            except Exception as e:
                individual.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
                return individual
        else:
            # already a flat decomposition
            x = np.asarray(decomposition, dtype=np.float64)

        # quick size check
        try:
            if x.size != total_vars:
                raise ValueError(f"Size mismatch: expected {total_vars}, got {x.size}")
        except Exception as e:
            individual.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
            return individual

        # ------------------------------------------------------------------ #
        # 2. reshape into matrices and verify answer
        # ------------------------------------------------------------------ #
        x = np.round(x / grid) * grid  # quantise to grid

        o1, o2 = nm * rank, (nm + mp) * rank
        f1 = x[:o1].reshape(nm, rank)
        f2 = x[o1:o2].reshape(mp, rank)
        f3 = x[o2:].reshape(pn, rank)

        # try an exact verification first
        try:
            verify_tensor_decomposition((f1, f2, f3), n, m, p, rank)
            err = 0.0
        except AssertionError:
            # not exact – compute Frobenius error
            t_hat = np.einsum("il,jl,kl->ijk", f1, f2, f3)
            err = np.linalg.norm(t_hat - t_target)

        # ------------------------------------------------------------------ #
        # 3. set fitness ( minimise: lower is better )
        # ------------------------------------------------------------------ #

        individual.set_scores(
            err, f"The algorithm {algorithm_name} has an error of err={err:.6f}"
        )

        return individual

    # Return the evaluate function to be used by LLaMEA
    return evaluate
