"""This function verifies the correctness of a tensor decomposition for the matrix multiplication tensor.
It is taken from https://colab.research.google.com/github/google-deepmind/alphaevolve_results/blob/master/mathematical_results.ipynb
"""

import logging

import numpy as np


def verify_tensor_decomposition(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int,
    m: int,
    p: int,
    rank: int,
):
    """Verifies the correctness of the tensor decomposition.

    Args:
      decomposition: a tuple of 3 factor matrices with the same number of columns.
        (The number of columns specifies the rank of the decomposition.) To
        construct a tensor, we take the outer product of the i-th column of the
        three factor matrices, for 1 <= i <= rank, and add up all these outer
        products.
      n: the first parameter of the matrix multiplication tensor.
      m: the second parameter of the matrix multiplication tensor.
      p: the third parameter of the matrix multiplication tensor.
      rank: the expected rank of the decomposition.

    Raises:
      AssertionError: If the decomposition does not have the correct rank, or if
      the decomposition does not construct the 3D tensor which corresponds to
      multiplying an n x m matrix by an m x p matrix.
    """
    # Check that each factor matrix has the correct shape.
    factor_matrix_1, factor_matrix_2, factor_matrix_3 = decomposition
    assert factor_matrix_1.shape == (
        n * m,
        rank,
    ), f"Expected shape of factor matrix 1 is {(n * m, rank)}. Actual shape is {factor_matrix_1.shape}."
    assert factor_matrix_2.shape == (
        m * p,
        rank,
    ), f"Expected shape of factor matrix 1 is {(m * p, rank)}. Actual shape is {factor_matrix_2.shape}."
    assert factor_matrix_3.shape == (
        p * n,
        rank,
    ), f"Expected shape of factor matrix 1 is {(p * n, rank)}. Actual shape is {factor_matrix_3.shape}."

    # Form the matrix multiplication tensor <n, m, p>.
    matmul_tensor = np.zeros((n * m, m * p, p * n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                matmul_tensor[i * m + j][j * p + k][k * n + i] = 1

    # Check that the tensor is correctly constructed.
    constructed_tensor = np.einsum("il,jl,kl -> ijk", *decomposition)
    assert np.array_equal(
        constructed_tensor, matmul_tensor
    ), f"Tensor constructed by decomposition does not match the matrix multiplication tensor <{(n,m,p)}>: {constructed_tensor}."
    logging.info(
        f"Verified a decomposition of rank {rank} for matrix multiplication tensor <{n},{m},{p}>."
    )

    # Print the set of values used in the decomposition.
    np.set_printoptions(linewidth=100)
    logging.info(
        f"This decomposition uses these factor entries:\n{np.array2string(np.unique(np.vstack((factor_matrix_1, factor_matrix_2, factor_matrix_3))), separator=', ')}"
    )


def validate_solution(solution, fitness: float, spec) -> None:
    """Validate and verify a tensor decomposition solution."""
    import logging

    import numpy as np

    # Extract solution and log its format/shape
    if hasattr(solution, "code"):
        solution_vector = solution.code
    else:
        solution_vector = solution

    # Enhanced debugging
    logging.info(f"Solution type: {type(solution_vector)}")
    if isinstance(solution_vector, list):
        logging.info(f"List length: {len(solution_vector)}")
        for i, item in enumerate(solution_vector):
            logging.info(
                f"Item {i} type: {type(item)}, shape: {np.shape(item) if hasattr(item, 'shape') else len(item)}"
            )

    # Skip verification for invalid solutions
    if fitness == float("-inf") or solution_vector is None:
        logging.warning(
            f"Skipping verification - no valid solution (fitness: {fitness})"
        )
        return

    # Attempt verification with proper format conversion
    try:
        # If solution is a list of matrices, convert to expected format
        if isinstance(solution_vector, list) and len(solution_vector) == 3:
            solution_vector = tuple(np.array(m) for m in solution_vector)

        verify_tensor_decomposition(solution_vector, spec.n, spec.m, spec.p, spec.rank)
        logging.info(f"Verified best solution – Frobenius error: {fitness:.6f}")
    except Exception as e:
        logging.error(f"Verification failed: {str(e)}")
        logging.error(f"Solution structure: {type(solution_vector)}")
