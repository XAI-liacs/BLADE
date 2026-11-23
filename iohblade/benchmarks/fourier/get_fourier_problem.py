from iohblade.benchmarks.fourier import UncertaintyInequality

# AlphaEvolve construction (results notebook, section B.4): coefficients of
# H_0, H_4 and H_8; the evaluator appends the H_12 coefficient that makes
# P(0) = 0. Scores 0.3520991 (best known 0.3521).
best_known_configuration = [
    0.3292519302257546,
    -0.01158510802599293,
    -8.921606035407065e-05,
]


def get_fourier_problems(use_best: bool) -> list[UncertaintyInequality]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Fourier Uncertianity Inequality benchamarks, as an array.

    Args:
        `use_best`: Try using best know solution as initial population in llm provided problems.

    Returns:
        An array of benchmark objects as follows:
            array[0] = Fourier Uncertainty Inequality benchmark object.

    """
    if use_best:
        ue1 = UncertaintyInequality(best_solution=best_known_configuration)
    else:
        ue1 = UncertaintyInequality()

    return [ue1]
