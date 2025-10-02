from iohblade.benchmarks.fourier import UncertaintyInequality

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
    ue1 = UncertaintyInequality(best_solution=best_known_configuration)

    return [ue1]
