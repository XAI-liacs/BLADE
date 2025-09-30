from iohblade.benchmarks.fourier import UncertaintyInequality


def get_fourier_problems() -> list[UncertaintyInequality]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Fourier Uncertianity Inequality benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Fourier Uncertainty Inequality benchmark object.

    """
    ue1 = UncertaintyInequality()

    return [ue1]
