from .sums_vs_differences import SumDifference


def get_sum_vs_difference_problem() -> list[SumDifference]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Sums vs Difference benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Sums vs Difference benchmark instance.
    """

    sums_vs_difference = SumDifference()

    return [sums_vs_difference]
