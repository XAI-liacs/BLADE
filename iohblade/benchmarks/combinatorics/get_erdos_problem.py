from iohblade.benchmarks.combinatorics import ErdosMinOverlap


def get_combinatorics_problems() -> list[ErdosMinOverlap]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Erdos Minimum Overlap Problem benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Erdos Min Overlap Problem

    """
    em1 = ErdosMinOverlap()

    return [em1]
