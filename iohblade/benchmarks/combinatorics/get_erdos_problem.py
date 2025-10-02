from iohblade.benchmarks.combinatorics import ErdosMinOverlap

best_solution = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.60911302e-10,
    3.62124044e-10,
    4.02849974e-12,
    4.47352578e-12,
    4.76914172e-12,
    0.506074303,
    0.632046692,
    0.679332798,
    0.888193865,
    0.889214704,
    0.678231235,
    2.976636922840846e-07,
    0.0947643739,
    0.0143926342,
    0.423931858,
    0.598073612,
    0.803909612,
    0.683098916,
    0.314749384,
    0.404059484,
    0.858443734,
    0.796503042,
    0.590433152,
    0.41056218,
    0.270932695,
    0.613384276,
    0.709501647,
    0.580573615,
    0.803538112,
    0.715263878,
    0.822611331,
    0.808433879,
    0.683533985,
    0.645719012,
    0.889417725,
    0.943389845,
    0.841536959,
    0.794505216,
    0.941943428,
    0.962223227,
    0.961270753,
    0.992409079,
]


def get_combinatorics_problems(use_best: bool) -> list[ErdosMinOverlap]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Erdos Minimum Overlap Problem benchamarks, as an array.

    Args:
        `use_best : bool`: Provide LLM with the best known solution to start search at.

    Returns:
        An array of benchmark objects as follows:
            array[0] = Erdos Min Overlap Problem

    """
    if use_best:
        em1 = ErdosMinOverlap(best_solution=best_solution)
        return [em1]
    em1 = ErdosMinOverlap()
    return [em1]
