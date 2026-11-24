from iohblade.benchmarks.combinatorics import ErdosMinOverlap
from iohblade.benchmarks.suite_utils import with_eval_timeout

# Left half (48 step heights) of the AlphaEvolve construction, from section
# B.5 of the AlphaEvolve results notebook. The full step function is
# symmetric; the notebook builds it as concatenate((seq[:-1], seq[::-1])),
# i.e. 95 heights on [-1, 1] (the middle height appears once). See
# ``best_solution`` below; it scores 0.3809230 (best known 0.380924).
best_solution_half = [
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


# Full symmetric 95-value seed, as constructed in the notebook (B.5).
best_solution = best_solution_half[:-1] + best_solution_half[::-1]


def get_combinatorics_problems(
    use_best: bool = False, eval_timeout: int = 600
) -> list[ErdosMinOverlap]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Erdos Minimum Overlap Problem benchamarks, as an array.

    Args:
        `use_best : bool`: Provide LLM with the best known solution to start search at.
        eval_timeout: Seconds before an evaluation is stopped (default 600).

    Returns:
        An array of benchmark objects as follows:
            array[0] = Erdos Min Overlap Problem

    """
    if use_best:
        em1 = ErdosMinOverlap(best_solution=best_solution)
        return with_eval_timeout([em1], eval_timeout)
    em1 = ErdosMinOverlap()
    return with_eval_timeout([em1], eval_timeout)
