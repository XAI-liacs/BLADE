import numpy as np
from numpy.typing import NDArray
from typing import Optional

from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace

"""
    Autocorrelation measures how similar a signal is to a shifted version of itself.
    It is commonly used to detect repeating patterns, periodicity, or structure in data.
    This class provides tools to compute and analyze autocorrelation for sequences or functions.

    The goal here is to make the most random signal; minimising the peak delta of a signal with
    it's shifted self.

    This is a base class, that takes in:
        * Task which could be one of the following.
            * autocorr_ineq_1: minimise ‖f∗f‖₁
            * autocorr_ineq_2: minimise -||f*f||_2^2
            * autocorr_ineq_3: minimise max(f*f)
        * n_bins: Level of discretisation, number of bins in [-1/4, 1/4].
        * require_nonnegative, depending on the problem, it can be true or false.
            * autocorr_ineq_1: TRUE
            * autocorr_ineq_2: TRUE
            * autocorr_ineq_3: FALSE
"""


class AutoCorrBaseSpec:
    """
        Autocorrelation measures how similar a signal is to a shifted version of itself.
        It is commonly used to detect repeating patterns, periodicity, or structure in data.
        This class provides tools to compute and analyze autocorrelation for sequences or functions.

        The goal here is to make the most random signal; minimising the peak delta of a signal with
        it's shifted self.

    Args:
        * `task_name: str` Could be one of the following.
                * autocorr_ineq_1: minimise ‖f∗f‖₁
                * autocorr_ineq_2: maximise ||f*f||_2^2
                * autocorr_ineq_3: minimise max(f*f)
        * `n_bins` int: Level of discretisation, number of bins in [-1/4, 1/4], **MUST** be positive.
        * `best_known` float: Best known evaluation of the benchmark.
        * `best_solution` list[float]: Best known configuration of the benchmark.
    """

    def __init__(
        self,
        task_name: str,
        n_bins: int,
        best_known: float,
        best_solution: list[float] | None,
    ):
        print(
            f"""
              --------------------------------------------------------
              Instantiated {task_name}, best known score is {best_known}.
              --------------------------------------------------------"""
        )
        self.best_solution = best_solution
        valid_task_names = ["auto_corr_ineq_1", "auto_corr_ineq_2", "auto_corr_ineq_3"]
        if task_name not in valid_task_names:
            error_msg = "Expected task_name to be one of the following: " + " | ".join(
                valid_task_names
            )
            raise ValueError(error_msg)
        if n_bins <= 0 or not isinstance(n_bins, int):
            raise ValueError("n_bins must be positive integer.")

        self.task_name = task_name
        self.n_bins = n_bins
        self.require_non_negative = True
        if self.task_name == "auto_corr_ineq_3":
            self.require_non_negative = False

        self.normalise_l2 = False
        self.L = 0.25
        self.dx = (2 * self.L) / self.n_bins  # Δx = 0.5 / N
        self.best_known = best_known

    def __repr__(self):
        return f"Auto-Correlation Base Class for {self.task_name}, with discretisation level of {self.n_bins}; require non-negative set to {self.require_non_negative}."

    def make_task_prompt(self, formula: str) -> str:
        positivity = (
            "all entries in the list f must be greater than or equal to 0"
            if self.require_non_negative
            else "entries in the list f may be positive or negative"
        )

        norm = (
            "do not normalise the f, scaling does not change the score"
            if not self.normalise_l2
            else "The runner will L2-normalize f; you do not need to"
        )

        return f"""

Write a python class with function `__call__`, that returns a list of floats f of length N.
- Where N is number of bins over [-1/4, 1/4] with discretization of dx = 0.5 / N.
- Auto-convolution of `g = dx * conv(f, f, mode="full")`, where g lies in range [-1/2, 1/2].
- Optimise for objective of {formula}, where {positivity} and {norm}.
- Symmetry or piecewise-constant structure is allowed if helpful.
- Set N = {self.n_bins} as default.

"""

    def make_example_prompt(self, class_name: str) -> str:
        best_known_initialiser = ""
        if self.best_solution is not None:
            best_known_initialiser = """
    def __init__(self, best_known_configuration: list[float] | None):
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        optimised for better results.
"""

        return f"""

An example template of such program is given by:
```python
class {class_name}:
    {best_known_initialiser}
    def __call__(self):
        return [0,0]*{self.n_bins}
```

"""

    def make_format_prompt(self):
        return """

Give an excellent and novel algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```

"""

    def _get_time_series(self, code) -> tuple[NDArray[np.float64], Optional[Exception]]:
        local_parameters = {}

        allowed = ["numpy", "scipy"]

        try:
            global_parameters = prepare_namespace(code, allowed)
            exec(code, global_parameters, local_parameters)
            local_parameters = clean_local_namespace(
                local_parameters, global_parameters
            )
            cls = next(v for v in local_parameters.values() if isinstance(v, type))
            try:
                f = np.asarray(
                    cls(best_known_configuration=self.best_known)(), dtype=np.float64
                )  # Runs if class has __init__(self, best_known_solution)
            except:
                f = np.asarray(
                    cls()(), dtype=np.float64
                )  # Rollback to empty initantiation.

            return f, None
        except Exception as e:
            print("\t Exception in `auto_correlation_ineq1.py`, " + e.__repr__())
            return (
                np.ndarray(
                    [
                        0,
                    ]
                ),
                e,
            )
