import builtins
import importlib
import inspect
import json
import math
import random
import textwrap
from typing import Any

import numpy as np

from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import clean_local_namespace, prepare_namespace
from iohblade.problem import Problem
from iohblade.solution import Solution


class ErdosMinOverlap(Problem):
    """
    The continuous Erdős minimum-overlap problem seeks to find measurable functions f,g: [-1,1] → [0,1] that satisfy:
        * Complementarity: f(x) + g(x) = 1 for all x ∈ [-1,1]
        * Unit mass: ∫_{-1}^1 f(x)dx = ∫_{-1}^1 g(x)dx = 1
        * Bounds: f(x), g(x) ∈ [0,1] for all x ∈ [-1,1]
    Optimisation;
        * Minimise superposition of x in [-2, 2], over f(t)g(x+t).
        * Alpha evolve: C_5 ≤ 0.380924 (previous best: 0.380927).

    A candidate returns the heights of a step function f on N equal bins of
    [-1, 1] (dx = 2/N). Any length 1 <= N <= ``max_bins`` is accepted;
    ``n_bins`` is only the resolution suggested in the prompt and used by
    the example.
    """

    def __init__(
        self,
        task_name="erdos_min_overlap",
        n_bins: int = 800,
        tolerance=1e-6,
        best_known=0.380924,
        best_solution: list[float] | None = None,
        max_bins: int = 100_000,
        eval_timeout: int = 600,
    ):
        super().__init__(name=task_name, eval_timeout=eval_timeout)
        self.task_name = task_name
        self.n_bins = n_bins
        # Upper bound on len(f). The overlap is computed with an FFT, so the
        # evaluation itself stays fast, but very long lists are slow to
        # build, transfer from the evaluation subprocess and log, and much
        # finer steps add nothing at the precision of the known bounds.
        self.max_bins = int(max_bins)
        self.tolerance = tolerance
        self.best_known = best_known
        self.best_solution = best_solution
        log_instantiated(
            f"Instantiated Erdös Min Overlap Problem, best known {self.best_known}."
        )

        self.minimisation = True
        self.dependencies += ["scipy"]

        self.task_prompt = textwrap.dedent("""
* Write a Python class whose `__call__` returns a list f of length N with values in range [0,1], such that:
    * f is a step function on [-1, 1] with `N` equal bins, dx = 2/N; you choose N (any length N >= 1 is allowed).
    * g is a point wise complement hence g = 1 - f, giving us f + g = 1. Both lie in range [0,1].
    * It follows integral constraints: ∫_{-1}^{1} f = 1 (hence ∫ g = 1).
        * The runner checks dx*sum(f)≈1.
    * Overlap functional uses zero-extension of g outside [-1,1].
    * Optimize the objective:
        * minimize  sup_{x ∈ [-2,2]} ∫_{-1}^{1} f(t) · g(x+t) dt,  with g = 1 - f
    * Do not use scipy's interp1d, it is no depricated.
""")

        self.task_prompt += f"""
    * N is free: any length 1 <= N <= {self.max_bins} is accepted, and dx = 2/N follows from it. N = {self.n_bins} is a reasonable starting resolution.
    * The tolerance of f + g similar to 1 is set to {self.tolerance}
        """

        initialiser = ""
        if best_solution is not None:
            initialiser = """
    def __init__(self, best_known_configuration=None):
        # Accepts a best known configuration (if available) for the problem,
        # as an initial configuration, which can then be optimised further.
        self.best_known_configuration = best_known_configuration
"""
        self.example_prompt = f"""

An example template of such program is given by:
```python
class ErdosCandidate:{initialiser}
    def __call__(self):
        # Simple baseline: f = 1/2 everywhere, so g = 1 - f = 1/2 and
        # dx * sum(f) = (2 / N) * (N / 2) = 1.
        N = {self.n_bins}
        return [0.5] * N
```

"""

        self.format_prompt = textwrap.dedent("""

Give an excellent and novel algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```

""")
        if self.n_bins <= 0:
            raise ValueError("Expected n-bins to be positive number.")
        if self.max_bins < 1:
            raise ValueError("Expected max_bins to be a positive number.")

        # dx of the suggested resolution; the evaluator uses 2 / len(f).
        self.dx = 2 / self.n_bins

    def problem_description(self) -> str:
        return (
            f"{self.task_name} | any N <= {self.max_bins} bins on [-1,1], "
            f"dx=2/N (suggested N={self.n_bins})"
        )

    def _sup_overlap(self, f: np.ndarray, g: np.ndarray, dx: float) -> float:
        """Return max over shifts s of dx * sum_i f[i] * g[i + s].

        g is zero-extended outside [-1, 1]. The step functions' overlap is
        piecewise linear in the shift, so its maximum is attained at a knot.
        Small inputs use an exact direct correlation, large inputs an FFT.
        """
        N = f.size
        if N <= 4096:
            corr = np.correlate(g, f, mode="full")
        else:
            L = 1 << int(np.ceil(np.log2(2 * N - 1)))
            corr = np.fft.irfft(np.fft.rfft(g, L) * np.conj(np.fft.rfft(f, L)), L)
            # Lags -(N-1)..-1 wrap to the end of the circular correlation.
            corr = np.concatenate((corr[L - (N - 1) :], corr[:N]))
        return max(0.0, dx * float(np.max(corr)))

    def evaluate(self, solution: Solution, explogger=None):
        local_ns = {}
        code = solution.code
        name = solution.name if solution.name else "ErdosCandidate"
        try:
            safe_globals = prepare_namespace(code, self.dependencies)
            compiled = compile(code, filename=name, mode="exec")
            exec(compiled, safe_globals, local_ns)

            cls = local_ns[name]

            if self.best_solution is not None:
                f = np.asarray(
                    cls(best_known_configuration=self.best_solution)(), dtype=np.float64
                )
            else:
                f = np.asarray(cls()(), dtype=np.float64)
        except Exception as e:
            solution = solution.set_scores(self.failure_fitness(), f"exec-error {e}", e)
            return solution

        try:
            if f.ndim != 1 or f.size < 1:
                raise ValueError(f"f must be a non-empty 1D list, got {f.shape}.")
            if f.size > self.max_bins:
                raise ValueError(
                    f"f has {f.size} entries; at most {self.max_bins} are allowed."
                )
            if not np.all(np.isfinite(f)):
                raise ValueError("entries of f must be finite (no NaN or inf)")
            # bounds f ∈ [0,1] within tolerance
            if np.any(f < -self.tolerance) or np.any(f > 1.0 + self.tolerance):
                raise ValueError("entries of f must lie in [0,1]")

            dx = 2.0 / f.size
            I_f = dx * float(np.sum(f))
            if abs(I_f - 1.0) > self.tolerance:
                raise ValueError(f"integral ∫f must be 1 (got {I_f:.6g})")

            g = 1.0 - f  # f+g=1 pointwise

            score = self._sup_overlap(f, g, dx)  # minimize
            if not np.isfinite(score):
                raise ValueError(f"computed overlap is not finite ({score})")
            msg = f"Score = {score:.6g}; with configuration: N={f.size}, dx={dx:.6g}, If={I_f:.6g}.\n\t Best known score = {self.best_known}"
            solution = solution.set_scores(score, msg)
        except Exception as e:
            solution = solution.set_scores(self.failure_fitness(), f"calc-error {e}", e)

        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__.copy()

    def get_config(self) -> builtins.dict[builtins.str, Any]:
        from iohblade.tags import (
            Benchmark,
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            StructureTag,
            VariableType,
        )

        tags: list[Any] = [PrimaryCategories.CO]
        tags.append(Benchmark.ERDOS_MIN_OVERLAP)
        tags.append(NoiseType.NOISELESS)
        tags.append(ObjectiveType.SINGLE_OBJECTIVE)
        tags.append(VariableType.DISCRETE)
        tags.append(StructureTag.GEOMETRIC)
        config = {
            "tags": tags[:],
            "name": "Erdös Minimum Overlap",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": inspect.getsource(self.evaluate),
            "config": {
                "n_bins": self.n_bins,
                "max_bins": self.max_bins,
                "tolerance": self.tolerance,
                "dependencies": self.dependencies,
            },
        }
        return config


if __name__ == "__main__":
    eop = ErdosMinOverlap()
    for key, value in eop.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
    print(json.dumps(eop.get_config()))
