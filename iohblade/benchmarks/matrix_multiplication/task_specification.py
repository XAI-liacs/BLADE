from __future__ import annotations

from typing import Optional

from iohblade.problem import Problem
from iohblade.solution import Solution


class MatMulTensorDecomposition(Problem):
    def __init__(self, n: int, m: int, p: int, grid: int, rank: int) -> None:
        super().__init__(name=f"mat_mult<{n},{m},{p}>")
        self.n: int = n
        self.m: int = m
        self.p: int = p
        self.grid: int = grid
        self.rank: int = rank

        self.task_prompt = f"""
Your goal is to find a rank-{self.rank} CP decomposition of the <{self.n},{self.m},{self.p}>
matrix‐multiplication tensor with zero error.

Concretely, you must produce factor matrices F1, F2, F3 of shape:
- F1: ({self.n*self.m} × {self.rank})
- F2: ({self.m*self.p} × {self.rank})
- F3: ({self.p*self.n} × {self.rank})

All entries must be multiples of {self.grid}.

The reconstruction is calculated as:
t_hat[i,j,k] = Σₗ F1[i*{self.m}+j, l] · F2[j*{self.p}+k, l] · F3[k*{self.n}+i, l]
Your goal is to minimise this score.
Return a flattened list of the factor matrices' entries.
"""

        self.example_prompt = f"""
An example code structure is as follows:

```python
import numpy as np

class TensorDecomposition:
    def __init__(self):
        self.n, self.m, self.p = {self.n}, {self.m}, {self.p}
        self.grid = {self.grid}

    def __call__(self):
        r = {self.rank}  # your computed rank
        F1 = np.zeros(({self.n*self.m}, r))
        F2 = np.zeros(({self.m*self.p}, r))
        F3 = np.zeros(({self.p*self.n}, r))

        # Fill F1, F2, F3 with your values

        return [F1.tolist(), F2.tolist(), F3.tolist()]
```
"""

        self.format_prompt = f"""
Give an excellent and novel rank-{self.rank} CP decomposition of the <{self.n},{self.m},{self.p}> matrix‐multiplication tensor with zero error.
Give the response in the format:

# Description: <short-description>
# Code:
```python
<code>
```
"""
        self.minimisation = True
        self.dependencies += []

        # ------------------------------------------------------------------ #
        #  ProblemSpec interface                                             #
        # ------------------------------------------------------------------ #

    def problem_description(self) -> str:
        """Return a short description for logging."""
        return f"<{self.n},{self.m},{self.p}> rank={self.rank} grid={self.grid}"

    def evaluate(self, individual, explogger=None):
        """
        Evaluate a single candidate by proxying to our existing get_evaluator().
        """
        from .get_evaluator import get_evaluator

        fn = get_evaluator(self)
        return fn(individual, self.dependencies, explogger)

    def test(self, individual: Solution) -> Solution:
        return self.evaluate(individual)

    def to_dict(self):
        return self.__dict__

    @property
    def target_fitness(self) -> Optional[float]:
        """
        For matrix multiplication, target fitness is 0.0 (perfect reconstruction).

        A perfect tensor decomposition has zero error in the Frobenius norm,
        meaning it exactly reconstructs the matrix multiplication tensor.
        """
        return 0.0
