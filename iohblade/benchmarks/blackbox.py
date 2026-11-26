"""A restricted view of an IOH problem for generated code.

The continuous benchmarks used to hand the IOH problem object itself to the
generated algorithm. That object tells the algorithm much more than a black
box should: ``func.optimum`` holds the optimal point and value (one call to
``func(func.optimum.x)`` gives a perfect score), ``func.meta_data`` names
the function and its instance, and ``func.reset()`` or
``func.detach_logger()`` would get around the evaluation budget.

:class:`BlackBoxFunction` only offers what an optimiser needs: calling the
function, its bounds, its dimension and the number of evaluations so far.
Python cannot fully hide an object from code running in the same process
(it stays reachable by introspection), so this removes the documented ways
to read the answer, not every conceivable one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


class BlackBoxFunction:
    """Callable black-box view of an IOH problem.

    Attributes available to the algorithm:

    * ``func(x)``: evaluate ``x`` (counts towards the budget).
    * ``func.bounds.lb`` / ``func.bounds.ub``: lower and upper bounds (copies).
    * ``func.meta_data.n_variables``: the dimension;
      ``func.meta_data.optimization_type``: minimisation or maximisation.
    * ``func.state.evaluations`` and ``func.state.current_best`` (with
      ``.x`` and ``.y``): progress of the run so far.

    Args:
        problem: The IOH problem to wrap. Loggers attached to it keep working,
            since every call is passed on to it.
    """

    __slots__ = ("_evaluate", "_progress", "bounds", "meta_data")

    def __init__(self, problem: Any):
        # The problem is only kept in closures, not as an attribute.
        def evaluate(x):
            return problem(x)

        def progress():
            state = problem.state
            best = state.current_best
            return SimpleNamespace(
                evaluations=state.evaluations,
                current_best=SimpleNamespace(x=np.array(best.x), y=best.y),
            )

        self._evaluate = evaluate
        self._progress = progress
        self.bounds = SimpleNamespace(
            lb=np.array(problem.bounds.lb, dtype=float),
            ub=np.array(problem.bounds.ub, dtype=float),
        )
        self.meta_data = SimpleNamespace(
            n_variables=problem.meta_data.n_variables,
            optimization_type=problem.meta_data.optimization_type,
        )

    def __call__(self, x):
        return self._evaluate(x)

    @property
    def state(self) -> SimpleNamespace:
        """Evaluations used and best solution found so far."""
        return self._progress()

    def __repr__(self) -> str:
        return f"BlackBoxFunction(dim={self.meta_data.n_variables})"
