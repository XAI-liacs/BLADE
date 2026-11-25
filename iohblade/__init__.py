import multiprocessing
import os

# MLflow (used by the MLflow logger and imported by the registry) logs an
# "agent hint" line on import, which would show up in every `iohblade`
# command. Silence it unless the user set the variable themselves.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Only import the minimal set of objects that do not pull in heavy
# dependencies.  Modules such as ``llm`` or ``plots`` require optional
# third-party packages and should be imported explicitly by consumers
# that need them.
from .fitness import Fitness
from .problem import Problem, wrap_problem
from .solution import Solution
from .utils import (
    NoCodeException,
    OverBudgetException,
    ThresholdReachedException,
    TimeoutException,
    aoc_logger,
    budget_logger,
    convert_to_serializable,
    correct_aoc,
)

__all__ = [
    "Fitness",
    "Problem",
    "wrap_problem",
    "Solution",
    "NoCodeException",
    "OverBudgetException",
    "ThresholdReachedException",
    "TimeoutException",
    "aoc_logger",
    "budget_logger",
    "convert_to_serializable",
    "correct_aoc",
]


def ensure_spawn_start_method():
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            raise RuntimeError(
                "Multiprocessing start method is not 'spawn'. "
                "Set it at the top of your main script:\n"
                "import multiprocessing\n"
                "multiprocessing.set_start_method('spawn', force=True)"
            )


ensure_spawn_start_method()
