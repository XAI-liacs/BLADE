"""Helpers shared by the benchmark suite factories."""

from __future__ import annotations

import logging
from typing import TypeVar

T = TypeVar("T")


def with_eval_timeout(problems: list[T], eval_timeout: int | None) -> list[T]:
    """Set ``eval_timeout`` (seconds) on every problem, unless it is None."""
    if eval_timeout is not None:
        if (
            not isinstance(eval_timeout, (int, float))
            or isinstance(eval_timeout, bool)
            or eval_timeout <= 0
        ):
            raise ValueError(
                f"eval_timeout must be a positive number of seconds "
                f"(got {eval_timeout!r})."
            )
        for problem in problems:
            problem.eval_timeout = eval_timeout
    return problems


def log_instantiated(message: str) -> None:
    """Log that a benchmark problem was created (DEBUG level).

    The AlphaEvolve problems used to print a banner per problem, which
    flooded the terminal when building a suite; set the log level of the
    ``iohblade.benchmarks`` logger to DEBUG to see these messages.
    """
    logging.getLogger("iohblade.benchmarks").debug(message)
