"""Prompt helpers shared by the built-in methods."""

from __future__ import annotations

from ..problem import Problem


def task_context(problem: Problem, keep_format: bool = False) -> str:
    """The problem prompt without its example solution (and, unless
    ``keep_format``, without its answer format).

    Methods that show their own programs and ask for their own reply format
    use this instead of ``problem.task_prompt`` alone, because problems can
    add context in ``get_prompt()`` (for example the description and
    algorithmic insights of the photonics problems).
    """
    text = problem.get_prompt()
    parts = [getattr(problem, "example_prompt", "")]
    if not keep_format:
        parts.append(getattr(problem, "format_prompt", ""))
    for part in parts:
        if isinstance(part, str) and part.strip() and part in text:
            text = text.replace(part, "\n", 1)
    return text.strip()
