"""Prompt helpers shared by the built-in methods."""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from ..llm import _extract_python_block
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


def _signature_args(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameter names of a method, without ``self``."""
    args = func.args
    names = [a.arg for a in args.posonlyargs + args.args]
    return names[1:] if names and names[0] == "self" else names


def example_interface(problem: Problem) -> dict[str, Any] | None:
    """The interface of the class in the problem's example code.

    Returns:
        ``{"class", "init_inputs", "func_name", "func_inputs"}`` of the first
        class in the code block of ``problem.example_prompt``, or None when
        there is no parsable example class. ``func_name`` is the problem's
        ``func_name`` if the class defines it, else ``__call__`` if defined,
        else None.
    """
    code = _extract_python_block(getattr(problem, "example_prompt", "") or "")
    if not code:
        return None
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if cls is None:
        return None
    methods = {
        n.name: n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    func_name = getattr(problem, "func_name", "__call__")
    if func_name not in methods:
        func_name = "__call__" if "__call__" in methods else None
    return {
        "class": cls.name,
        "init_inputs": (
            _signature_args(methods["__init__"]) if "__init__" in methods else []
        ),
        "func_name": func_name,
        "func_inputs": _signature_args(methods[func_name]) if func_name else [],
    }


def declared_interface_matches(problem: Problem) -> bool:
    """Whether the interface a problem declares for EoH and ReEvo
    (``init_inputs``, ``func_name``, ``func_inputs``) is that of the class in
    its example code (True when there is no example class).

    ``Problem`` defaults these to the BBOB interface (``__init__(self,
    budget, dim)``, ``__call__(self, func)`` returning ``f_opt, x_opt``),
    which contradicts the task of most other problems; methods that state
    the interface in their prompts use the example's interface when this is
    False.
    """
    example = example_interface(problem)
    if example is None:
        return True
    declared = (
        list(getattr(problem, "init_inputs", []) or []),
        getattr(problem, "func_name", "__call__"),
        list(getattr(problem, "func_inputs", []) or []),
    )
    return declared == (
        example["init_inputs"],
        example["func_name"],
        example["func_inputs"],
    )
