"""The example solution in every built-in problem's prompt must work.

The example is shown to the LLM as a model answer, and `iohblade check`
evaluates it as the reference solution, so a broken example misleads both.
The examples are evaluated in-process here (no virtualenv), which keeps the
test fast; `iohblade check` covers the isolated evaluation.
"""

import math
import re

import pytest

from iohblade.registry import registry
from iohblade.solution import Solution
from iohblade.utils import first_class_name

CODE = re.compile(r"```(?:python)?\n(.*?)\n```", re.S)

# Problems that need network data, GPUs or optional dependency groups.
SKIPPED = {"automl", "kerneltuner", "photonics"}


def _problem_instances():
    cases = {
        "circle_packing": lambda: [registry.resolve("problem", "circle_packing")()],
        "tsp": lambda: [registry.resolve("problem", "tsp")("A-n53-k7")],
        "vrp": lambda: [registry.resolve("problem", "vrp")("A-n53-k7")],
        "graph_coloring": lambda: [registry.resolve("problem", "graph_coloring")("1")],
        "euclidian_steiner_tree": lambda: [
            registry.resolve("problem", "euclidian_steiner_tree")(20)
        ],
        "spherical_code": lambda: [registry.resolve("problem", "spherical_code")()],
        "toy_multiobjective": lambda: [
            registry.resolve("problem", "toy_multiobjective")()
        ],
        "rectangle_packing": lambda: [
            registry.resolve("problem", "rectangle_packing")()
        ],
        "bbob_sbox": lambda: [
            registry.resolve("problem", "bbob_sbox")(
                dims=[2], training_instances=[(1, 1)], test_instances=[(1, 2)]
            )
        ],
        "ma_bbob": lambda: [
            registry.resolve("problem", "ma_bbob")(
                dims=[2], training_instances=[1], test_instances=[2]
            )
        ],
    }
    for suite in registry.keys("suite"):
        if suite == "matrix_multiplication":
            cases[f"suite:{suite}"] = lambda s=suite: [
                registry.resolve("suite", s)()[i] for i in (0, 27, 53)
            ]
            continue
        for best in (False, True):
            cases[f"suite:{suite}:best={best}"] = (
                lambda s=suite, b=best: registry.resolve("suite", s)(b)
            )
    return cases


CASES = _problem_instances()


def test_every_builtin_problem_is_covered():
    covered = set()
    for key, make in CASES.items():
        if key.startswith("suite:"):
            continue
        covered.add(key)
    problems_from_suites = set()
    for suite in registry.keys("suite"):
        args = () if suite == "matrix_multiplication" else (False,)
        for problem in registry.resolve("suite", suite)(*args):
            problems_from_suites.add(type(problem).__name__)
    missing = []
    for key in registry.keys("problem"):
        if key in SKIPPED or key in covered:
            continue
        cls = registry.resolve("problem", key)
        if cls.__name__ not in problems_from_suites:
            missing.append(key)
    assert not missing, f"Add these problems to CASES: {missing}"


@pytest.mark.parametrize("key", sorted(CASES))
def test_example_in_prompt_evaluates(key):
    for problem in CASES[key]():
        match = CODE.search(problem.example_prompt or "")
        assert match, f"{problem.name}: no ```python block in example_prompt"
        code = match.group(1)
        name = first_class_name(code)
        assert name, f"{problem.name}: no class in the example"
        result = problem.evaluate(Solution(code=code, name=name))
        assert not result.error, f"{problem.name}: {result.error}"
        try:
            values = [float(result.fitness)]
        except TypeError:  # multi-objective
            values = [float(v) for v in result.fitness.values().values()]
        assert all(
            math.isfinite(v) for v in values
        ), f"{problem.name}: fitness {result.fitness} ({result.feedback})"
