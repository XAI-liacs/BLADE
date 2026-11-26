"""Budget accounting of the search methods (GitHub issue #206).

A method gets ``budget`` = the number of solutions it may evaluate with
``problem(solution)``. These tests use a fake LLM (no API calls) and a fake
problem that counts its calls.
"""

import contextlib
import itertools
import math
import os
import random
import types
from typing import Any

import pytest

from iohblade.llm import LLM
from iohblade.methods import FunSearch, RandomSearch, ShinkaEvolve
from iohblade.methods.lhns.lhns import LHNS, LHNS_Method
from iohblade.methods.lhns.taboo_table import TabooElement
from iohblade.methods.mcts_ahd.mcts import MCTS, MCTS_Method
from iohblade.methods.moeh_method.moeh import MoEH, MoEH_Method
from iohblade.problem import Problem
from iohblade.solution import Solution

try:
    import eoh.methods.eoh.eoh_interface_EC as eoh_interface_ec
except Exception:  # pragma: no cover - optional dependency
    eoh_interface_ec = None


# region Fakes.
class FakeLLM(LLM):
    """LLM whose ``sample_solution`` returns a new, distinct code every call."""

    def __init__(self, fail_first: int = 0):
        super().__init__("", "fake", None)
        self.sample_calls = 0
        self.prompts: list[str] = []
        self.fail_first = fail_first
        self._counter = itertools.count(1)

    def _query(self, session_messages):
        # Used for MCTS descriptions and by EoH (which parses `{...}` + code).
        i = next(self._counter)
        return (
            f"{{Algorithm {i}}}\n"
            f"import numpy as np\n"
            f"class Algo{i}:\n"
            f"    def __call__(self, func):\n"
            f"        return {i}\n"
        )

    def sample_solution(self, session_messages, parent_ids=[], **kwargs):
        self.sample_calls += 1
        self.prompts.append(session_messages[-1]["content"])
        if self.sample_calls <= self.fail_first:
            raise RuntimeError("transient LLM failure")
        i = next(self._counter)
        code = (
            f"class Algo{i}:\n"
            f"    def __init__(self, budget, dim):\n"
            f"        self.budget = budget\n"
            f"        self.offset = {i}\n"
            f"\n"
            f"    def __call__(self, func):\n"
            f"        a = self.offset\n"
            f"        b = a * 2\n"
            f"        return a + b\n"
        )
        return Solution(
            code=code, name=f"Algo{i}", description="desc", parent_ids=parent_ids
        )

    def get_config(self) -> dict[str, Any]:
        return {}


class CountingProblem(Problem):
    """Counts the calls to ``problem(solution)``; every ``fail_every``-th
    evaluation fails with the worst fitness for the problem direction."""

    def __init__(self, minimisation: bool = False, fail_every: int = 0):
        super().__init__(name="counting")
        self.minimisation = minimisation
        self.fail_every = fail_every
        self.calls = 0

    def __call__(self, solution, logger=None):
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            solution.set_scores(self.failure_fitness(), "failed", "error")
        else:
            solution.set_scores(random.random(), "ok")
        return solution

    def evaluate(self, solution):
        return self(solution)

    def test(self, solution):
        return self(solution)

    def to_dict(self):
        return {}

    def get_config(self) -> dict[str, Any]:
        return {"minimisation": self.minimisation}


# endregion


# region LHNS.
@pytest.mark.parametrize("method", ["vns", "ils", "ts"])
@pytest.mark.parametrize("budget", [1, 2, 10, 23])
def test_lhns_evaluates_exactly_budget(method, budget):
    problem = CountingProblem()
    llm = FakeLLM()
    LHNS(problem, llm, method, budget=budget).run()
    assert problem.calls == budget
    # One LLM call for the initial solution and one per iteration.
    assert llm.sample_calls == budget


def test_lhns_initialise_samples_once():
    llm = FakeLLM()
    lhns = LHNS(CountingProblem(), llm, "vns")
    lhns.initialise()
    assert llm.sample_calls == 1
    assert lhns.current_solution.name == "Algo1"


def test_lhns_initialise_still_retries_on_exceptions():
    llm = FakeLLM(fail_first=3)
    lhns = LHNS(CountingProblem(), llm, "vns")
    lhns.initialise()
    assert llm.sample_calls == 4
    assert lhns.current_solution.code


def test_lhns_ts_sends_taboo_prompt(monkeypatch):
    llm = FakeLLM()
    lhns = LHNS(CountingProblem(), llm, "ts", budget=10)
    lhns.initialise()
    current = lhns.current_solution
    element = TabooElement("taboo desc", "taboo code", 0.5, ["feature"])
    monkeypatch.setattr(lhns.taboo_table, "get_distinct_entry", lambda s: element)
    monkeypatch.setattr(
        lhns.prompt_generator,
        "get_prompt_taboo_search",
        lambda individual, destroyed_code, taboo_element: "TABOO PROMPT",
    )

    new = lhns.mutate_lhns_ts(9)

    assert llm.prompts[-1] == "TABOO PROMPT"
    assert new is not current


def test_lhns_ts_run_sends_taboo_prompt_once_every_10(monkeypatch):
    problem = CountingProblem()
    llm = FakeLLM()
    lhns = LHNS(problem, llm, "ts", budget=10)
    element = TabooElement("taboo desc", "taboo code", 0.5, ["feature"])
    monkeypatch.setattr(lhns.taboo_table, "get_distinct_entry", lambda s: element)
    monkeypatch.setattr(
        lhns.prompt_generator,
        "get_prompt_taboo_search",
        lambda individual, destroyed_code, taboo_element: "TABOO PROMPT",
    )
    lhns.run()
    # Iterations 1..9: iteration 9 is the taboo iteration.
    assert llm.prompts.count("TABOO PROMPT") == 1
    assert problem.calls == 10


# endregion


# region MCTS-AHD.
@pytest.mark.parametrize("budget", [1, 5, 10, 23])
@pytest.mark.parametrize("fail_every", [0, 3])
def test_mcts_evaluates_exactly_budget(budget, fail_every):
    problem = CountingProblem(fail_every=fail_every)
    llm = FakeLLM()
    mcts = MCTS(llm, problem, budget, max_children=5)
    mcts.run()
    assert problem.calls == budget
    assert mcts.eval_remain == 0
    # No node is generated (paid LLM call) that cannot be evaluated.
    assert llm.sample_calls == budget


def test_mcts_simulate_does_not_evaluate_beyond_budget():
    problem = CountingProblem()
    mcts = MCTS(FakeLLM(), problem, 2)
    mcts.initialise(3)
    for child in mcts.root.children:
        mcts.simulate(child)
    assert problem.calls == 2
    assert mcts.eval_remain == 0


def test_mcts_expansion_respects_max_nodes():
    mcts = MCTS(FakeLLM(), CountingProblem(), 100, expansion_factor=2)
    mcts.initialise(1)
    child = mcts.root.children[0]
    mcts.expansion(child, max_nodes=3)
    assert [c.approach for c in child.children] == ["m1", "m2", "m1"]


# endregion


# region MoEH.
@pytest.mark.parametrize(
    "budget,population_size",
    [(1, 1), (2, 2), (7, 3), (9, 4), (10, 4), (12, 4), (5, 8)],
)
def test_moeh_evaluates_exactly_budget(budget, population_size):
    problem = CountingProblem()
    llm = FakeLLM()
    moeh = MoEH(llm, problem, 100, budget, population_size, False, max_workers=1)
    moeh.run()
    assert problem.calls == budget
    assert moeh._remaining_budget == 0


def test_moeh_stops_at_iterations_before_budget():
    problem = CountingProblem()
    moeh = MoEH(FakeLLM(), problem, 2, 100, 3, False, max_workers=1)
    moeh.run()
    # 3 initial + 2 generations of 3 offspring.
    assert problem.calls == 9


# endregion


# region EoH.
@pytest.mark.parametrize(
    "budget,pop_size,expected",
    [
        (1, 4, (1, 0)),
        (5, 4, (2, 0)),
        (8, 4, (4, 0)),
        (23, 4, (4, 0)),
        (24, 4, (4, 1)),
        (40, 4, (4, 2)),
        (100, 4, (4, 5)),
        (13, 2, (2, 1)),
    ],
)
def test_eoh_schedule_fits_budget(budget, pop_size, expected):
    from iohblade.methods.eoh import _eoh_schedule

    pop, n_gen = _eoh_schedule(budget, pop_size, 4)
    assert (pop, n_gen) == expected
    total = 2 * pop + n_gen * 4 * pop
    # Within budget (budget 1 relies on the adapter cap for the 2nd initial
    # evaluation), and less than one more generation left unused.
    assert total <= max(budget, 2)
    assert budget - total < 4 * pop or n_gen == 0


def test_eoh_schedule_rejects_empty_budget():
    from iohblade.methods.eoh import _eoh_schedule

    with pytest.raises(ValueError):
        _eoh_schedule(0, 4, 4)


@pytest.mark.parametrize("minimisation", [True, False])
def test_eoh_adapter_stops_at_budget(minimisation):
    from iohblade.methods.eoh import _BladeProblemAdapter

    problem = CountingProblem(minimisation=minimisation)
    adapter = _BladeProblemAdapter(problem, budget=2)
    assert math.isfinite(adapter.evaluate("class A: pass"))
    assert math.isfinite(adapter.evaluate("class B: pass"))
    assert adapter.evaluate("class C: pass") == math.inf
    assert problem.calls == 2


def _fast_eoh(monkeypatch, tmp_path):
    if eoh_interface_ec is None:  # pragma: no cover - optional dependency
        pytest.skip("eoh is not installed")
    # EoH sleeps 2s per operator call and writes ./EoHrun.jsonl.
    monkeypatch.setattr(
        eoh_interface_ec,
        "time",
        types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0),
    )
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    "budget,pop_size,expected_calls",
    [(1, 4, 1), (5, 4, 4), (13, 2, 12), (24, 4, 24)],
)
def test_eoh_run_stays_within_budget(
    monkeypatch, tmp_path, budget, pop_size, expected_calls
):
    from iohblade.methods.eoh import EoH

    _fast_eoh(monkeypatch, tmp_path)
    problem = CountingProblem()
    method = EoH(FakeLLM(), budget, pop_size=pop_size, output_path=str(tmp_path))
    with (
        pytest.warns(UserWarning) if pop_size * 2 > budget else contextlib.nullcontext()
    ):
        solution = method(problem)
    assert problem.calls == expected_calls
    assert solution.code


def _recording_tempdirs(monkeypatch):
    import tempfile

    used = []
    real = tempfile.TemporaryDirectory

    class Recording(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            used.append(self.name)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", Recording)
    return used


def test_eoh_runs_without_output_path_use_private_folders(monkeypatch, tmp_path):
    """Parallel runs used to share ./results and could collide (#238)."""
    from iohblade.methods.eoh import EoH

    _fast_eoh(monkeypatch, tmp_path)
    used = _recording_tempdirs(monkeypatch)
    for _ in range(2):
        assert EoH(FakeLLM(), 13, pop_size=2)(CountingProblem()).code
    assert len(set(used)) == 2  # one folder per run
    assert not any(os.path.exists(path) for path in used)  # removed
    assert not (tmp_path / "results").exists()  # not the working directory


def test_eoh_results_go_to_the_run_directory(monkeypatch, tmp_path):
    from iohblade.loggers import RunLogger
    from iohblade.methods.eoh import EoH

    _fast_eoh(monkeypatch, tmp_path)
    problem = CountingProblem()
    problem.set_logger(RunLogger(name="r", root_dir=str(tmp_path / "exp")))
    EoH(FakeLLM(), 13, pop_size=2)(problem)
    best = os.path.join(problem.logger_dir, "eoh", "results", "pops_best")
    assert os.listdir(best)
    assert not (tmp_path / "results").exists()


def test_eoh_shared_output_path_folders_exist_up_front(monkeypatch, tmp_path):
    from iohblade.methods.eoh import EoH

    _fast_eoh(monkeypatch, tmp_path)
    out = tmp_path / "shared"
    method = EoH(FakeLLM(), 13, pop_size=2, output_path=str(out))
    with contextlib.ExitStack() as stack:
        assert method._output_dir(CountingProblem(), stack) == str(out)
    for sub in ("history", "pops", "pops_best"):
        assert (out / "results" / sub).is_dir()
    # A second run on the same folder must not fail on existing folders.
    assert method(CountingProblem()).code


_TSP_EXAMPLE = """
An example:
```python
import random

class Tour:
    def __init__(self, customers):
        self.customers = customers

    def __call__(self):
        return [c[0] for c in self.customers]
```
"""

_BBOB_EXAMPLE = """
An example:
```python
class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget

    def __call__(self, func):
        return self.f_opt, self.x_opt
```
"""


class RecordingProblem(CountingProblem):
    """Records the code of every evaluated solution."""

    def __init__(self, example_prompt: str = _TSP_EXAMPLE):
        super().__init__()
        self.task_prompt = "Write a class that returns a tour. "
        self.example_prompt = example_prompt
        self.format_prompt = "Give the code in a python block."
        self.codes: list[str] = []

    def __call__(self, solution, logger=None):
        self.codes.append(solution.code)
        return super().__call__(solution, logger)


class TourLLM(LLM):
    """Answers like a chat model: a description, then a fenced class that
    has no imports, uses a dict literal and returns ``paths``."""

    def __init__(self, prefix: str = "Tour"):
        super().__init__("", "fake", None)
        self.prefix = prefix
        self.prompts: list[str] = []
        self._counter = itertools.count(1)

    def _query(self, session_messages):
        self.prompts.append(session_messages[-1]["content"])
        i = next(self._counter)
        return (
            f"{{Visit the customers in order {i}.}}\n"
            "```python\n"
            f"class {self.prefix}{i}:\n"
            "    def __init__(self, customers):\n"
            "        self.customers = customers\n"
            f"        self.weights = {{'a': {i}}}\n"
            "\n"
            "    def __call__(self):\n"
            "        paths = [c[0] for c in self.customers]\n"
            "        return paths\n"
            "```\n"
            "The class returns the tour.\n"
        )

    def get_config(self) -> dict[str, Any]:
        return {}


def test_eoh_extraction_keeps_the_code_as_written():
    from iohblade.methods.eoh import _extract_eoh_response

    code, description = _extract_eoh_response(TourLLM()._query([{"content": ""}]))
    assert code.startswith("class Tour1:")  # no import: class line kept
    assert code.rstrip().endswith("return paths")  # return not rewritten
    assert "f_opt" not in code and "```" not in code
    assert description == "Visit the customers in order 1."


def test_eoh_extraction_without_fences():
    from iohblade.methods.eoh import _extract_eoh_response

    response = (
        "# Description: Greedy tour.\n"
        "class Greedy:\n"
        "    def __call__(self):\n"
        "        return []\n"
    )
    code, description = _extract_eoh_response(response)
    assert code.startswith("class Greedy:") and code.endswith("return []")
    assert description == "Greedy tour."
    assert _extract_eoh_response("no code at all")[0] == ""


def test_eoh_prompt_states_the_example_interface():
    from iohblade.methods.eoh import _BladeEvolution, _BladePrompts

    prompts = _BladePrompts(RecordingProblem(_TSP_EXAMPLE))
    assert not prompts.declared_interface
    evol = _BladeEvolution("", "", "fake", False, "", False, prompts)
    for prompt in (
        evol.get_prompt_i1(),
        evol.get_prompt_m1({"algorithm": "a", "code": "c"}),
        evol.get_prompt_e2([{"algorithm": "a", "code": "c"}]),
        evol.get_prompt_m3({"algorithm": "a", "code": "c"}),
    ):
        assert "f_opt" not in prompt and "budget, dim" not in prompt
        assert "`__init__(self, customers)`" in prompt
        assert "`__call__(self)`" in prompt


def test_eoh_prompt_keeps_a_matching_declared_interface():
    from iohblade.methods.eoh import _BladeEvolution, _BladePrompts

    prompts = _BladePrompts(RecordingProblem(_BBOB_EXAMPLE))
    assert prompts.declared_interface
    evol = _BladeEvolution("", "", "fake", False, "", False, prompts)
    assert "return 2 output(s): 'f_opt', 'x_opt'" in evol.get_prompt_i1()


def test_declared_interface_is_checked_against_the_example():
    from iohblade.methods.prompt_utils import (
        declared_interface_matches,
        example_interface,
    )

    tsp = RecordingProblem(_TSP_EXAMPLE)
    assert not declared_interface_matches(tsp)
    assert example_interface(tsp) == {
        "class": "Tour",
        "init_inputs": ["customers"],
        "func_name": "__call__",
        "func_inputs": [],
    }
    assert declared_interface_matches(RecordingProblem(_BBOB_EXAMPLE))
    # No example class: nothing to check against.
    assert declared_interface_matches(RecordingProblem("no code here"))


def test_reevo_prompt_states_the_example_interface():
    from iohblade.methods.reevo import _func_desc, _func_signature

    tsp = RecordingProblem(_TSP_EXAMPLE)
    desc = _func_desc(tsp, "Tour")
    assert "budget, dim" not in desc and "func" not in desc
    assert "`__init__(self, customers)`" in desc
    assert _func_signature(tsp) == "__call__(self)"

    bbob = RecordingProblem(_BBOB_EXAMPLE)
    assert "`__init__(self, budget, dim)`" in _func_desc(bbob, "RandomSearch")
    assert _func_signature(bbob) == "__call__(self, func)"


def test_eoh_evaluates_the_code_the_llm_wrote(monkeypatch, tmp_path):
    from iohblade.methods.eoh import EoH

    _fast_eoh(monkeypatch, tmp_path)
    problem = RecordingProblem()
    llm = TourLLM()
    solution = EoH(llm, 12, pop_size=2, output_path=str(tmp_path / "out"))(problem)
    assert len(problem.codes) == 12
    for code in problem.codes:
        assert code.rstrip().endswith("return paths")
        assert "f_opt" not in code
    assert solution.code in problem.codes
    assert all("f_opt" not in p for p in llm.prompts)
    # EoH's run log goes to the run folder, not the working directory.
    assert (tmp_path / "out" / "EoHrun.jsonl").exists()
    assert not (tmp_path / "EoHrun.jsonl").exists()


def test_parallel_eoh_runs_use_their_own_llm(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from iohblade.methods.eoh import EoH

    _fast_eoh(monkeypatch, tmp_path)

    def run(prefix):
        problem = RecordingProblem()
        out = tmp_path / prefix
        EoH(TourLLM(prefix), 12, pop_size=2, output_path=str(out))(problem)
        return prefix, problem.codes

    with ThreadPoolExecutor(2) as ex:
        results = list(ex.map(run, ["Alpha", "Beta"]))
    for prefix, codes in results:
        assert len(codes) == 12
        assert all(code.startswith(f"class {prefix}") for code in codes)


# endregion


# region All methods.
def _make_method(name: str, llm: LLM, budget: int):
    match name:
        case "RandomSearch":
            return RandomSearch(llm, budget)
        case "FunSearch":
            return FunSearch(llm, budget, num_islands=3, samples_per_prompt=3)
        case "ShinkaEvolve":
            return ShinkaEvolve(llm, budget, novelty_rejection=True, meta_interval=5)
        case "LHNS-vns":
            return LHNS_Method(llm, budget, "vns")
        case "LHNS-ils":
            return LHNS_Method(llm, budget, "ils")
        case "LHNS-ts":
            return LHNS_Method(llm, budget, "ts")
        case "MCTS_AHD":
            return MCTS_Method(llm, budget)
        case "MoEH":
            return MoEH_Method(llm, budget, 1000, 3, minimisation=False)
        case "EoH":
            from iohblade.methods.eoh import EoH

            return EoH(llm, budget, pop_size=2)
    raise ValueError(name)


@pytest.mark.parametrize(
    "name",
    [
        "RandomSearch",
        "FunSearch",
        "ShinkaEvolve",
        "LHNS-vns",
        "LHNS-ils",
        "LHNS-ts",
        "MCTS_AHD",
        "MoEH",
        "EoH",
    ],
)
@pytest.mark.parametrize("budget", [4, 12, 21])
def test_method_evaluations_match_budget(monkeypatch, tmp_path, name, budget):
    if name == "EoH":
        _fast_eoh(monkeypatch, tmp_path)
    problem = CountingProblem()
    method = _make_method(name, FakeLLM(), budget)
    if name == "EoH":
        method.kwargs["output_path"] = str(tmp_path)
    method(problem)
    if name == "EoH":
        # EoH runs whole generations (2 * pop_size initial evaluations, then
        # 4 * pop_size per generation): the remainder of the budget smaller
        # than a generation is not used.
        assert problem.calls == 4 + (budget - 4) // 8 * 8
        assert problem.calls <= budget
    else:
        assert problem.calls == budget


# endregion
