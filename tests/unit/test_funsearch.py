"""Tests for the FunSearch method (#216)."""

import math
from typing import Any

import numpy as np
import pytest

from iohblade import cli
from iohblade.config import load_plan
from iohblade.config.errors import ConfigError
from iohblade.config.templates import write_template
from iohblade.fitness import Fitness
from iohblade.llm import LLM
from iohblade.methods import FunSearch, funsearch
from iohblade.methods.funsearch import ProgramsDatabase
from iohblade.problem import Problem
from iohblade.registry import BuiltinMethod, ComponentKind, registry
from iohblade.solution import Solution


class _LLM(LLM):
    """Returns a new class each call; records the prompts."""

    def __init__(self):
        super().__init__("", "fake")
        self.prompts: list[str] = []
        self.parents: list[list[str]] = []

    def sample_solution(self, session_messages, parent_ids=[], **kwargs):
        self.prompts.append(session_messages[-1]["content"])
        self.parents.append(list(parent_ids))
        i = len(self.prompts)
        return Solution(
            code=f"class A{i}:\n    x = {i}\n", name=f"A{i}", parent_ids=parent_ids
        )

    def _query(self, session_messages, **kwargs):
        return ""

    def get_config(self) -> dict[str, Any]:
        return {}


class _Scripted(Problem):
    """Scores the i-th evaluation with ``scores[i]`` (None = failure)."""

    def __init__(self, scores, minimisation=False):
        super().__init__(name="scripted")
        self.scores = list(scores)
        self.minimisation = minimisation
        self.task_prompt = "TASK"
        self.format_prompt = "FORMAT"
        self.calls = 0

    def __call__(self, solution, logger=None):
        score = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        if score is None:
            return solution.set_scores(self.failure_fitness(), "failed", "error")
        return solution.set_scores(score, f"score {score}")

    def evaluate(self, solution):
        return self(solution)

    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


# region Budget and direction.
@pytest.mark.parametrize("budget", [0, 1, 2, 5, 13, 40])
@pytest.mark.parametrize("samples_per_prompt", [1, 4])
def test_evaluates_exactly_budget(budget, samples_per_prompt):
    np.random.seed(0)
    problem = _Scripted([1.0, None, 2.0, 0.5])
    llm = _LLM()
    best = FunSearch(llm, budget, samples_per_prompt=samples_per_prompt)(problem)
    assert problem.calls == budget == len(llm.prompts)
    assert (best is None) == (budget == 0)


@pytest.mark.parametrize("minimisation, expected", [(True, 1.0), (False, 9.0)])
def test_best_follows_the_direction_and_skips_failures(minimisation, expected):
    np.random.seed(1)
    problem = _Scripted([5.0, None, 9.0, 1.0, None, 7.0], minimisation)
    method = FunSearch(_LLM(), 30, num_islands=2)
    best = method(problem)
    assert best.fitness == expected
    # Failed evaluations never enter the database.
    for island in method.database.islands:
        assert all(math.isfinite(score) for score in island.clusters)
    # FunSearch maximises its score: lower fitness is better when minimising.
    sign = -1 if minimisation else 1
    assert max(method.database.best_score) == sign * expected


def test_failures_before_the_first_success_are_retried():
    np.random.seed(0)
    problem = _Scripted([None, None, 3.0, 4.0])
    llm = _LLM()
    method = FunSearch(llm, 6)
    best = method(problem)
    # The first three samples come from the problem prompt.
    assert llm.prompts[:3] == [problem.get_prompt()] * 3
    assert llm.prompts[3] != problem.get_prompt()
    assert best.fitness == 4.0
    # The first successful program is on every island.
    assert all(p is not None for p in method.database.best_program)


def test_all_failures_return_a_failed_solution():
    problem = _Scripted([None])
    best = FunSearch(_LLM(), 3)(problem)
    assert problem.calls == 3 and best.fitness == -math.inf
    assert best.name == "A1"


def test_multi_objective_fitness_is_rejected():
    class _Multi(_Scripted):
        def __call__(self, solution, logger=None):
            self.calls += 1
            return solution.set_scores(Fitness({"a": 1.0, "b": 2.0}))

    with pytest.raises(ValueError, match="single-objective"):
        FunSearch(_LLM(), 3)(_Multi([1.0]))


# endregion


# region Prompts and solutions.
def test_prompt_shows_versions_worst_first():
    np.random.seed(0)
    problem = _Scripted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    llm = _LLM()
    method = FunSearch(llm, 6, num_islands=1, samples_per_prompt=1)
    method(problem)
    prompt = llm.prompts[-1]
    assert prompt.startswith("TASK") and prompt.rstrip().endswith("FORMAT")
    assert "# Version 0" in prompt and "# Version 1 (improved version" in prompt
    assert "Write version 2: an improved version of version 1." in prompt
    # Version 0 has the lower score of the two programs in the prompt.
    parents = llm.parents[-1]
    assert len(parents) == 2
    database = method.database.islands[0]
    score = {
        p.id: s for s, cluster in database.clusters.items() for p in cluster.programs
    }
    assert score[parents[0]] <= score[parents[1]]


def test_solutions_record_generation_island_and_parents():
    np.random.seed(0)
    problem = _Scripted([1.0, 2.0])
    llm = _LLM()
    method = FunSearch(llm, 5, num_islands=3, samples_per_prompt=2)
    method(problem)
    programs = {
        p.id: p
        for island in method.database.islands
        for cluster in island.clusters.values()
        for p in cluster.programs
    }
    initial = [p for p in programs.values() if p.generation == 0]
    assert len(initial) == 1 and initial[0].parent_ids == []
    for program in programs.values():
        if program.generation > 0:
            assert program.metadata["island"] in range(3)
            assert program.parent_ids and set(program.parent_ids) <= set(programs)


# endregion


# region Programs database.
def _solution(code: str) -> Solution:
    return Solution(code=code, name="A")


def test_database_clusters_by_score_and_registers_initial_everywhere():
    db = ProgramsDatabase(num_islands=3)
    assert db.empty
    first = _solution("a")
    db.register(first, None, 1.0)
    assert not db.empty
    assert all(island.num_programs == 1 for island in db.islands)
    db.register(_solution("bb"), 0, 1.0)
    db.register(_solution("c"), 0, 2.0)
    assert sorted(db.islands[0].clusters) == [1.0, 2.0]
    assert len(db.islands[0].clusters[1.0].programs) == 2
    assert db.best_score == [2.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        db.register(_solution("d"), 0, -math.inf)


def test_database_reset_restarts_weaker_half_from_strong_islands():
    np.random.seed(0)
    db = ProgramsDatabase(num_islands=4)
    db.register(_solution("init"), None, 0.0)
    strong = {2: _solution("two"), 3: _solution("three")}
    db.register(strong[2], 2, 5.0)
    db.register(strong[3], 3, 9.0)
    db.register(_solution("weak"), 0, 1.0)
    reset = db.reset_islands()
    assert sorted(reset) == [0, 1]
    for island_id in reset:
        island = db.islands[island_id]
        assert island.num_programs == 1
        assert db.best_program[island_id] in strong.values()
        assert db.best_score[island_id] in (5.0, 9.0)
    # The stronger islands are kept as they were.
    assert db.best_program[2] is strong[2] and db.best_program[3] is strong[3]
    assert db.islands[3].num_programs == 2


def test_run_resets_islands_every_reset_period(monkeypatch):
    np.random.seed(0)
    calls = []
    original = ProgramsDatabase.reset_islands

    def spy(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(ProgramsDatabase, "reset_islands", spy)
    FunSearch(_LLM(), 20, samples_per_prompt=1, reset_period=5)(_Scripted([1.0, 2.0]))
    assert len(calls) == 3  # after 5, 10 and 15 evaluations, not at the end
    calls.clear()
    FunSearch(_LLM(), 20, reset_period=0)(_Scripted([1.0]))
    assert calls == []
    calls.clear()
    FunSearch(_LLM(), 20, samples_per_prompt=1)(_Scripted([1.0]))  # budget // 5
    assert len(calls) == 4  # after 4, 8, 12 and 16 evaluations
    calls.clear()
    # A prompt's samples can pass a reset point: one reset, then the next.
    FunSearch(_LLM(), 20, samples_per_prompt=3, reset_period=4)(_Scripted([1.0]))
    assert len(calls) == 4  # after 4, 10, 13 and 16 evaluations


def test_sampling_prefers_better_clusters_and_shorter_programs():
    np.random.seed(0)
    db = ProgramsDatabase(num_islands=1, programs_per_prompt=1)
    db.register(_solution("bad"), 0, 0.0)
    good_short, good_long = _solution("g"), _solution("g" * 500)
    db.register(good_short, 0, 1.0)
    db.register(good_long, 0, 1.0)
    picks = [db.get_prompt().programs[0] for _ in range(300)]
    assert sum(p.code != "bad" for p in picks) > 290
    assert picks.count(good_short) > picks.count(good_long)


def test_temperature_decays_linearly_per_period():
    db = ProgramsDatabase(num_islands=1, temperature_init=1.0, temperature_period=4)
    island = db.islands[0]
    temperatures = []
    for i in range(5):
        temperatures.append(island.temperature())
        island.register(_solution(str(i)), float(i))
    assert temperatures == [1.0, 0.75, 0.5, 0.25, 1.0]


def test_get_prompt_on_empty_database_raises():
    with pytest.raises(ValueError, match="empty"):
        ProgramsDatabase(num_islands=2).get_prompt()


# endregion


# region Configuration.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_islands": 0},
        {"programs_per_prompt": 0},
        {"samples_per_prompt": 1.5},
        {"reset_period": -1},
        {"temperature_init": 0},
        {"temperature_period": 0},
        {"budget": -1},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    args = {"budget": 10, **kwargs}
    with pytest.raises(ValueError):
        FunSearch(_LLM(), **args)


def test_config_and_old_name():
    method = FunSearch(_LLM(), 10, num_islands=4, reset_period=3)
    assert funsearch is FunSearch
    config = method.get_config()
    assert config["name"] == "FunSearch"
    assert config["config"]["num_islands"] == 4
    assert method.to_dict()["reset_period"] == 3
    entry = registry.get(ComponentKind.METHOD, BuiltinMethod.FUNSEARCH)
    assert entry.load() is FunSearch


def _funsearch_yaml(tmp_path, params: str) -> str:
    write_template("builtin", tmp_path)
    config = tmp_path / "experiment.yml"
    text = config.read_text().replace(
        '  - type: random_search\n    name: "RS-{llm}"\n',
        f'  - type: funsearch\n    name: "FS-{{llm}}"\n    params: {params}\n',
    )
    assert "type: funsearch" in text
    config.write_text(text)
    return str(config)


def test_yaml_reports_misspelled_parameters(tmp_path):
    config = _funsearch_yaml(tmp_path, "{num_island: 4}")
    with pytest.raises(ConfigError, match="num_island"):
        load_plan(config)


def test_iohblade_check_runs_funsearch(tmp_path, capsys):
    config = _funsearch_yaml(tmp_path, "{num_islands: 2}")
    assert cli.main(["check", config]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "FS-" in out and "[FAIL]" not in out, out


# endregion
