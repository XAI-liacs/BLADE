"""Tests for the ShinkaEvolve method (#217). No real LLM or embedding calls."""

import math
import os
import re
from typing import Any

import numpy as np
import pytest

from iohblade import cli
from iohblade.config import load_plan
from iohblade.config.errors import ConfigError
from iohblade.config.templates import write_template
from iohblade.fitness import Fitness
from iohblade.llm import LLM, Multi_LLM
from iohblade.methods import ShinkaEvolve
from iohblade.methods.shinka_evolve import (
    AsymmetricUCB,
    ProgramDatabase,
    ngram_embedding,
)
from iohblade.methods.shinka_evolve.novelty import cosine_similarity
from iohblade.methods.shinka_evolve.patches import PatchError, apply_diff, extract_code
from iohblade.problem import Problem
from iohblade.registry import BuiltinMethod, ComponentKind, registry
from iohblade.solution import Solution

_CURRENT = re.compile(r"# Current program.*?```python\n(.*?)```", re.DOTALL)
_FIX = re.compile(r"# Incorrect Program to Fix.*?```python\n(.*?)```", re.DOTALL)


def _code(i: int) -> str:
    return f"class A{i}:\n    def __call__(self, f):\n        value = {i}\n"


class _LLM(LLM):
    """Answers each prompt type in ShinkaEvolve's reply formats.

    ``diff_ok=False`` makes diffs fail (SEARCH text not in the program).
    """

    def __init__(self, diff_ok=True, novel=True, model="fake"):
        super().__init__("", model)
        self.diff_ok = diff_ok
        self.novel = novel
        self.prompts: list[str] = []
        self.sessions: list[list[dict]] = []
        self.counter = 0

    def _query(self, session, **kwargs):
        prompt = session[0]["content"]
        self.prompts.append(prompt)
        self.sessions.append(list(session))
        if "Respond with NOVEL or NOT_NOVEL" in prompt:
            return "NOVEL: different." if self.novel else "NOT_NOVEL: same."
        if "Create a standalone summary" in prompt:
            return "**Program Name: x**"
        if "ONLY PERFORM THE ANALYSIS" in prompt:
            return "## Successful Algorithmic Patterns\n- value"
        if "actionable recommendations" in prompt:
            return "1. Try larger values.\n   Really.\n2. Try smaller values."
        self.counter += 1
        i = self.counter
        if "SEARCH/REPLACE" in prompt:
            parent = _CURRENT.search(prompt).group(1)
            line = next(l for l in parent.splitlines() if "value =" in l)
            search = line if self.diff_ok else "    missing line"
            return (
                f"<NAME>edit{i}</NAME>\n<DESCRIPTION>change {i}</DESCRIPTION>\n"
                f"<DIFF>\n<<<<<<< SEARCH\n{search}\n=======\n"
                f"        value = {i}\n>>>>>>> REPLACE\n</DIFF>"
            )
        return (
            f"<NAME>rewrite{i}</NAME>\n<DESCRIPTION>new {i}</DESCRIPTION>\n"
            f"<CODE>\n```python\n{_code(i)}```\n</CODE>"
        )

    def get_config(self) -> dict[str, Any]:
        return {}


class _Problem(Problem):
    """Fitness = the ``value`` in the code; ``fail_every``-th call fails."""

    def __init__(self, minimisation=False, fail_every=0, scores=None):
        super().__init__(name="values")
        self.minimisation = minimisation
        self.fail_every = fail_every
        self.scores = scores
        self.task_prompt = "TASK"
        self.calls = 0
        self.seen: list[Solution] = []

    def __call__(self, solution, logger=None):
        self.calls += 1
        self.seen.append(solution)
        if self.fail_every and self.calls % self.fail_every == 0:
            return solution.set_scores(self.failure_fitness(), "failed", "error!")
        match = re.search(r"value = (\d+)", solution.code)
        if self.scores is not None:
            value = self.scores[(self.calls - 1) % len(self.scores)]
        elif match:
            value = float(match.group(1))
        else:
            return solution.set_scores(self.failure_fitness(), "no code", "no code")
        return solution.set_scores(value, f"value {value}")

    def evaluate(self, solution):
        return self(solution)

    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


# region Budget and direction.
@pytest.mark.parametrize("budget", [0, 1, 2, 5, 17])
@pytest.mark.parametrize("fail_every", [0, 3])
def test_evaluates_exactly_budget(budget, fail_every):
    np.random.seed(0)
    problem = _Problem(fail_every=fail_every)
    best = ShinkaEvolve(_LLM(), budget)(problem)
    assert problem.calls == budget
    assert (best is None) == (budget == 0)


@pytest.mark.parametrize("minimisation", [True, False])
def test_best_follows_direction_and_skips_failures(minimisation):
    np.random.seed(1)
    problem = _Problem(minimisation, fail_every=4)
    best = ShinkaEvolve(_LLM(), 20)(problem)
    values = [s.fitness for s in problem.seen if math.isfinite(s.fitness)]
    assert best.fitness == (min(values) if minimisation else max(values))


def test_all_failures_return_a_failed_solution():
    problem = _Problem(fail_every=1)
    best = ShinkaEvolve(_LLM(), 4)(problem)
    assert problem.calls == 4 and best.fitness == -math.inf


def test_multi_objective_fitness_is_rejected():
    class _Multi(_Problem):
        def __call__(self, solution, logger=None):
            return solution.set_scores(Fitness({"a": 1.0, "b": 2.0}))

    with pytest.raises(ValueError, match="single-objective"):
        ShinkaEvolve(_LLM(), 3)(_Multi())


def test_no_files_are_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.random.seed(0)
    ShinkaEvolve(_LLM(), 8, novelty_rejection=True, meta_interval=3)(_Problem())
    assert os.listdir(tmp_path) == []


# endregion


# region Patches and prompts.
def test_children_are_patched_from_their_parent():
    np.random.seed(0)
    problem = _Problem()
    ShinkaEvolve(_LLM(), 15)(problem)
    by_id = {s.id: s for s in problem.seen}
    operators = {s.operator for s in problem.seen[1:]}
    assert problem.seen[0].operator == "init"
    assert operators <= {"diff", "full", "cross"} and "diff" in operators
    for child in problem.seen[1:]:
        assert child.generation > 0 and child.parent_ids
        assert child.parent_ids[0] in by_id
        assert child.metadata["island"] in (0, 1)
        if child.operator == "diff":
            parent = by_id[child.parent_ids[0]]
            assert child.code != parent.code
            assert child.code.split("value")[0] == parent.code.split("value")[0]
        if child.operator == "cross":
            assert len(child.parent_ids) == 2


def test_prompts_show_direction_task_and_inspirations():
    np.random.seed(0)
    llm = _LLM()
    ShinkaEvolve(llm, 12)(_Problem(minimisation=True))
    prompts = [p for p in llm.prompts if "# Current program" in p]
    assert prompts and all("# Problem\n\nTASK" in p for p in prompts)
    assert all("Fitness (lower is better)" in p for p in prompts)
    assert any("# Prior programs" in p for p in prompts)
    assert all("EVOLVE-BLOCK" not in p for p in llm.prompts)


def test_failed_patches_are_retried_then_the_reply_is_evaluated():
    np.random.seed(0)
    problem = _Problem()
    llm = _LLM(diff_ok=False)
    method = ShinkaEvolve(
        llm,
        6,
        patch_type_probs={"diff": 1.0},
        max_patch_attempts=2,
        max_patch_resamples=2,
    )
    method(problem)
    assert problem.calls == 6
    retries = [s for s in llm.sessions if len(s) == 3]
    assert retries and "previous edit was not successful" in retries[0][2]["content"]
    for child in problem.seen[1:]:
        assert "SEARCH block 1 was not found" in child.metadata["patch_error"]
        assert not math.isfinite(child.fitness)  # a diff reply has no code
    # 2 resamples x 2 attempts per generation.
    assert len(llm.prompts) == 5 * 4 + 1


def test_fix_mode_on_islands_without_a_correct_program():
    np.random.seed(0)

    class _FirstFails(_Problem):
        def __call__(self, solution, logger=None):
            if self.calls == 0:
                self.calls += 1
                self.seen.append(solution)
                return solution.set_scores(-math.inf, "failed", "Boom error")
            return super().__call__(solution, logger)

    problem = _FirstFails()
    llm = _LLM()
    ShinkaEvolve(llm, 3)(problem)
    assert "# Incorrect Program to Fix" in llm.prompts[1]
    assert "Boom error" in llm.prompts[1]
    assert problem.seen[1].operator == "fix"


def test_meta_recommendations_reach_later_prompts():
    np.random.seed(0)
    llm = _LLM()
    method = ShinkaEvolve(llm, 12, meta_interval=4, meta_max_recommendations=1)
    method(_Problem())
    assert method.recommendations == ["Try larger values. Really."]
    assert "ONLY PERFORM THE ANALYSIS" in "".join(llm.prompts)
    assert any(
        "# Potential Recommendations\n\nTry larger values. Really." in p
        for p in llm.prompts
    )


# endregion


# region Novelty rejection.
def _constant(code):
    return [1.0, 0.0]


@pytest.mark.parametrize(
    "judge, novel, rejections", [(True, False, 2), (True, True, 0), (False, None, 2)]
)
def test_novelty_rejection(judge, novel, rejections):
    np.random.seed(0)
    problem = _Problem()
    llm = _LLM(novel=bool(novel))
    method = ShinkaEvolve(
        llm,
        6,
        novelty_rejection=True,
        novelty_judge=judge,
        embedding=_constant,
        max_novelty_attempts=3,
    )
    method(problem)
    assert problem.calls == 6  # rejected candidates are not evaluated
    for child in problem.seen[1:]:
        assert child.metadata["novelty_rejections"] == rejections
    judged = [p for p in llm.prompts if "NOT_NOVEL" in p]
    assert bool(judged) == judge
    # Rejected candidates cost LLM calls, not evaluations.
    proposals = len(llm.prompts) - len(judged) - 1
    assert proposals == 5 * (rejections + 1)


def test_novelty_off_or_dissimilar_accepts():
    np.random.seed(0)
    llm = _LLM()
    ShinkaEvolve(llm, 6, novelty_rejection=True)(_Problem())  # n-gram default
    assert not any("NOT_NOVEL" in p for p in llm.prompts)


def test_ngram_embedding():
    a = ngram_embedding("def f(x):\n    return x + 1\n")
    b = ngram_embedding("def f(x):   return x + 1")
    c = ngram_embedding("class Totally:\n    different = [1, 2, 3]\n")
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) < 0.5
    assert cosine_similarity(ngram_embedding(""), a) == 0.0


# endregion


# region Bandit.
def test_bandit_tries_unseen_arms_then_prefers_improving_arm():
    np.random.seed(0)
    bandit = AsymmetricUCB(2, c=0.0, epsilon=0.2, decay=1.0)
    assert np.allclose(bandit.probabilities(), [0.5, 0.5])
    bandit.n_submitted[0] = 1
    assert np.allclose(bandit.probabilities(), [0.0, 1.0])
    bandit.set_baseline(1.0)
    assert bandit.update(0, 3.0, parent_score=2.0) == 1.0  # over the parent
    assert bandit.update(1, 0.5) == 0.0  # no improvement
    assert bandit.update(1, None) == 0.0  # failure
    assert np.allclose(bandit.probabilities(), [0.8, 0.2])


def test_bandit_decay_and_single_arm():
    bandit = AsymmetricUCB(1)
    assert bandit.select() == 0 and bandit.probabilities()[0] == 1.0
    bandit = AsymmetricUCB(2, decay=0.5)
    bandit.update(0, 2.0)
    assert bandit.s[0] == 1.0 and bandit.divs[0] == 0.5
    with pytest.raises(ValueError):
        AsymmetricUCB(0)


def test_bandit_chooses_llms_of_a_multi_llm():
    np.random.seed(0)
    good, bad = _LLM(model="good"), _LLM(model="bad")

    class _Scored(_Problem):
        def __call__(self, solution, logger=None):
            self.calls += 1
            self.seen.append(solution)
            gain = 1.0 if solution.metadata.get("llm") == "good" else -1.0
            return solution.set_scores(self.calls * gain, "ok")

    problem = _Scored()
    ShinkaEvolve(Multi_LLM([good, bad]), 40, bandit_c=0.1)(problem)
    used = [s.metadata["llm"] for s in problem.seen[1:]]
    assert used.count("good") > used.count("bad") > 0


# endregion


# region Database.
def _program(i: int) -> Solution:
    return Solution(code=_code(i), name=f"A{i}")


def test_initial_program_is_copied_and_correct_children_seed_islands():
    db = ProgramDatabase(num_islands=3)
    initial = db.add_initial(_program(0), None)
    assert len(db.programs) == 3 and not db.initialised
    assert db.sample().fix
    child = db.add(_program(1), 1.0, initial, 1)
    assert db.initialised and child.island == initial.island == 0
    assert {p.island for p in db.programs if p.correct} == {0, 1, 2}
    assert initial.children == 1


def test_weighted_selection_prefers_good_and_unexplored_parents():
    np.random.seed(0)
    db = ProgramDatabase(num_islands=1)
    root = db.add_initial(_program(0), 0.0)
    good = db.add(_program(1), 10.0, root, 1)
    middle = db.add(_program(2), 5.0, root, 1)
    picks = [db.sample().parent for _ in range(300)]
    assert picks.count(good) > picks.count(middle) > picks.count(root)
    good.children = 1000
    picks = [db.sample().parent for _ in range(300)]
    assert picks.count(good) < 150


def test_power_law_and_beam_search():
    np.random.seed(0)
    db = ProgramDatabase(num_islands=1, parent_selection="power_law")
    root = db.add_initial(_program(0), 0.0)
    best = db.add(_program(1), 3.0, root, 1)
    db.add(_program(2), 2.0, root, 1)
    picks = [db.sample().parent for _ in range(300)]
    assert picks.count(best) > picks.count(root)
    db = ProgramDatabase(num_islands=1, parent_selection="beam_search", num_beams=2)
    root = db.add_initial(_program(0), 0.0)
    best = db.add(_program(1), 3.0, root, 1)
    second = db.add(_program(2), 2.0, root, 1)
    assert db.sample().parent is best
    best.children = 2
    assert db.sample().parent is second
    with pytest.raises(ValueError):
        ProgramDatabase(parent_selection="random")


def test_inspirations_exclude_parent_and_are_sorted():
    np.random.seed(0)
    db = ProgramDatabase(
        num_islands=1, num_archive_inspirations=2, num_top_k_inspirations=2
    )
    root = db.add_initial(_program(0), 0.0)
    for i in range(1, 8):
        db.add(_program(i), float(i), root, 1)
    for _ in range(20):
        sample = db.sample()
        ids = [p.id for p in sample.inspirations]
        assert sample.parent.id not in ids and len(set(ids)) == len(ids) == 4
        scores = [p.score for p in sample.inspirations]
        assert scores == sorted(scores)
        assert 7.0 in scores or sample.parent.score == 7.0  # the global best


def test_archive_keeps_the_best():
    db = ProgramDatabase(num_islands=1, archive_size=3)
    root = db.add_initial(_program(0), 0.0)
    for i in range(1, 6):
        db.add(_program(i), float(i), root, 1)
    assert sorted(p.score for p in db.archive) == [3.0, 4.0, 5.0]


def test_migration_moves_programs_but_keeps_island_best():
    np.random.seed(0)
    db = ProgramDatabase(num_islands=2, migration_rate=0.5, migration_interval=5)
    root = db.add_initial(_program(0), 0.0)
    children = [db.add(_program(i), float(i), root, i) for i in range(1, 5)]
    assert db.migrate(3) == []
    moved = db.migrate(5)
    assert moved and all(p.island == 1 for p in moved)
    assert children[-1] not in moved  # island 0's best stays
    assert ProgramDatabase(num_islands=2).migrate(10) == []  # rate 0: off


# endregion


# region Patch parsing.
def test_apply_diff_exact_and_reindented():
    code = "def f(x):\n    if x:\n        return 1\n    return 0\n"
    new, n = apply_diff(
        code, "<<<<<<< SEARCH\n    return 0\n=======\n    return 2\n>>>>>>> REPLACE"
    )
    assert n == 1 and new.endswith("return 2\n")
    reply = (
        "<<<<<<< SEARCH\nif x:\n    return 1\n=======\nif x > 1:\n    return 3\n"
        ">>>>>>> REPLACE"
    )
    new, _ = apply_diff(code, reply)
    assert "    if x > 1:\n        return 3\n" in new


@pytest.mark.parametrize(
    "reply, message",
    [
        ("no blocks", "No SEARCH/REPLACE"),
        ("<<<<<<< SEARCH\nnope\n=======\nyes\n>>>>>>> REPLACE", "not found"),
        (
            "<<<<<<< SEARCH\n    return 0\n=======\n    return 0\n>>>>>>> REPLACE",
            "does not change",
        ),
    ],
)
def test_apply_diff_errors(reply, message):
    with pytest.raises(PatchError, match=message):
        apply_diff("def f(x):\n    return 0\n", reply)


def test_extract_code():
    assert extract_code("<CODE>\n```python\nx = 1\n```\n</CODE>") == "x = 1\n"
    assert extract_code("text\n```python\nx = 1\n```") == "x = 1\n"
    assert extract_code("<CODE>\nx = 2\n</CODE>") == "x = 2\n"
    with pytest.raises(PatchError):
        extract_code("no code here")
    with pytest.raises(PatchError):
        extract_code("<CODE>\n```python\n\n```\n</CODE>")


# endregion


# region Configuration.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_islands": 0},
        {"budget": -1},
        {"migration_rate": 1.5},
        {"parent_selection": "random"},
        {"patch_type_probs": {"mutate": 1.0}},
        {"patch_type_probs": {"diff": 0.0}},
        {"patch_type_probs": {"diff": -1}},
        {"patch_type_probs": []},
        {"embedding": "openai"},
        {"meta_interval": 0},
        {"novelty_threshold": 2},
        {"max_patch_attempts": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ShinkaEvolve(_LLM(), **{"budget": 10, **kwargs})


def test_config_and_registry():
    method = ShinkaEvolve(_LLM(), 10, num_islands=3, embedding=_constant)
    config = method.get_config()
    assert config["name"] == "ShinkaEvolve"
    assert config["config"]["num_islands"] == 3
    assert config["config"]["embedding"] == "custom"
    assert method.num_islands == 3
    assert method.to_dict()["patch_type_probs"] == {
        "diff": 0.6,
        "full": 0.3,
        "cross": 0.1,
    }
    entry = registry.get(ComponentKind.METHOD, BuiltinMethod.SHINKA_EVOLVE)
    assert entry.load() is ShinkaEvolve


def _shinka_yaml(tmp_path, params: str) -> str:
    write_template("builtin", tmp_path, force=True)
    config = tmp_path / "experiment.yml"
    text = config.read_text().replace(
        '  - type: random_search\n    name: "RS-{llm}"\n',
        f'  - type: shinka_evolve\n    name: "Shinka-{{llm}}"\n    params: {params}\n',
    )
    assert "type: shinka_evolve" in text
    config.write_text(text)
    return str(config)


def test_yaml_parameters(tmp_path):
    config = _shinka_yaml(tmp_path, "{num_island: 4}")
    with pytest.raises(ConfigError, match="num_island"):
        load_plan(config)
    config = _shinka_yaml(
        tmp_path, "{num_islands: 3, patch_type_probs: {diff: 0.5, full: 0.5}}"
    )
    plan = load_plan(config)
    methods = [m for m in plan.methods if isinstance(m, ShinkaEvolve)]
    assert methods[0].patch_type_probs == {"diff": 0.5, "full": 0.5}


def test_iohblade_check_runs_shinka_evolve(tmp_path, capsys):
    config = _shinka_yaml(tmp_path, "{num_islands: 2, novelty_rejection: true}")
    assert cli.main(["check", config]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Shinka-" in out and "[FAIL]" not in out, out


# endregion
