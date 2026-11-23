"""The methods' prompts keep the context a problem adds in get_prompt()."""

import numpy as np

from iohblade.methods import FunSearch, ShinkaEvolve
from iohblade.methods.prompt_utils import task_context
from iohblade.problem import Problem
from iohblade.solution import Solution


class _Extra(Problem):
    """Like the photonics problems: get_prompt() adds two sections."""

    def __init__(self):
        super().__init__(name="extra")
        self.task_prompt = "TASK.\n"
        self.description_prompt = "DESCRIPTION.\n"
        self.example_prompt = "EXAMPLE CODE.\n"
        self.extra_prompt = "INSIGHTS.\n"
        self.format_prompt = "FORMAT.\n"
        self.calls = 0

    def get_prompt(self):
        return (
            self.task_prompt
            + self.description_prompt
            + self.example_prompt
            + self.extra_prompt
            + self.format_prompt
        )

    def __call__(self, solution, logger=None):
        self.calls += 1
        return solution.set_scores(float(self.calls), "ok")

    def evaluate(self, solution):
        return self(solution)

    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


def test_task_context_drops_example_and_format_only():
    text = task_context(_Extra())
    assert text == "TASK.\nDESCRIPTION.\n\nINSIGHTS."
    kept = task_context(_Extra(), keep_format=True)
    assert "FORMAT." in kept and "EXAMPLE CODE." not in kept


class _Recording:
    model = "recording"

    def __init__(self):
        self.prompts = []

    def sample_solution(self, messages, parent_ids=[], **kwargs):
        self.prompts.append(messages[-1]["content"])
        return Solution(code=f"x = {len(self.prompts)}\n", parent_ids=parent_ids)


def test_funsearch_prompts_keep_problem_context():
    np.random.seed(0)
    llm = _Recording()
    FunSearch(llm, 4, samples_per_prompt=1)(_Extra())
    later = llm.prompts[1:]
    assert later and all("DESCRIPTION." in p and "INSIGHTS." in p for p in later)
    assert all("EXAMPLE CODE." not in p for p in later)
    assert all(p.rstrip().endswith("FORMAT.") for p in later)


def test_shinka_prompts_keep_problem_context():
    from .test_shinka_evolve import _LLM

    np.random.seed(0)
    llm = _LLM()
    ShinkaEvolve(llm, 4)(_Extra())
    later = [p for p in llm.prompts if "# Problem" in p]
    assert later and all("DESCRIPTION." in p and "INSIGHTS." in p for p in later)
    assert all("EXAMPLE CODE." not in p and "FORMAT." not in p for p in later)
