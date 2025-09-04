import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the trackio package can be imported even if not installed
sys.modules.setdefault("trackio", MagicMock())

from iohblade.llm import LLM
from iohblade.loggers.trackio import TrackioExperimentLogger, TrackioRunLogger
from iohblade.method import Method
from iohblade.problem import Problem
from iohblade.solution import Solution


@pytest.fixture
def mock_trackio():
    with patch("iohblade.loggers.trackio.trackio") as mock_module:
        yield mock_module


def test_run_logger_start_and_finish(tmp_path, mock_trackio):
    class DummyMethod(Method):
        def __call__(self, problem):
            pass

        def to_dict(self):
            return {"type": "DummyMethod"}

    class DummyProblem(Problem):
        def get_prompt(self):
            return "prompt"

        def evaluate(self, s):
            return s

        def test(self, s):
            return s

        def to_dict(self):
            return {"type": "DummyProblem"}

    class DummyLLM(LLM):
        def _query(self, *args, **kwargs):
            return ""

    dm = DummyMethod(None, budget=10, name="MyMethod")
    dp = DummyProblem(name="MyProblem")
    llm = DummyLLM(api_key="", model="gpt")
    logger = TrackioExperimentLogger(name=str(tmp_path / "exp"))
    run_logger = logger.open_run(dm, dp, 1, seed=0)
    run_logger.start_run(llm)
    run_logger.finish_run(Solution(name="sol"))

    mock_trackio.init.assert_called_once()
    mock_trackio.finish.assert_called_once()


def test_experiment_logger_add_run(tmp_path, mock_trackio):
    class DummyMethod(Method):
        def __call__(self, problem):
            pass

        def to_dict(self):
            return {"type": "DummyMethod"}

    class DummyProblem(Problem):
        def get_prompt(self):
            return "prompt"

        def evaluate(self, s):
            return s

        def test(self, s):
            return s

        def to_dict(self):
            return {"type": "DummyProblem"}

    method = DummyMethod(None, budget=10, name="MyMethod")
    problem = DummyProblem(name="MyProblem")

    class DummyLLM(LLM):
        def _query(self, *args, **kwargs):
            return ""

    llm = DummyLLM(api_key="", model="gpt")
    solution = Solution(name="sol")
    solution.set_scores(99.0)

    logger = TrackioExperimentLogger(name=str(tmp_path / "exp"))
    logger.add_run(method, problem, llm, solution, log_dir="fake", seed=42)

    mock_trackio.log.assert_not_called()
    mock_trackio.finish.assert_not_called()

    exp_log_path = os.path.join(logger.dirname, "experimentlog.jsonl")
    assert os.path.isfile(exp_log_path)


def test_run_logger_log_conversation(tmp_path, mock_trackio):
    run_logger = TrackioRunLogger(
        name="test_run",
        root_dir=str(tmp_path),
        budget=5,
        project="proj",
        method_name="m",
        problem_name="p",
        seed=0,
    )
    run_logger.log_conversation("user", "Hello Trackio", cost=1.0, tokens=5)

    mock_trackio.init.assert_called_once()
    mock_trackio.log.assert_called()
    convo_path = os.path.join(run_logger.dirname, "conversationlog.jsonl")
    assert os.path.exists(convo_path)
    with open(convo_path, "r") as f:
        lines = [json.loads(l) for l in f]
    assert any(d.get("content") == "Hello Trackio" for d in lines)


def test_run_logger_log_individual(tmp_path, mock_trackio):
    run_logger = TrackioRunLogger(
        name="test_run",
        root_dir=str(tmp_path),
        budget=5,
        project="proj",
        method_name="m",
        problem_name="p",
        seed=0,
    )
    sol = Solution(name="test_solution")
    sol.set_scores(3.14)
    run_logger.log_individual(sol)

    mock_trackio.init.assert_called_once()
    assert mock_trackio.log.call_count >= 1
    log_path = os.path.join(run_logger.dirname, "log.jsonl")
    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        lines = f.read()
    assert "test_solution" in lines
