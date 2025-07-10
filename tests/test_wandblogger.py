import json
import os
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest
import wandb

from iohblade import LLM, Method, Problem, Solution

# Adjust imports to your actual package structure
from iohblade.loggers import ExperimentLogger, RunLogger
from iohblade.loggers.wandb import WAndBExperimentLogger, WAndBRunLogger


@pytest.fixture
def mock_wandb():
    """
    Fixture to patch 'wandb.init', 'wandb.log', and 'wandb.finish'
    for the duration of each test.
    """
    with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log, patch(
        "wandb.config"
    ) as mock_config, patch("wandb.finish") as mock_finish:
        # Create a mock "run" object that has a config property
        mock_run = MagicMock()
        mock_run.config = MagicMock()
        patch("wandb.config", new=mock_run.config)

        mock_init.return_value = mock_run

        mock_run.config = mock_config

        yield {
            "init": mock_init,
            "finish": mock_finish,
            "log": mock_log,
            "config": mock_config,
            "run": mock_run,
        }


@pytest.fixture
def mock_run_logger(tmp_path):
    """
    Create a WAndBRunLogger object pointing to a temp directory
    so we can check any file-based logs if desired.
    """
    name = "test_run"
    logger = WAndBRunLogger(name=name, root_dir=str(tmp_path), budget=5)
    return logger


@pytest.fixture
def mock_experiment_logger(tmp_path):
    """
    Create a WAndBExperimentLogger object. The directory is created
    by the parent constructor for file-based logs.
    """
    name = str(tmp_path / "test_experiment")
    logger = WAndBExperimentLogger(
        name=name, entity="dummy_entity", project="dummy_project", read=False
    )
    return logger


def test_experiment_logger_init_file_dir(mock_experiment_logger):
    """
    Verify that the parent constructor created a logging directory
    (file-based) and that wandb was not automatically started.
    """
    assert os.path.exists(
        mock_experiment_logger.dirname
    ), "Directory should be created by super().__init__"
    # W&B run should not be active until open_run() is called
    assert not mock_experiment_logger._wandb_run_active


def test_experiment_logger_open_run(mock_experiment_logger, mock_wandb):
    """
    Ensure open_run() calls wandb.init and sets _wandb_run_active to True.
    """

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

    dm = DummyMethod(None, budget=10, name="MyMethod")
    dp = DummyProblem(name="MyProblem")
    mock_experiment_logger.open_run(dm, dp, 1)
    mock_wandb["init"].assert_called_once()
    assert mock_experiment_logger._wandb_run_active is True


def test_experiment_logger_add_run_file_and_wandb(mock_experiment_logger, mock_wandb):
    """
    add_run() should:
      1. Start a W&B run if not active.
      2. Log final data to wandb.
      3. Finish the run.
      4. Also call super().add_run() for file-based logs.
    """

    # Create sample method/problem/llm/solution
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
        def _query(self, s):
            return "dummy response"

        def to_dict(self):
            return {"model": "dummy_LLM"}

    sol = Solution(name="test_solution")
    sol.set_scores(42.0)

    dm = DummyMethod(None, budget=10, name="MyMethod")
    dp = DummyProblem(name="MyProblem")
    dl = DummyLLM(api_key="key", model="dummy_model")

    # Initially, no active run
    assert not mock_experiment_logger._wandb_run_active

    # Call add_run
    mock_experiment_logger.add_run(
        method=dm, problem=dp, llm=dl, solution=sol, log_dir="some_log_dir", seed=123
    )

    # wandb.init should have been called automatically (since no run was active)
    mock_wandb["init"].assert_called_once()
    # wandb.finish should have been called
    mock_wandb["finish"].assert_called_once()

    # We should have logged final_fitness, final_run_object, etc.
    # wandb.log is invoked multiple times, we can check that "final_fitness" was included
    log_calls = [args[0] for args, kwargs in mock_wandb["log"].call_args_list]
    # log_calls is a list of dicts passed to wandb.log
    found_final_fitness = any("final_fitness" in d for d in log_calls)
    found_final_run_object = any("final_run_object" in d for d in log_calls)
    assert found_final_fitness, "We should have logged final_fitness"
    assert found_final_run_object, "We should have logged final_run_object"

    # Check that the experimentlogger also wrote to experimentlog.jsonl
    # Because we called super().add_run()
    exp_file = os.path.join(mock_experiment_logger.dirname, "experimentlog.jsonl")
    assert os.path.isfile(exp_file), "File-based experiment log should exist"
    with open(exp_file, "r") as f:
        content = f.read()
    assert "MyMethod" in content, "Method name should be in file-based logs"
    assert "MyProblem" in content, "Problem name should be in file-based logs"


def test_run_logger_init_file_dirs(mock_run_logger):
    """
    Confirm that the run logger (inheriting from WAndBRunLogger)
    created directories for local logs.
    """
    assert os.path.exists(mock_run_logger.dirname)


def test_run_logger_log_individual(mock_run_logger, mock_wandb):
    """
    log_individual() should call super() (writing to log.jsonl)
    and also log to wandb.
    """
    sol = Solution(name="test_sol")
    sol.set_scores(3.14)
    mock_run_logger.log_individual(sol)

    # Check W&B log call
    log_calls = [args[0] for args, _ in mock_wandb["log"].call_args_list]
    found_fitness = any("fitness" in d for d in log_calls)
    found_solution = any("solution" in d for d in log_calls)
    assert found_fitness, "Should have logged solution fitness to W&B"
    assert found_solution, "Should have logged entire solution object to W&B"

    # Check file-based logs
    log_file = os.path.join(mock_run_logger.dirname, "log.jsonl")
    with open(log_file, "r") as f:
        lines = f.read()
    assert "test_sol" in lines, "Solution name should be in file-based logs"


def test_run_logger_log_conversation(mock_run_logger, mock_wandb):
    """
    log_conversation() should record to local conversationlog.jsonl and also wandb.log().
    """
    mock_run_logger.log_conversation(
        role="user", content="Hello AI", cost=0.5, tokens=2
    )
    mock_run_logger.log_conversation(
        role="assistant", content="Hi, user!", cost=0.7, tokens=3
    )

    # Check wandb calls
    # The wandb.log calls appear once per log_conversation call
    assert mock_wandb["log"].call_count == 2
    all_calls = [args[0] for args, kwargs in mock_wandb["log"].call_args_list]
    # Each call is a dict like {"conversation": {...}}
    for cdict in all_calls:
        assert "conversation" in cdict

    # Check file logs
    convo_file = os.path.join(mock_run_logger.dirname, "conversationlog.jsonl")
    assert os.path.exists(convo_file)
    with open(convo_file, "r") as f:
        lines = [json.loads(l) for l in f]
    assert any(d.get("content") == "Hello AI" and d.get("tokens") == 2 for d in lines)
    assert any(d.get("content") == "Hi, user!" and d.get("tokens") == 3 for d in lines)


def test_run_logger_log_code(mock_run_logger, mock_wandb):
    """
    log_code() should store code in W&B as text plus local .py file.
    """
    sol = Solution(name="sol_code", code="print('Code snippet')")
    mock_run_logger.log_code(sol)

    # Check wandb call
    # Usually is wandb.log({ "code_sol.id": "print('Code snippet')" })
    log_calls = [args[0] for args, _ in mock_wandb["log"].call_args_list]
    found_code = any(f"code_{sol.id}" in d for d in log_calls)
    assert found_code, "Should log the code snippet to W&B"

    # Check local file
    code_dir = os.path.join(mock_run_logger.dirname, "code")
    matching_files = [
        f for f in os.listdir(code_dir) if f.endswith(".py") and sol.name in f
    ]
    assert len(matching_files) == 1, "A .py file should be created for the code"
