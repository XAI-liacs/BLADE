import json
import os
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import mlflow
import pytest

from iohblade import LLM, Method, Problem, Solution

# Adjust these imports to your actual package structure
from iohblade.loggers import ExperimentLogger, RunLogger
from iohblade.loggers.mlflow import MLFlowExperimentLogger, MLFlowRunLogger


@pytest.fixture
def mock_mlflow():
    """
    Fixture that patches common mlflow calls for the duration of a test.
    Yields a dict of the mocks for easy inspection.
    """
    with patch("mlflow.set_tracking_uri") as mock_set_uri, patch(
        "mlflow.create_experiment"
    ) as mock_create_experiment, patch(
        "mlflow.get_experiment_by_name"
    ) as mock_get_experiment_by_name, patch(
        "mlflow.start_run"
    ) as mock_start_run, patch(
        "mlflow.end_run"
    ) as mock_end_run, patch(
        "mlflow.log_param"
    ) as mock_log_param, patch(
        "mlflow.log_metric"
    ) as mock_log_metric, patch(
        "mlflow.log_text"
    ) as mock_log_text:
        # Provide default behaviors for the experiment calls
        mock_create_experiment.return_value = "12345"  # some fake experiment_id
        fake_experiment = MagicMock()
        fake_experiment.experiment_id = "12345"
        mock_get_experiment_by_name.return_value = fake_experiment

        yield {
            "set_uri": mock_set_uri,
            "create_experiment": mock_create_experiment,
            "get_experiment_by_name": mock_get_experiment_by_name,
            "start_run": mock_start_run,
            "end_run": mock_end_run,
            "log_param": mock_log_param,
            "log_metric": mock_log_metric,
            "log_text": mock_log_text,
        }


@pytest.fixture
def mock_experiment_logger(tmp_path, mock_mlflow):
    """
    Create a MLFlowExperimentLogger object, pointing to a temp path
    (for file-based logs). The MLflow calls are mocked.
    """
    # We'll pass a made-up tracking URI
    name = str(tmp_path / "my_experiment")
    logger = MLFlowExperimentLogger(
        name=name, read=False, mlflow_tracking_uri="file:/fake_tracking"
    )
    return logger


@pytest.fixture
def mock_run_logger(tmp_path):
    """
    Create a MLFlowRunLogger for testing the run-level logic.
    """
    logger = MLFlowRunLogger(name="test_run", root_dir=str(tmp_path), budget=5)
    return logger


def test_experiment_logger_init(mock_experiment_logger, mock_mlflow):
    """
    On init, it should call mlflow.set_tracking_uri(...)
    and either create or get an experiment.
    """
    # mlflow.set_tracking_uri should have been called
    mock_mlflow["set_uri"].assert_called_once_with("file:/fake_tracking")
    # Either create_experiment or get_experiment_by_name is called
    # by the constructor
    create_exp_calls = mock_mlflow["create_experiment"].call_count
    get_exp_calls = mock_mlflow["get_experiment_by_name"].call_count
    assert create_exp_calls + get_exp_calls > 0, "Should create or get experiment."

    # Also check that parent constructor made a directory for file logging:
    assert os.path.exists(mock_experiment_logger.dirname)


def test_experiment_logger_open_run(mock_experiment_logger, mock_mlflow):
    """
    open_run() should start an mlflow run.
    """
    assert not mock_experiment_logger._mlflow_run_active

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

    # Should call mlflow.start_run
    mock_mlflow["start_run"].assert_called_once()
    assert mock_experiment_logger._mlflow_run_active


def test_experiment_logger_add_run(mock_experiment_logger, mock_mlflow):
    """
    add_run() is normally called at the end of a run.
    If no run is active, it opens one automatically.
    Then it logs params, metric, an artifact, ends the run,
    and calls super().add_run(...) for file-based logging.
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

    class DummyLLM(LLM):
        def _query(self, s):
            return "dummy response"

        def to_dict(self):
            return {"model": "dummy_LLM"}

    method = DummyMethod(None, budget=10, name="myMethod")
    problem = DummyProblem(name="myProblem")
    llm = DummyLLM("fake_key", "dummy_model")
    solution = Solution(name="some_solution")
    solution.set_scores(99.0)

    # Check initial state
    assert not mock_experiment_logger._mlflow_run_active

    # Call add_run
    mock_experiment_logger.add_run(
        method=method,
        problem=problem,
        llm=llm,
        solution=solution,
        log_dir="fake_log_dir",
        seed=42,
    )

    # mlflow.start_run should be called because no run was active
    mock_mlflow["start_run"].assert_called_once()
    # We should have logged several params
    log_param_calls = [args for args, _ in mock_mlflow["log_param"].call_args_list]
    # Expect something like ("method_name", "myMethod"), ("problem_name","myProblem"), ...
    expected_params = {"method_name", "problem_name", "llm_name", "seed"}
    for param_name, param_value in log_param_calls:
        assert param_name in expected_params

    # final_fitness as a metric
    mock_mlflow["log_metric"].assert_called_with("final_fitness", 99.0)

    # mlflow.end_run
    mock_mlflow["end_run"].assert_called_once()
    assert not mock_experiment_logger._mlflow_run_active

    # Check file-based logs
    exp_log_path = os.path.join(mock_experiment_logger.dirname, "experimentlog.jsonl")
    assert os.path.isfile(
        exp_log_path
    ), "super().add_run should write to experimentlog.jsonl"
    with open(exp_log_path, "r") as f:
        content = f.read()
    assert "myMethod" in content
    assert "myProblem" in content


def test_run_logger_init_file_structure(mock_run_logger):
    """
    Confirm the parent (file-based) directories exist.
    """
    assert os.path.exists(mock_run_logger.dirname)


def test_run_logger_log_conversation(mock_run_logger, mock_mlflow):
    """
    log_conversation() in MLFlowRunLogger should log an artifact or some text to mlflow
    plus call the parent's file-based logs.
    """
    with patch("mlflow.log_text") as mock_log_text:
        mock_run_logger.log_conversation(
            role="user", content="Hello MLFlow", cost=1.23, tokens=5
        )

    # Check mlflow.log_text call
    mock_log_text.assert_called_once()
    args, kwargs = mock_log_text.call_args
    # The first arg is the text; second is artifact_file
    conversation_str = args[0]
    artifact_name = kwargs["artifact_file"]
    assert "Hello MLFlow" in conversation_str
    assert "conversation" in artifact_name  # e.g. conversation_0.jsonl or something

    # Check the file-based conversation log
    convo_path = os.path.join(mock_run_logger.dirname, "conversationlog.jsonl")
    assert os.path.exists(convo_path)
    with open(convo_path, "r") as f:
        lines = [json.loads(l) for l in f]
    assert any(
        d.get("content") == "Hello MLFlow" and d.get("tokens") == 5 for d in lines
    )


def test_run_logger_log_individual(mock_run_logger, mock_mlflow):
    """
    log_individual() should log fitness and solution as text to mlflow
    plus add to local log.jsonl.
    """
    sol = Solution(name="test_solution")
    sol.set_scores(3.14)
    with patch("mlflow.log_metric") as mock_log_metric, patch(
        "mlflow.log_text"
    ) as mock_log_text:
        mock_run_logger.log_individual(sol)

        mock_log_metric.assert_called_once_with("fitness", 3.14)
        # log_text for entire solution object
        mock_log_text.assert_called_once()
        # Check file-based log
        log_path = os.path.join(mock_run_logger.dirname, "log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path, "r") as f:
            lines = f.read()
        assert "test_solution" in lines


def test_run_logger_log_code(mock_run_logger, mock_mlflow):
    """
    log_code() logs code as a text artifact + local .py file.
    """
    sol = Solution(name="code_sol", code="print('Hello from MLflow')")
    with patch("mlflow.log_text") as mock_log_text:
        mock_run_logger.log_code(sol)

        # Check that it was called with the snippet
        mock_log_text.assert_called_once()
        args, kwargs = mock_log_text.call_args
        code_snippet = args[0]
        artifact_file = kwargs["artifact_file"]
        assert "Hello from MLflow" in code_snippet
        assert "code_" in artifact_file  # "code_<id>.py"
        assert sol.id in artifact_file  # "code_<id>.py"

    # File-based check
    code_dir = os.path.join(mock_run_logger.dirname, "code")
    code_files = os.listdir(code_dir)
    matched = [f for f in code_files if "code_sol" in f and f.endswith(".py")]
    assert len(matched) == 1, "We should have a .py file containing code_sol"
