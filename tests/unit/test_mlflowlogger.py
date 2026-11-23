import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from iohblade.llm import LLM
from iohblade.loggers.mlflow import MLFlowExperimentLogger, MLFlowRunLogger
from iohblade.method import Method
from iohblade.problem import Problem
from iohblade.solution import Solution


@pytest.fixture
def mock_mlflow():
    """
    Fixture that patches common mlflow calls for the duration of a test.
    Yields a dict of the mocks for easy inspection.
    """
    with (
        patch("mlflow.set_tracking_uri") as mock_set_uri,
        patch("mlflow.create_experiment") as mock_create_experiment,
        patch("mlflow.get_experiment_by_name") as mock_get_experiment_by_name,
        patch("mlflow.start_run") as mock_start_run,
        patch("mlflow.end_run") as mock_end_run,
        patch("mlflow.log_param") as mock_log_param,
        patch("mlflow.log_metric") as mock_log_metric,
        patch("mlflow.log_text") as mock_log_text,
    ):
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
def tracking_uri(tmp_path, monkeypatch):
    """A real, local MLflow tracking store (sqlite) for the test.

    The experiment logger talks to MLflow through ``MlflowClient`` with
    explicit run ids, so it is tested against a real store instead of
    mocks of the fluent API. (MLflow 3 refuses the plain file store.)
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.chdir(tmp_path)  # default artifact location: ./mlruns
    previous = mlflow.get_tracking_uri()
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def mock_experiment_logger(tmp_path, tracking_uri):
    """
    Create a MLFlowExperimentLogger object, pointing to a temp path
    (for file-based logs) and a temporary sqlite tracking store.
    """
    name = str(tmp_path / "my_experiment")
    logger = MLFlowExperimentLogger(
        name=name, read=False, mlflow_tracking_uri=tracking_uri
    )
    return logger


class DummyMethod(Method):
    def __call__(self, problem):
        pass

    def to_dict(self):
        return {"type": "DummyMethod"}

    def get_config(self) -> dict[str, Any]:
        return {}


class DummyProblem(Problem):
    def get_prompt(self):
        return "prompt"

    def evaluate(self, s):
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {"type": "DummyProblem"}

    def get_config(self) -> dict[str, Any]:
        return {}


class DummyLLM(LLM):
    def _query(self, s):
        return "dummy response"

    def to_dict(self):
        return {"model": "dummy_LLM"}

    def get_config(self) -> list[dict[str, Any]]:
        return [{}]


@pytest.fixture
def mock_run_logger(tmp_path):
    """
    Create a MLFlowRunLogger for testing the run-level logic.
    """
    logger = MLFlowRunLogger(name="test_run", root_dir=str(tmp_path), budget=5)
    return logger


def test_experiment_logger_init(mock_experiment_logger, tracking_uri):
    """
    On init, it should set the tracking URI and create (or get) an experiment.
    """
    assert mlflow.get_tracking_uri() == tracking_uri
    experiment = mock_experiment_logger.client.get_experiment(
        mock_experiment_logger.experiment_id
    )
    assert experiment.name == mock_experiment_logger.dirname

    # A second logger for the same experiment name reuses the experiment.
    again = MLFlowExperimentLogger(
        name=mock_experiment_logger.dirname, mlflow_tracking_uri=tracking_uri
    )
    assert again.experiment_id == mock_experiment_logger.experiment_id

    # Also check that parent constructor made a directory for file logging:
    assert os.path.exists(mock_experiment_logger.dirname)


def test_experiment_logger_open_run(mock_experiment_logger):
    """
    open_run() should create an MLflow run for the run (not a fluent active
    run), and hand its id to the run logger.
    """
    assert not mock_experiment_logger._mlflow_run_active

    dm = DummyMethod(DummyLLM("", "dummy_model"), budget=10, name="MyMethod")
    dp = DummyProblem(name="MyProblem")
    run_logger = mock_experiment_logger.open_run(dm, dp, 1)

    assert mock_experiment_logger._mlflow_run_active
    assert isinstance(run_logger, MLFlowRunLogger)
    assert run_logger.run_id == mock_experiment_logger._current_run_id
    run = mock_experiment_logger.client.get_run(run_logger.run_id)
    assert run.info.status == "RUNNING"
    assert run.info.run_name == "MyMethod-dummy_model-MyProblem-0"
    assert mlflow.active_run() is None

    # A second run can be opened while the first one is still open.
    second = mock_experiment_logger.open_run(dm, dp, 1, seed=1)
    assert second.run_id != run_logger.run_id


def test_experiment_logger_add_run(mock_experiment_logger):
    """
    add_run() is normally called at the end of a run.
    If no run is open, it creates one.
    Then it logs params, metric, an artifact, terminates the run,
    and calls super().add_run(...) for file-based logging.
    """
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

    client = mock_experiment_logger.client
    (run,) = client.search_runs([mock_experiment_logger.experiment_id])
    assert run.data.params == {
        "method_name": "myMethod",
        "problem_name": "myProblem",
        "llm_name": "dummy_model",
        "seed": "42",
    }
    # final_fitness as a metric
    assert run.data.metrics["final_fitness"] == 99.0
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert "final_run_object.json" in artifacts
    # the run is terminated
    assert run.info.status == "FINISHED"
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


def test_run_logger_with_run_id_logs_to_that_run(mock_experiment_logger):
    """With a run id, the run logger does not need an active MLflow run."""
    dm = DummyMethod(DummyLLM("", "dummy_model"), budget=10, name="MyMethod")
    run_logger = mock_experiment_logger.open_run(dm, DummyProblem(name="P"), 5)
    sol = Solution(name="sol", code="print(1)")
    sol.set_scores(2.5)
    run_logger.log_individual(sol)
    run_logger.log_code(sol)
    run_logger.log_conversation("user", "hi")

    client = mock_experiment_logger.client
    assert client.get_run(run_logger.run_id).data.metrics["fitness"] == 2.5
    artifacts = {a.path for a in client.list_artifacts(run_logger.run_id)}
    assert {f"solution_{sol.id}.json", f"code_{sol.id}.py", "conversation_0.json"} <= (
        artifacts
    )
    assert mlflow.active_run() is None


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
    with (
        patch("mlflow.log_metric") as mock_log_metric,
        patch("mlflow.log_text") as mock_log_text,
    ):
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
