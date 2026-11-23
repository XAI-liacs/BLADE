import json
import os
import threading
from datetime import datetime

import pandas as pd
from ConfigSpace.read_and_write import json as cs_json

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import convert_to_serializable
from .base import ExperimentLogger, RunLogger

try:  # pragma: no cover - import guard
    import mlflow
    import mlflow.exceptions
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
except Exception as e:  # pragma: no cover - handled in __init__
    mlflow = None
    MlflowClient = None
    _import_error = e
else:
    _import_error = None


def _run_key(method_name, llm_name, problem_name, seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        pass
    return (method_name, llm_name, problem_name, seed)


class MLFlowExperimentLogger(ExperimentLogger):
    """
    An ExperimentLogger subclass that keeps the original file-based logging,
    and also logs runs to MLflow. The idea is:
      - Call open_run() at the start of a run: this creates an MLflow run
        (``MlflowClient.create_run``) for it.
      - Run your optimization, logging step-level data via MLFlowRunLogger.
      - Call add_run() at the end, which logs final info and terminates the
        MLflow run (``FINISHED``); a run that raised is terminated as
        ``FAILED`` by mark_run_failed().

    Every call names its MLflow run explicitly (``client.log_*(run_id, ...)``)
    instead of using MLflow's *active run*: that is a single run per thread,
    so it cannot hold several open runs (``Experiment`` opens all runs before
    they execute) and is unknown in the worker threads that execute the
    runs.
    """

    def __init__(self, name="", read=False, mlflow_tracking_uri=None):
        """
        Args:
            name (str): The name of the experiment (used as the MLflow experiment name).
            read (bool): If True, read the existing log directory for file-based logs only.
            mlflow_tracking_uri (str): The MLflow Tracking URI, e.g.
                ``"sqlite:///mlflow.db"`` (a local database file) or the URL of
                a tracking server. MLflow 3 refuses the plain file store
                (``"file:/tmp/mlruns"``) unless the environment variable
                ``MLFLOW_ALLOW_FILE_STORE=true`` is set. Without a URI,
                MLflow's default (``MLFLOW_TRACKING_URI``) is used.
        """
        if mlflow is None:
            raise ImportError(
                "MLflow is not installed. Install with `pip install mlflow`."
            ) from _import_error
        super().__init__(name=name, read=read)
        # If you want to store the logs in some custom place
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)

        # Create or retrieve a named MLflow experiment
        try:
            self.experiment_id = self.client.create_experiment(name)
        except mlflow.exceptions.MlflowException:
            # Already exists
            self.experiment_id = self.client.get_experiment_by_name(name).experiment_id

        self._mlflow_runs = {}  # run key -> MLflow run id (open runs)
        self._mlflow_lock = threading.Lock()
        self._pending_run_id = None
        self._mlflow_run_active = False  # whether any MLflow run is open
        self._current_run_id = None  # the most recently opened run

    def _before_open_run(self, run_name, method, problem, budget, seed):
        llm_name = self._get_clean_llm_model_name(getattr(method, "llm", None))
        tags = {
            "method_name": str(method.name),
            "problem_name": str(problem.name),
            "llm_name": str(llm_name),
            "seed": str(seed),
        }
        run = self.client.create_run(self.experiment_id, run_name=run_name, tags=tags)
        run_id = run.info.run_id
        with self._mlflow_lock:
            key = _run_key(method.name, llm_name, problem.name, seed)
            previous = self._mlflow_runs.pop(key, None)
            if previous is not None:  # reopened without add_run: abandon it
                self._terminate(previous, "KILLED")
            self._mlflow_runs[key] = run_id
            self._pending_run_id = run_id
            self._mlflow_run_active = True
            self._current_run_id = run_id

    def _create_run_logger(self, run_name, budget, progress_cb):
        run_id, self._pending_run_id = self._pending_run_id, None
        return MLFlowRunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
            progress_callback=progress_cb,
            run_id=run_id,
            client=self.client,
        )

    def open_run(self, method, problem, budget=100, seed=0):
        return super().open_run(method, problem, budget, seed)

    def _terminate(self, run_id, status):
        try:
            self.client.set_terminated(run_id, status=status)
        except Exception:  # never let tracking break the experiment
            pass

    def _take_run_id(self, method, problem, llm, seed):
        """Remove and return the open MLflow run of a BLADE run (or None)."""
        llm_name = self._get_clean_llm_model_name(llm)
        with self._mlflow_lock:
            run_id = self._mlflow_runs.pop(
                _run_key(method.name, llm_name, problem.name, seed), None
            )
            self._mlflow_run_active = bool(self._mlflow_runs)
            if run_id is not None and run_id == self._current_run_id:
                self._current_run_id = None
        return run_id

    def mark_run_failed(self, method, problem, seed, error):
        """Records a failed run and terminates its MLflow run as FAILED."""
        run_id = self._take_run_id(method, problem, getattr(method, "llm", None), seed)
        if run_id is not None:
            try:
                self.client.set_tag(run_id, "error", str(error)[:5000])
            except Exception:
                pass
            self._terminate(run_id, "FAILED")
        super().mark_run_failed(method, problem, seed, error)

    def add_run(
        self,
        method: Method,
        problem: Problem,
        llm: LLM,
        solution: Solution,
        log_dir="",
        seed=None,
    ):
        """
        Normally called at the end of a run.
        1) Logs final run metadata to the run's MLflow run
        2) Terminates the MLflow run (FINISHED)
        3) Calls super().add_run(...) so we keep the file-based logs
        """
        # --- MLflow logging ---
        run_id = self._take_run_id(method, problem, llm, seed)
        if run_id is None:
            # No open run (add_run without open_run): create one.
            run_name = f"{method.name}_{problem.name}_seed{seed}"
            run_id = self.client.create_run(
                self.experiment_id, run_name=run_name
            ).info.run_id

        client = self.client
        client.log_param(run_id, "method_name", method.name)
        client.log_param(run_id, "problem_name", problem.name)
        client.log_param(run_id, "llm_name", llm.model)
        if seed is not None:
            client.log_param(run_id, "seed", seed)

        # Log final fitness as a metric
        final_fitness = _as_metric(solution.fitness)
        client.log_metric(
            run_id,
            "final_fitness",
            float("nan") if final_fitness is None else final_fitness,
        )

        # Log a serialized object as an artifact
        try:
            rel_log_dir = os.path.relpath(log_dir, self.dirname)
        except ValueError:
            rel_log_dir = log_dir
        run_object = {
            "method_name": method.name,
            "problem_name": problem.name,
            "llm_name": llm.model,
            "method": method.to_dict(),
            "problem": problem.to_dict(),
            "llm": llm.to_dict(),
            "solution": solution.to_dict(),
            "log_dir": rel_log_dir,
            "seed": seed,
        }
        client.log_text(
            run_id,
            json.dumps(convert_to_serializable(run_object), indent=2),
            artifact_file="final_run_object.json",
        )

        # End the MLflow run
        self._terminate(run_id, "FINISHED")

        # --- File-based logging (super) ---
        super().add_run(
            method=method,
            problem=problem,
            llm=llm,
            solution=solution,
            log_dir=log_dir,
            seed=seed,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["_mlflow_lock"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._mlflow_lock = threading.Lock()

    # Optionally override get_data and get_problem_data to *also*
    # combine data from MLflow runs. Right now, we leave them alone
    # so they read from your local experimentlog.jsonl as before.


def _as_metric(value):
    """A fitness as a float metric (None if it is not a single number)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MLFlowRunLogger(RunLogger):
    """
    A RunLogger subclass that logs data to MLflow *and* to file.

    The MLflow run is named explicitly by ``run_id`` (given by
    MLFlowExperimentLogger.open_run), so logging works from any thread. A run
    logger created without ``run_id`` logs to MLflow's active run of the
    calling thread, as older versions did.
    """

    def __init__(
        self,
        name="",
        root_dir="",
        budget=100,
        progress_callback=None,
        run_id=None,
        client=None,
    ):
        # We do want to keep the parent's file-based logging directories
        super().__init__(name, root_dir, budget, progress_callback=progress_callback)
        self.run_id = run_id
        self.client = client
        if run_id is not None and client is None:
            self.client = MlflowClient()
        self._step_lock = threading.Lock()
        self._fitness_step = 0

    def _log_text(self, text, artifact_file):
        if self.run_id is not None:
            self.client.log_text(self.run_id, text, artifact_file)
        else:
            mlflow.log_text(text, artifact_file=artifact_file)

    def _log_metric(self, key, value, step=None):
        if self.run_id is not None:
            self.client.log_metric(self.run_id, key, value, step=step)
        elif step is None:
            mlflow.log_metric(key, value)
        else:
            mlflow.log_metric(key, value, step=step)

    def log_conversation(self, role, content, cost=0.0, tokens=0):
        """
        Logs conversation details to MLflow, plus calls super() to keep the local file logs if you wish.
        """
        conversation = {
            "role": role,
            "time": str(datetime.now()),
            "content": content,
            "cost": float(cost),
            "tokens": int(tokens),
        }
        with self._step_lock:
            attempt = self.attempt
            self.attempt += 1
        # Example: We can log each conversation snippet as a text artifact
        self._log_text(
            json.dumps(conversation), artifact_file=f"conversation_{attempt}.json"
        )

        # Also do the usual file-based logs or any other logic you have in the parent
        super().log_conversation(role, content, cost, tokens)

    def log_individual(self, individual):
        """
        Logs an individual solution to MLflow, then calls super() to keep the normal file logging.
        """
        ind_dict = individual.to_dict()

        # Log the fitness as a metric (one step per evaluation)
        fitness = _as_metric(ind_dict.get("fitness"))
        if fitness is not None:
            with self._step_lock:
                step = self._fitness_step
                self._fitness_step += 1
            if self.run_id is not None:
                self._log_metric("fitness", fitness, step=step)
            else:
                self._log_metric("fitness", fitness)

        # Store the entire solution as a JSON artifact
        self._log_text(
            json.dumps(convert_to_serializable(ind_dict)),
            artifact_file=f"solution_{individual.id}.json",
        )

        # Keep file-based logging
        super().log_individual(individual)

    def log_code(self, individual):
        """
        Log code as a text artifact in MLflow, plus the normal .py file in your local run folder.
        """
        self._log_text(individual.code, artifact_file=f"code_{individual.id}.py")
        super().log_code(individual)

    def log_configspace(self, individual):
        """
        If there's a config space, log it to MLflow as well as the local file.
        """
        if individual.configspace is not None:
            cs_text = cs_json.write(individual.configspace)
        else:
            cs_text = "Failed to extract config space"
        self._log_text(cs_text, artifact_file=f"configspace_{individual.id}.json")

        super().log_configspace(individual)

    def budget_exhausted(self):
        """
        Optionally still rely on the file-based approach for counting lines in log.jsonl
        or store a separate counter. For now, we call super() to preserve the old logic.
        """
        return super().budget_exhausted()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_step_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._step_lock = threading.Lock()
