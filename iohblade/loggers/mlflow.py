import json
import os
from datetime import datetime

import mlflow
import mlflow.exceptions
import mlflow.pyfunc
import pandas as pd
from ConfigSpace.read_and_write import json as cs_json

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import convert_to_serializable
from .base import ExperimentLogger, RunLogger


class MLFlowExperimentLogger(ExperimentLogger):
    """
    An ExperimentLogger subclass that keeps the original file-based logging,
    and also logs runs to MLflow. The idea is:
      - Call open_run() at the start of a run (this calls mlflow.start_run()).
      - Run your optimization, logging step-level data via MLFlowRunLogger.
      - Call add_run() at the end, which logs final info and ends the MLflow run.
    """

    def __init__(self, name="", read=False, mlflow_tracking_uri=None):
        """
        Args:
            name (str): The name of the experiment (used as the MLflow experiment name).
            read (bool): If True, read the existing log directory for file-based logs only.
            mlflow_tracking_uri (str): The MLflow Tracking URI (e.g. 'file:/tmp/mlruns',
                                       or your remote server).
        """
        super().__init__(name=name, read=read)
        # If you want to store the logs in some custom place
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Create or retrieve a named MLflow experiment
        try:
            self.experiment_id = mlflow.create_experiment(name)
        except mlflow.exceptions.MlflowException:
            # Already exists
            self.experiment_id = mlflow.get_experiment_by_name(name).experiment_id

        self._mlflow_run_active = False  # Track if we have an active run
        self._current_run_id = None

    def open_run(self, method, problem, budget=100, seed=0):
        """
        Opens (starts) a new MLflow run for logging.
        Typically call this right before your run, so that the RunLogger can log step data.
        """
        if self._mlflow_run_active:
            print("Warning: An MLflow run is already active. Not starting a new one.")
            return
        run_name = f"{method.name}-{problem.name}-{seed}"

        run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        self._mlflow_run_active = True
        self._current_run_id = run.info.run_id

        self.run_logger = MLFlowRunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
            progress_callback=lambda: self.increment_eval(
                method.name, problem.name, seed
            ),
        )
        problem.set_logger(self.run_logger)
        return self.run_logger

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
        1) Logs final run metadata to MLflow
        2) Ends the MLflow run
        3) Calls super().add_run(...) so we keep the file-based logs
        """
        # --- MLflow logging ---
        if not self._mlflow_run_active:
            # If there's no open run, you could open one automatically
            # or simply warn. We'll open one just in case:
            run_name = f"{method.name}_{problem.name}_seed{seed}"
            run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            self._mlflow_run_active = True
            self._current_run_id = run.info.run_id

        # Log whatever final data you like
        mlflow.log_param("method_name", method.name)
        mlflow.log_param("problem_name", problem.name)
        mlflow.log_param("llm_name", llm.model)
        if seed is not None:
            mlflow.log_param("seed", seed)

        # Log final fitness as a metric
        final_fitness = (
            solution.fitness if solution.fitness is not None else float("nan")
        )
        mlflow.log_metric("final_fitness", final_fitness)

        # Log a serialized object as an artifact
        rel_log_dir = os.path.relpath(log_dir, self.dirname)
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
        mlflow.log_text(
            json.dumps(convert_to_serializable(run_object), indent=2),
            artifact_file="final_run_object.json",
        )

        # End the MLflow run
        mlflow.end_run()
        self._mlflow_run_active = False
        self._current_run_id = None

        # --- File-based logging (super) ---
        super().add_run(
            method=method,
            problem=problem,
            llm=llm,
            solution=solution,
            log_dir=log_dir,
            seed=seed,
        )

    # Optionally override get_data and get_problem_data to *also*
    # combine data from MLflow runs. Right now, we leave them alone
    # so they read from your local experimentlog.jsonl as before.


class MLFlowRunLogger(RunLogger):
    """
    A RunLogger subclass that logs data to MLflow *and* to file,
    relying on the fact that the MLFlowExperimentLogger has opened a run.
    """

    def __init__(self, name="", root_dir="", budget=100, progress_callback=None):
        # We do want to keep the parent's file-based logging directories
        super().__init__(name, root_dir, budget, progress_callback=progress_callback)

    def log_conversation(self, role, content, cost=0.0, tokens=0):
        """
        Logs conversation details to MLflow, plus calls super() to keep the local file logs if you wish.
        """
        # Log to MLflow (assumes an MLflow run is active)
        conversation = {
            "role": role,
            "time": str(datetime.now()),
            "content": content,
            "cost": float(cost),
            "tokens": int(tokens),
        }
        # Example: We can log each conversation snippet as a text artifact
        mlflow.log_text(
            json.dumps(conversation), artifact_file=f"conversation_{self.attempt}.json"
        )
        self.attempt += 1

        # Also do the usual file-based logs or any other logic you have in the parent
        # (You may have to define such a method if you want to keep conversation logs in a file.)
        super().log_conversation(role, content, cost, tokens)

    def log_individual(self, individual):
        """
        Logs an individual solution to MLflow, then calls super() to keep the normal file logging.
        """
        ind_dict = individual.to_dict()

        # Example: log a metric if there's a fitness
        if "fitness" in ind_dict:
            mlflow.log_metric("fitness", ind_dict["fitness"])

        # Store the entire solution as a JSON artifact
        mlflow.log_text(
            json.dumps(convert_to_serializable(ind_dict)),
            artifact_file=f"solution_{individual.id}.json",
        )

        # Keep file-based logging
        super().log_individual(individual)

    def log_code(self, individual):
        """
        Log code as a text artifact in MLflow, plus the normal .py file in your local run folder.
        """
        mlflow.log_text(individual.code, artifact_file=f"code_{individual.id}.py")
        super().log_code(individual)

    def log_configspace(self, individual):
        """
        If there's a config space, log it to MLflow as well as the local file.
        """
        if individual.configspace is not None:
            cs_text = cs_json.write(individual.configspace)
        else:
            cs_text = "Failed to extract config space"
        mlflow.log_text(cs_text, artifact_file=f"configspace_{individual.id}.json")

        super().log_configspace(individual)

    def budget_exhausted(self):
        """
        Optionally still rely on the file-based approach for counting lines in log.jsonl
        or store a separate counter. For now, we call super() to preserve the old logic.
        """
        return super().budget_exhausted()
