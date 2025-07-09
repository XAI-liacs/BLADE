import json
import os
from datetime import datetime

import wandb
from ConfigSpace.read_and_write import json as cs_json

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import convert_to_serializable
from .base import ExperimentLogger, RunLogger


class WAndBExperimentLogger(ExperimentLogger):
    """
    An ExperimentLogger subclass that *also* logs to Weights & Biases (W&B).
    The flow is:
      1. Call open_run() to start a W&B run (wandb.init()).
      2. Perform your optimization (the RunLogger logs intermediate data).
      3. Call add_run() to log final info and finish the W&B run,
         plus the original file-based logging.
    """

    def __init__(self, name="", entity=None, project=None, read=False):
        """
        Args:
            name (str): The name of the experiment. Used in W&B as the 'group' or 'project' name if you prefer.
            entity (str): W&B username or team where you're sending runs.
            project (str): W&B project name.
            read (bool): If True, read logs from an existing directory (original file-based usage).
        """
        # Still call the parent constructor so we keep file-based logging
        super().__init__(name=name, read=read)
        self.wandb_entity = entity
        self.wandb_project = project if project else "DefaultProject"
        self._wandb_run_active = False

    def open_run(self, method, problem, budget=100, seed=0):
        """
        Opens (starts) a new MLflow run for logging.
        Typically call this right before your run, so that the RunLogger can log step data.
        """
        if self._wandb_run_active:
            print("Warning: An MLflow run is already active. Not starting a new one.")
            return
        run_name = f"{method.name}-{problem.name}-{seed}"

        wandb.init(
            entity=self.wandb_entity,
            project=self.wandb_project,
            name=run_name,
            config={},
            # group=name to group runs in your W&B dashboard, if desired
        )
        self._wandb_run_active = True

        self.run_logger = WAndBRunLogger(
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
        Called at the end of a run:
          - Logs final data to W&B (if active).
          - Finishes (wandb.finish()) the run.
          - Calls super().add_run(...) to keep file-based logs.
        """
        # If no run is active, open one automatically
        if not self._wandb_run_active:
            self.open_run(method, problem, seed=seed)

        # Log final parameters
        wandb.config.update({"method_name": method.name}, allow_val_change=True)
        wandb.config.update({"problem_name": problem.name}, allow_val_change=True)
        wandb.config.update({"llm_name": llm.model}, allow_val_change=True)
        if seed is not None:
            wandb.config.update({"seed": seed}, allow_val_change=True)

        # Log final fitness as a metric
        final_fitness = (
            solution.fitness if solution.fitness is not None else float("nan")
        )
        wandb.log({"final_fitness": final_fitness})

        # Log a serialized run object
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
        # For large data, consider W&B artifacts. For quick usage, we do a single JSON log:
        wandb.log({"final_run_object": run_object})

        # End the W&B run
        wandb.finish()
        self._wandb_run_active = False

        # Also do file-based logging
        super().add_run(
            method=method,
            problem=problem,
            llm=llm,
            solution=solution,
            log_dir=log_dir,
            seed=seed,
        )


class WAndBRunLogger(RunLogger):
    """
    A RunLogger subclass that logs intermediate data to Weights & Biases while
    preserving the original file-based logs.
    """

    def __init__(self, name="", root_dir="", budget=100, progress_callback=None):
        """
        As before, call super() so we keep the local directories.
        """
        super().__init__(
            name=name,
            root_dir=root_dir,
            budget=budget,
            progress_callback=progress_callback,
        )

    def log_conversation(self, role, content, cost=0.0, tokens=0):
        """
        Log conversation data to W&B plus the normal local file-based logs.
        """
        conversation = {
            "role": role,
            "time": str(datetime.now()),
            "content": content,
            "cost": float(cost),
            "tokens": int(tokens),
        }
        # wandb.log can be repeated, but if content is large, consider an artifact
        wandb.log({"conversation": conversation})

        # Also do file-based logging
        super().log_conversation(role, content, cost, tokens)

    def log_individual(self, individual: Solution):
        """
        Log an individual solution's data/fitness to W&B.
        Then call super() for file-based logging.
        """
        ind_dict = individual.to_dict()

        # Log the fitness as a metric
        if "fitness" in ind_dict:
            wandb.log({"fitness": ind_dict["fitness"]})

        # Also log the entire solution as a dict
        wandb.log({"solution": ind_dict})

        # Continue with file-based logging
        super().log_individual(individual)

    def log_code(self, individual: Solution):
        """
        For code, you can store it as text in W&B. Large code might benefit from an artifact approach.
        """
        code_text = individual.code
        # For short code, we can do:
        wandb.log({f"code_{individual.id}": code_text})
        # If it's large or you want versioning, you'd do a W&B artifact instead.

        # File-based approach
        super().log_code(individual)

    def log_configspace(self, individual: Solution):
        """
        Log configspace as text in W&B, plus file-based logging.
        """
        if individual.configspace is not None:
            cs_text = cs_json.write(individual.configspace)
        else:
            cs_text = "Failed to extract config space"
        wandb.log({f"configspace_{individual.id}": cs_text})

        super().log_configspace(individual)

    def budget_exhausted(self):
        """
        Optionally keep using the parent's file-based line counting logic, or switch to an in-memory count.
        """
        return super().budget_exhausted()
