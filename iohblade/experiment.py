import contextlib
import copy
import logging
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .loggers import ExperimentLogger
from .problems import MA_BBOB

BLADE_ASCII = r"""
    ____  __    ___    ____  ______
   / __ )/ /   /   |  / __ \/ ____/
  / __  / /   / /| | / / / / __/
 / /_/ / /___/ ___ |/ /_/ / /___
/_____/_____/_/  |_/_____/_____/
"""


class Experiment(ABC):
    """
    Abstract class for an entire experiment, running multiple algorithms on multiple problems.
    """

    def __init__(
        self,
        methods: list,
        problems: list,
        runs=5,
        budget=100,
        seeds=None,
        show_stdout=False,
        exp_logger=None,
        n_jobs=1,
    ):
        """
        Initializes an experiment with multiple methods and problems.

        Args:
            methods (list): List of method instances.
            problems (list): List of problem instances.
            runs (int): Number of runs for each method.
            budget (int): Number of evaluations per run for each method.
            seeds (list, optional): The exact seeds to use for the runs, len(seeds) overwrites the number of runs if set.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            exp_logger (ExperimentLogger, optiona): The logger object, can be a standard file logger or a WandB or MLFlow logger.
            n_jobs (int): Number of runs to execute in parallel.
        """
        self.methods = methods
        self.problems = problems
        self.runs = runs
        self.budget = budget
        if seeds is None:
            self.seeds = np.arange(runs)
        else:
            self.seeds = seeds
            self.runs = len(seeds)
        self.show_stdout = show_stdout
        self.n_jobs = n_jobs
        if exp_logger is None:
            exp_logger = ExperimentLogger("results/experiment")
        self.exp_logger = exp_logger

    def _clear_console(self) -> None:
        """Clear the console using ANSI escape codes."""
        print("\033c", end="")

    def _print_run_overview(self) -> None:
        """Pretty print the planned runs and their status."""
        runs = getattr(self.exp_logger, "progress", {}).get("runs", [])
        header = f"{'Method':<15} {'Problem':<15} {'Seed':<5} Status"
        lines = ["Run overview:", header, "-" * len(header)]
        for r in runs:
            if r.get("end_time"):
                status = "✅"
            elif r.get("start_time"):
                status = "🔄"
            else:
                status = "⏳"
            lines.append(
                f"{r['method_name']:<15} {r['problem_name']:<15} {r['seed']:<5} {status}"
            )
        print("\n".join(lines))

    def _refresh_console(self) -> None:
        """Clear the console and show the banner and run overview."""
        with contextlib.redirect_stdout(sys.__stdout__):
            self._clear_console()
            self._print_welcome_message()
            self._print_run_overview()
            sys.__stdout__.flush()

    def _print_welcome_message(self) -> None:
        """Print a welcome banner with instructions."""
        message = (
            f"\n{BLADE_ASCII}\n"
            "Welcome to BLADE!\n"
            "You can inspect this experiment in your browser by running:\n"
            "    uv run iohblade-webapp\n\n"
            "While BLADE hides most output from experiments by default, "
            "some logs or warnings may still appear.\n"
        )
        print(message)

    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        total_runs = len(self.problems) * len(self.methods) * len(self.seeds)
        if hasattr(self.exp_logger, "start_progress"):
            self.exp_logger.start_progress(
                total_runs,
                methods=self.methods,
                problems=self.problems,
                seeds=self.seeds,
                budget=self.budget,
            )
        if not self.show_stdout:
            logging.disable(logging.CRITICAL)
            self._refresh_console()
        else:
            self._print_welcome_message()
            self._print_run_overview()
        tasks = {}  # future -> (method, problem, logger, seed)
        # set up problem envs
        for problem in self.problems:
            problem._ensure_env()
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for problem in self.problems:
                for method in self.methods:
                    for seed in self.seeds:
                        np.random.seed(seed)
                        if hasattr(
                            self.exp_logger, "is_run_pending"
                        ) and not self.exp_logger.is_run_pending(method, problem, seed):
                            continue

                        m_copy = copy.deepcopy(method)
                        p_copy = copy.deepcopy(problem)
                        logger = self.exp_logger.open_run(
                            m_copy, p_copy, self.budget, seed
                        )

                        future = executor.submit(
                            self._run_single,
                            m_copy,
                            p_copy,
                            logger,
                            seed,
                        )
                        tasks[future] = (m_copy, p_copy, logger, seed)

            for fut in as_completed(tasks):
                method, problem, logger, seed = tasks[fut]
                solution = fut.result()
                self.exp_logger.add_run(
                    method,
                    problem,
                    method.llm,
                    solution,
                    log_dir=logger.dirname,
                    seed=seed,
                )

                if not self.show_stdout:
                    self._refresh_console()
                else:
                    self._print_run_overview()
        for problem in self.problems:
            problem.cleanup()
        return

    def _run_single(self, method, problem, logger, seed):
        np.random.seed(seed)
        method.llm.set_logger(logger)
        if hasattr(logger, "start_run"):
            logger.start_run(method.llm)
        if self.show_stdout:
            problem._ensure_env()
            result = method(problem)
        else:
            with contextlib.redirect_stdout(None):
                with contextlib.redirect_stderr(None):
                    problem._ensure_env()
                    result = method(problem)
        if hasattr(logger, "finish_run"):
            logger.finish_run(result)
        return result


class MA_BBOB_Experiment(Experiment):
    def __init__(
        self,
        methods: list,
        show_stdout=False,
        runs=5,
        budget=100,
        seeds=None,
        dims=[2, 5],
        budget_factor=2000,
        exp_logger=None,
        n_jobs=1,
        **kwargs,
    ):
        """
        Initializes an experiment on MA-BBOB.

        Args:
            methods (list): List of method instances.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            runs (int): Number of runs for each method.
            budget (int): Number of algorithm evaluations per run per method.
            seeds (list, optional): Seeds for each run.
            dims (list): List of problem dimensions.
            budget_factor (int): Budget factor for the problems.
            **kwargs: Additional keyword arguments for the MA_BBOB problem.
            exp_logger (ExperimentLogger): The logger to store the data.
            n_jobs (int): Number of runs to execute in parallel.
        """
        super().__init__(
            methods,
            [
                MA_BBOB(
                    dims=dims,
                    budget_factor=budget_factor,
                    name="MA_BBOB",
                    **kwargs,
                )
            ],
            runs=runs,
            budget=budget,
            seeds=seeds,
            show_stdout=show_stdout,
            exp_logger=exp_logger,
            n_jobs=n_jobs,
        )
