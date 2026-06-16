import os
import contextlib
import copy
import logging
import sys
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .loggers import ExperimentLogger
from .benchmarks import MA_BBOB

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
        log_stdout=False,
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
            log_stdout (bool): If True, capture stdout and stderr to files via the
                logger when output is hidden.
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
        self.log_stdout = log_stdout
        self.n_jobs = n_jobs
        if exp_logger is None:
            exp_logger = ExperimentLogger("results/experiment")
        self.exp_logger = exp_logger

    def _clear_console(self) -> None:
        """Clear the console using ANSI escape codes."""
        print("\033c", end="")

    def _status_tokens(self):
        enc = (getattr(sys.__stdout__, "encoding", "") or "").lower()
        if "utf" in enc or "65001" in enc:  # utf-8 or Windows UTF-8 codepage
            return {"done": "✅", "running": "🔄", "pending": "⏳"}
        else:
            return {"done": "[DONE]", "running": "[RUN]", "pending": "[WAIT]"}

    def _print_run_overview(self) -> None:
        """Pretty print the planned runs and their status."""
        runs = getattr(self.exp_logger, "progress", {}).get("runs", [])
        header = f"{'Method':<15} {'Problem':<15} {'Seed':<5} Status"
        lines = ["Run overview:", header, "-" * len(header)]
        tokens = self._status_tokens()
        for r in runs:
            if r.get("end_time"):
                status = tokens["done"]
            elif r.get("start_time"):
                status = tokens["running"]
            else:
                status = tokens["pending"]
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
        logger.log_stdout = self.log_stdout
        if hasattr(logger, "start_run"):
            logger.start_run(method.llm)
        if self.show_stdout:
            problem._ensure_env()
            result = method(problem)
        elif self.log_stdout:
            import io

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf):
                with contextlib.redirect_stderr(stderr_buf):
                    problem._ensure_env()
                    result = method(problem)
            logger.log_output(stdout_buf.getvalue(), stderr_buf.getvalue(), append=True)
            if getattr(problem, "_last_stdout", "") or getattr(
                problem, "_last_stderr", ""
            ):
                logger.log_output(
                    getattr(problem, "_last_stdout", ""),
                    getattr(problem, "_last_stderr", ""),
                    append=True,
                )
        else:
            with contextlib.redirect_stdout(None):
                with contextlib.redirect_stderr(None):
                    problem._ensure_env()
                    result = method(problem)
        if hasattr(logger, "finish_run"):
            logger.finish_run(result)
        success = self._log_data(method, problem, logger)
        print(
            "Logged config files successfully"
            if success
            else "Some config logs failed."
        )
        return result

    def _log_data(self, method, problem, logger) -> bool:
        root = logger.get_log_dir()
        success = True
        # method
        method_path = os.path.join(root, "method.json")
        method_config = method.get_config()
        method_success = self._export_config_data(method_config, method_path)
        success = method_success and success

        # problem
        problem_path = os.path.join(root, "problem.json")
        problem_config = problem.get_config()
        problem_success = self._export_config_data(problem_config, problem_path)
        success = problem_success and success

        # llm
        llm_path = os.path.join(root, "llm.json")
        llm_config = method.llm.get_config()
        llm_success = self._export_config_data(llm_config, llm_path)
        success = llm_success and success

        return success

    def _export_config_data(self, config: dict, location: str) -> bool:
        """
        Exports configuration data to associated data, for each method, problem and LLM object, configuration
        using `get_config` function and exports it to the provided location.
        ## Parameters:
            `config: dict`: the configuration dictionary that needs to be exported,
            `location: str`: Full location of the file where the configuration needs to be written.

        ## Returns:
            `bool`: Truth about whether the data was exported.
        """
        try:
            with open(location, "w") as f:
                data = json.dumps(config, indent=4)
                f.write(data)
        except Exception as e:
            print(f"Error logging configuration {config} into file {location}: {e}.")
            return False
        return True


class MA_BBOB_Experiment(Experiment):
    def __init__(
        self,
        methods: list,
        show_stdout=False,
        log_stdout=False,
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
            log_stdout (bool): If True, capture stdout and stderr to files via the
                logger when output is hidden.
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
            log_stdout=log_stdout,
            exp_logger=exp_logger,
            n_jobs=n_jobs,
        )
