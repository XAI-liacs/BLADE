import contextlib
import copy
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .loggers import ExperimentLogger
from .problems import MA_BBOB


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
        tasks = {}  # future -> (method, problem, logger, seed)
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
        return

    def _run_single(self, method, problem, logger, seed):
        np.random.seed(seed)
        method.llm.set_logger(logger)
        if self.show_stdout:
            return method(problem)
        with contextlib.redirect_stdout(None):
            with contextlib.redirect_stderr(None):
                return method(problem)


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
