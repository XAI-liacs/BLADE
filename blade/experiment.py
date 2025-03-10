from abc import ABC, abstractmethod
from .problems import MA_BOB
from .loggers import ExperimentLogger, RunLogger
from .llm import LLM


class Experiment(ABC):
    """
    Abstract class for an entire experiment, running multiple algorithms on multiple problems.
    """

    def __init__(self, methods: list, problems: list, llm: LLM, runs=1):
        """
        Initializes an experiment with multiple methods and problems.

        Args:
            methods (list): List of method instances.
            problems (list): List of problem instances.
        """
        self.methods = methods
        self.problems = problems
        self.runs = runs
        self.log_dirs = []
        self.llm = llm

    @abstractmethod
    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        pass


class MA_BBOB_Experiment(Experiment):
    def __init__(
        self, methods: list, llm: LLM, runs=1, dims=[2, 5], budget_factor=2000
    ):
        """
        Initializes an experiment on MA-BBOB.

        Args:
            methods (list): List of method instances.
            runs (int): Number of runs for each method.
            dims (list): List of problem dimensions.
            budget_factor (int): Budget factor for the problems.
        """
        super().__init__(
            methods, [MA_BOB(dims=dims, budget_factor=budget_factor)], llm, runs
        )
        self.exp_logger = ExperimentLogger("MA_BBOB")

    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        for method in self.methods:
            for problem in self.problems:
                for i in range(self.runs):
                    logger = RunLogger(
                        name=f"{method.__class__.__name__}-{problem.__class__.__name__}",
                        root_dir=self.exp_logger.dirname,
                    )
                    problem.set_logger(logger)
                    self.llm.set_logger(logger)
                    method(problem)
                    self.log_dirs.append(logger.dirname)
        return self.log_dirs
