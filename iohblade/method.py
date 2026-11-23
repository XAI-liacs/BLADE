from abc import ABC, abstractmethod
from typing import Any

from .llm import LLM
from .problem import Problem
from .solution import Solution


class Method(ABC):
    def __init__(self, llm: LLM, budget, name="Method"):
        """
        Initializes a method (optimization algorithm) instance.

        Args:
            llm (LLM): LLM instance to be used.
            budget (int): Budget of evaluations.
            name (str): Name of the method (or variation).
        """
        self.llm = llm
        self.budget = budget
        self.name = name

    @property
    def uses_hpo(self) -> bool:
        """Whether the method samples solutions with a configuration space.

        Such solutions are meant to be tuned (HPO) by the problem; an
        :class:`~iohblade.experiment.Experiment` enables HPO on the problems
        that support it (see :func:`iohblade.hpo.configure_hpo`).
        """
        return False

    @abstractmethod
    def __call__(self, problem: Problem) -> Solution:
        """
        Executes the search algorithm and returns the best found solution.

        Args:
            problem (Problem): Problem instance being optimized.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Each method in order to be loggable must return the a dictionary with following keys:
        ```
            {
                `name: str`: Name of the method,
                `source: str`: Global identifier of method, say a repository: <github.com/vendor/repository@<commit_hash>,
                `config: dict`: Key value arguements defining the configuration of the method.
            }
        ```
        """
