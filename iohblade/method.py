from abc import ABC, abstractmethod

from .llm import LLM
from .problem import Problem


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

    @abstractmethod
    def __call__(self, problem: Problem):
        """
        Executes the search algorithm and returns the best found solution.

        Args:
            problem (Problem): Problem instance being optimized.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        pass
