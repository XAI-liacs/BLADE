
from abc import ABC, abstractmethod
from .problem import Problem
from .llm import LLM

class Method(ABC):
    def __init__(self, llm: LLM, budget):
        """
        Initializes a method (optimization algorithm) instance.

        Args:
            llm (LLM): LLM instance to be used.
            budget (int): Budget of evaluations.
        """
        self.llm = llm
        self.budget = budget

    @abstractmethod
    def __call__(self, problem: Problem):
        """
        Executes the search algorithm and returns the best found solution.

        Args:
            problem (Problem): Problem instance being optimized.
        """
        pass