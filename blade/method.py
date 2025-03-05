
from abc import ABC, abstractmethod
from problem import Problem
from llm import LLM

class Method(ABC):
    def __init__(self, problem: Problem, llm: LLM, budget):
        """
        Initializes a method (optimization algorithm) instance.

        Args:
            problem (Problem): Problem instance being optimized.
            llm (LLM): LLM instance to be used.
            budget (int): Budget of evaluations.
        """
        self.problem = problem
        self.llm = llm
        self.budget = budget

    @abstractmethod
    def run(self):
        """
        Executes the search algorithm and returns the best found solution.
        """
        pass