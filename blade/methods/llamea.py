
from ..problem import Problem
from ..llm import LLM
from ..method import Method

from llamea import LLaMEA as LLAMEA_Algorithm
# We import the LLaMEA algorithm directly from the pypi package. This has the advantage that we can easily get the latest version.

class LLaMEA(Method):
    def __init__(self, problem: Problem, llm: LLM, budget, **kwargs):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        Args:
            problem (Problem): The problem instance to optimize.
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            kwargs: Additional arguments for configuring LLaMEA.
        """
        super().__init__(problem, llm, budget)
        self.llamea_instance = LLAMEA_Algorithm(
            f=self.problem.evaluate,  # Ensure evaluation integrates with our framework
            llm=self.llm,
            log=None, #We do not use the LLaMEA native logger, we use the experiment logger instead
            budget=budget,
            **kwargs
        )

    @abstractmethod
    def run(self):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        return self.llamea_instance.run()