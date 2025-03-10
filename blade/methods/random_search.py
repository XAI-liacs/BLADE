
from ..problem import Problem
from ..llm import LLM
from ..method import Method

class RandomSearch(Method):
    def __init__(self, llm: LLM, budget, **kwargs):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        Args:
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            kwargs: Additional arguments for configuring LLaMEA.
        """
        super().__init__(llm, budget)
        

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        best_solution = None
        for i in range(self.budget):
            solution = self.llm.sample_solution([{"role": "client", "content": problem.get_prompt()}])
            solution = problem(solution)
            if (best_solution is None or solution.fitness > best_solution.fitness):
                best_solution = solution
        return best_solution