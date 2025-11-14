from .taboo_table import TabooElement

from iohblade.problem import Problem
from iohblade.solution import Solution


class Prompt:
    def __init__(self, for_problem: Problem):
        self.problem = for_problem

    def get_prompt_i1(self) -> str:
        """
        Returns initialisation prompt.
        """
        return self.problem.get_prompt()

    def get_prompt_o_index(self, individual: Solution) -> str:
        return f"""I have one algorithm with its code as follows.\\
Algorithm description: " + {individual.description}\\
Code:\n
{individual.code}\\
Modify the provided algorithm to improve its performance, where you can determine the degree of modification needed.\\
{self.problem.task_prompt}\\
{self.problem.example_prompt}\\
{self.problem.format_prompt}\\"""

    def get_prompt_destroy_repair(
        self, individual: Solution, destroyed_code: str, deleted_line_count: int
    ) -> str:
        return f"""
I have one algorithm with its partial code as follows.\\
Algorithm description: {individual.description} \\
Code:\\
{destroyed_code}\\
{deleted_line_count} lines have been removed from the provided code. Please review the code, add the necessary lines to get a better result.\\
{self.problem.task_prompt}\\
{self.problem.format_prompt}\\
{self.problem.example_prompt}\\
"""

    def get_prompt_taboo_search(
        self, individual: Solution, destroyed_code: str, taboo_element: TabooElement
    ) -> str:
        b_features = "\n".join(taboo_element.code_feature)
        return f"""
I have algorithm A with its destroyed code, algorithm B's features, i.e. the lines that help improve it's performance in previous iterations.\\
Algorithm A description: {individual.description}\\
Code:\\
{destroyed_code}\\
Algorithm B description: {taboo_element.description}\\
Algorithm B's features: {b_features}
Please review the given code, integrating two algorithm descriptions provided to rearrange it to get a better result for following task.\\
{self.problem.task_prompt}
{self.problem.example_prompt}
{self.problem.format_prompt}"""
