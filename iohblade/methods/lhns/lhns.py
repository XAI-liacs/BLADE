import re
import math
import random
from typing import Optional


from iohblade.llm import LLM
from iohblade.problem import Problem
from iohblade.solution import Solution

from .prompt import Prompt
from .taboo_table import TabooTable

class LHNS:
    def __init__(self, problem: Problem, llm: LLM, method: str, cooling_rate: float=0.1, table_size:int=10, budget=100, minimisation=False):
        """
        LHNS is a single individual based optimisation method, that destroyes current iteration of code, by deleting number of certain lines of code
        and uses LLMs to repair them. More info on (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025)

        ## Args:
        `problem: iohblade.Problem` instance of a problem to be solved, with prompts and evaluate function as it's members.
        `llm: iohblade.LLM`: A llm object to communicate with LLMs.
        `method: str`: String literal in one of 'vns', 'ils' or 'ts'.
        `cooling_rate: float`: Used to guide current solution update decision, higher cooling_rate leads to more random selection of next iteration earlier.
        `table_size: int`: Max table size for storing taboo table elements.
        `minimisation: bool`: Optimisation direction of the problem.
        """
        self.problem: Problem = problem
        self.llm: LLM = llm
        self.table_size: int = table_size
        try: 
            assert method in ['vns', 'ils', 'ts']
        except:
            raise ValueError(f"Expected method parameter to be one of 'vns', 'ils', 'ts', got {method}")
        
        self.method = method


        self.prompt_generator = Prompt(problem)
        self.taboo_table : TabooTable = TabooTable(size=table_size, minimisation=minimisation)

        self.alpha = cooling_rate
        self.budget = budget
        self.minimisation = minimisation

        self.current_solution = Solution()
    
    def simulated_annealing(self, next_solution: Solution, iteration_number: int):
        """
        Selects the replacement of `self.current_solution` with next solution with probability $P(r) = e^{-|f_1-f_2|/T$, 
        where $T = \alpha iteration_number/budget$.

        ## Args:
        `next_solution: Solution`: Next repaired solution which is potentially going to replace the current individual.
        `iteration_number: int`: Current iteration number.

        ## Returns:
        `None`: Will replace self.current_solution with aforementioned probability.
        """
        if abs(self.current_solution.fitness) == float('inf'):
            if abs(next_solution.fitness) == float('inf'):
                self.current_solution = random.choice([self.current_solution, next_solution])
            else:
                self.current_solution = next_solution
            return
        if abs(next_solution.fitness) == float('inf'):
            return

        temperature = self.alpha * iteration_number / self.budget
        
        if self.minimisation:
            if next_solution.fitness < self.current_solution.fitness:
                self.current_solution = next_solution
            else:
                delta = abs(next_solution.fitness - self.current_solution.fitness)
                p = math.e ** (-1 * delta / temperature)
                if random.random <= p:
                    self.current_solution = next_solution
        else:
            if next_solution.fitness > self.current_solution.fitness:
                self.current_solution = next_solution
            else:
                delta = abs(next_solution.fitness - self.current_solution.fitness)
                p = math.e ** (-1 * delta / temperature)
                if random.random <= p:
                    self.current_solution = next_solution
            

    def initialise(self):
        """
        Initialises the lhns loop, by generating a initialised solution and assigning it to `self.current_solution`.

        ## Args:
        `None`: Inline method, that updates the LHNS object.

        ## Returns:
        `None`: Nothing to return.
        """
        initialisation_prompt = self.prompt_generator.get_prompt_i1()
        solution = self.llm.sample_solution([{
            'role': 'client',
            'content': initialisation_prompt
        }])
        self.current_solution = solution

    def evaluate(self, solution: Solution):
        """
        Evaluates the solution with `problem.evaluate` function.

        ## Args:
        `solution: Solution`: A solution object that needs to be evaluated.

        ## Returns:
        `Solution`: An instance of `solution` input parameters, with updated `fitness`, `feedback`, and `error` members.
        """

        return self.problem.evaluate(solution)
    
    def _extract_executable_lines_with_indices(self, code: str) -> list[tuple[int, str]]:
        """
        Return list of (line_number, line_text) for lines that are executable,
        excluding class/def declarations, comments and blank lines.

        ## Args;
        `code: str`: A python code string, that is going through destruction phase.

        ## Returns:
        `(line_number, str)`: line_number is 0 indexed line number corresponding to the text, representing executable code.
        """
        doc_pat = re.compile(r'(?s)(""".*?"""|\'\'\'.*?\'\'\')')

        def _preserve_lines(m):
            matched = m.group(0)
            lines = matched.count('\n')
            return '\n' * lines

        code_preserve_lines = doc_pat.sub(_preserve_lines, code)

        lines = code_preserve_lines.splitlines()

        pattern = re.compile(r'^(?!\s*(?:class\s+\w+|def\s+\w+|#))\s*\S.*$')

        result = []
        for i, line in enumerate(lines):
            if pattern.match(line):
                result.append((i, line.rstrip()))
        return result

    def get_destroyed_code(self, r: float, solution: Solution) -> str:
        """
        Destroy repair mutation, takes `self.current_solution`, deletes `r * 100`% of the code. And uses LLM to repair that code\
            fragment into a new code, from that destroyed code.
        ## Args:
            `r: float`: Ratio of executable lines that needs to be destroyed.
            `solution: Solution`: An instance of the current_individual, that needs to be mutated.

        ## Returns:
            `code_fragment: str`: A destroyed code with r * number of executable lines in code, removed.
        """
        code = solution.code
        code_lines = code.split("\n")
        destructable_code = self._extract_executable_lines_with_indices(code)

        # delete r * len(destructable_code) lines.
        for _ in range(int(r * len(destructable_code))):
            delete_line = random.choice(destructable_code)
            destructable_code.remove(delete_line)
            code_lines = code_lines.pop(delete_line[0]) if isinstance(code_lines, list) else code_lines
        
        return "\n".join(code_lines)

    def mutate_lhns_vns(self, iteration_number: int) -> Optional[Solution]:
        """
        Apply LHNS VNS from work (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025), the initial r value is not stated, 
        so it will be randomly generated.

        ## Args:
        `iteration_number: int`: Current iteration of the algorithm.

        ## Returns
        `Solution`: An instance of solution generated from LHNS-VNS mapping onto the self.current_solution.
        """
        current = self.current_solution
        r = 0.1 * (1 + (iteration_number % 10))

        destroyed_code = self.get_destroyed_code(r, current)
        destruction_count = len(current.code.split("\n")) - len(destroyed_code.split("\n"))
        destruction_repair_prompt = self.prompt_generator.get_prompt_destroy_repair(current, destroyed_code, destruction_count)

        try:
            new = self.llm.sample_solution([{
                'role': 'client',
                'content': destruction_repair_prompt
            }])
        except:
            return
        return new
    

    def mutate_lhns_ils(self, iteration_number: int):
        """
        Apply LHNS INS from work (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025), the value of r is set constant to 
        0.5, and this mutation eiter generarates a new code with 50% repaired code, or a complete new initialisation (rewrite of the code.)

        ## Args:
        `iteration_number: int`: Current iteration of the algorithm.

        ## Returns
        `Optional[Solution]`: An instance of solution generated from LHNS-ILS mapping onto the self.current_solution, provided LLM generated 
        a solution.
        """
        if iteration_number % 10 == 9:
            initialisation_prompt = self.prompt_generator.get_prompt_i1()
            try:
                new = self.llm.sample_solution([{
                    'role': 'client',
                    'content': initialisation_prompt
                }])
                return new
            except:
                return
        else:
            current = self.current_solution
            destroyed_code = self.get_destroyed_code(0.5, current)
            destruction_count = len(current.code.split("\n")) - len(destroyed_code.split("\n"))
            ils_prompt = self.prompt_generator.get_prompt_destroy_repair(current, destroyed_code, destruction_count)
            try:
                new = self.llm.sample_solution([{
                    'role': 'client',
                    'content': ils_prompt
                }])
                return new
            except:
                return

    def mutate_lhns_ts(self, iteration_number: int):
        """
        Apply LHNS TS from work (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025), the value of r is set constant to 
        0.5, if applying taboo search, (once every 10 iterations) else apply VNS mutation for rest of the cases.
        ## Args:
        `iteration_number: int`: Current iteration of the algorithm.

        ## Returns
        `Optional[Solution]`: An instance of solution generated from LHNS-TS mapping onto the self.current_solution, provided LLM generated 
        a valid solution.
        """
        current = self.current_solution
        if iteration_number % 10 == 9:
            destroyed_code = self.get_destroyed_code(0.5, current)
            destruction_count = len(current.code.split("\n")) - len(destroyed_code.split("\n"))
            
            taboo_element = self.taboo_table.get_distinct_entry(current)

            if taboo_element:
                taboo_search_prompt = self.prompt_generator.get_prompt_taboo_search(current, destroyed_code, taboo_element)
                try:
                    new = self.llm.sample_solution([{
                        'role': 'client',
                        'content': taboo_search_prompt
                    }])
                    return new
                except:
                    return
            return
        else:
            return self.mutate_lhns_vns(iteration_number)


