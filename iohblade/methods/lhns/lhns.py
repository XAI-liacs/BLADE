import re
import math
import random

from iohblade.llm import LLM
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.method import Method

from .prompt import Prompt
from .taboo_table import TabooTable


class LHNS:
    def __init__(
        self,
        problem: Problem,
        llm: LLM,
        method: str,
        cooling_rate: float = 0.1,
        table_size: int = 10,
        budget=100,
        minimisation=False,
    ):
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
            assert method in ["vns", "ils", "ts"]
        except:
            raise ValueError(
                f"Expected method parameter to be one of 'vns', 'ils', 'ts', got {method}"
            )

        self.method = method

        self.prompt_generator = Prompt(problem)
        self.taboo_table: TabooTable = TabooTable(
            size=table_size, minimisation=minimisation
        )

        self.alpha = cooling_rate
        self.budget = budget
        self.minimisation = minimisation

        self.current_solution = Solution()
        self.best_solution = Solution()

    def _log_best_solution(self, next: Solution):
        print(f"Self score: {self.best_solution.fitness}; new score: {next.fitness}")
        if math.isnan(self.best_solution.fitness):
            self.best_solution = next
            return
        else:
            if self.minimisation and self.best_solution.fitness > next.fitness:
                self.best_solution = next
            elif (not self.minimisation) and self.best_solution.fitness < next.fitness:
                self.best_solution = next

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
        print("Simulated Annealing....")
        if math.isnan(self.current_solution.fitness) or math.isinf(
            self.current_solution.fitness
        ):
            if math.isnan(next_solution.fitness) or math.isinf(next_solution.fitness):
                self.current_solution = random.choice(
                    [self.current_solution, next_solution]
                )
            else:
                self.current_solution = next_solution
            return
        if math.isnan(next_solution.fitness) or math.isinf(next_solution.fitness):
            return

        temperature = self.alpha * iteration_number / self.budget

        if self.minimisation:
            if next_solution.fitness < self.current_solution.fitness:
                self.current_solution = next_solution
            else:
                delta = abs(next_solution.fitness - self.current_solution.fitness)
                p = math.e ** (-1 * delta / temperature)
                if random.random() <= p:
                    self.current_solution = next_solution
        else:
            if next_solution.fitness > self.current_solution.fitness:
                self.current_solution = next_solution
            else:
                delta = abs(next_solution.fitness - self.current_solution.fitness)
                p = math.e ** (-1 * delta / temperature)
                if random.random() <= p:
                    self.current_solution = next_solution

    def initialise(self):
        """
        Initialises the lhns loop, by generating a initialised solution and assigning it to `self.current_solution`.

        ## Args:
        `None`: Inline method, that updates the LHNS object.

        ## Returns:
        `None`: Nothing to return.
        """
        print("Initialise....")
        initialisation_prompt = self.prompt_generator.get_prompt_i1()
        solution = None
        for i in range(5):
            try:
                solution = self.llm.sample_solution(
                    [{"role": "user", "content": initialisation_prompt}]
                )
                self.current_solution = solution
            except Exception as e:
                if i == 4:
                    raise e

    def evaluate(self, solution: Solution) -> Solution:
        """
        Evaluates the solution with `problem()` function, and returns if it returns, else returns solution un-mutated.

        ## Args:
        `solution: Solution`: A solution object that needs to be evaluated.

        ## Returns:
        `Solution`: An instance of `solution` input parameters, with updated `fitness`, `feedback`, and `error` members.
        """
        print("Evaluate....")
        evaluated_solution = solution
        evaluated_solution = self.problem(evaluated_solution)
        if evaluated_solution:
            self._log_best_solution(evaluated_solution)
            return evaluated_solution
        return solution

    def _extract_executable_lines_with_indices(
        self, code: str
    ) -> list[tuple[int, str]]:
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
            lines = matched.count("\n")
            return "\n" * lines

        code_preserve_lines = doc_pat.sub(_preserve_lines, code)

        lines = code_preserve_lines.splitlines()

        pattern = re.compile(r"^(?!\s*(?:class\s+\w+|def\s+\w+|#))\s*\S.*$")

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
        delete_count = 0
        # select deletable lines:
        lines_to_delete = []
        for _ in range(int(r * len(destructable_code))):
            delete_line = random.choice(destructable_code)
            lines_to_delete.append(delete_line)
            destructable_code.remove(delete_line)

        lines_to_delete.sort(key=lambda line_to_delete: line_to_delete[0], reverse=True)
        # delete r * len(destructable_code) lines.
        for lc, line_to_delete in lines_to_delete:
            if code_lines[lc] == line_to_delete:
                code_lines.pop(lc)
                delete_count += 1
        print(f"\t\tDeleted {delete_count} lines.")
        return "\n".join(code_lines)

    def mutate_lhns_vns(self, iteration_number: int) -> Solution:
        """
        Apply LHNS VNS from work (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025), the initial r value is not stated,
        so it will be randomly generated.

        ## Args:
        `iteration_number: int`: Current iteration of the algorithm.

        ## Returns
        `Solution`: An instance of solution generated from LHNS-VNS mapping onto the self.current_solution.

        ## Raises:
        If the LLM is not following the contract, or if you run out of tokens, the Errors are reaised respectively.
        """
        print("VNS Mutations....")
        current = self.current_solution
        r = 0.1 * (1 + (iteration_number % 10))

        destroyed_code = self.get_destroyed_code(r, current)
        destruction_count = len(current.code.split("\n")) - len(
            destroyed_code.split("\n")
        )
        destruction_repair_prompt = self.prompt_generator.get_prompt_destroy_repair(
            current, destroyed_code, destruction_count
        )

        for i in range(5):
            try:
                new = self.llm.sample_solution(
                    [{"role": "user", "content": destruction_repair_prompt}]
                )
                return new
            except Exception as e:
                if i == 4:
                    raise e
        return current

    def mutate_lhns_ils(self, iteration_number: int) -> Solution:
        """
        Apply LHNS INS from work (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025), the value of r is set constant to
        0.5, and this mutation eiter generarates a new code with 50% repaired code, or a complete new initialisation (rewrite of the code.)

        ## Args:
        `iteration_number: int`: Current iteration of the algorithm.

        ## Returns
        `Solution`: An instance of solution generated from LHNS-ILS mapping onto the self.current_solution, provided LLM generated
        a solution.

        ## Raises:
        If the LLM is not following the contract, or if you run out of tokens, the Errors are reaised respectively.
        """
        print("ILS Mutation....")
        if iteration_number % 10 == 9:
            initialisation_prompt = self.prompt_generator.get_prompt_i1()
            for i in range(5):
                try:
                    new = self.llm.sample_solution(
                        [{"role": "client", "content": initialisation_prompt}]
                    )
                    return new
                except Exception as e:
                    if i == 4:
                        raise e
                    else:
                        print(
                            "mutate_lhns_ils: Failed to communicate with LLM, retrying..."
                        )
        else:
            current = self.current_solution
            destroyed_code = self.get_destroyed_code(0.5, current)
            destruction_count = len(current.code.split("\n")) - len(
                destroyed_code.split("\n")
            )
            ils_prompt = self.prompt_generator.get_prompt_destroy_repair(
                current, destroyed_code, destruction_count
            )
            for i in range(5):
                try:
                    new = self.llm.sample_solution(
                        [{"role": "user", "content": ils_prompt}]
                    )
                    return new
                except Exception as e:
                    if i == 4:
                        raise e
                    else:
                        print(
                            "mutate_lhns_ils: Failed to communicate with LLM, retrying..."
                        )
        return self.current_solution

    def mutate_lhns_ts(self, iteration_number: int) -> Solution:
        """
        Apply LHNS TS from work (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11043025), the value of r is set constant to
        0.5, if applying taboo search, (once every 10 iterations) else apply VNS mutation for rest of the cases.
        ## Args:
        `iteration_number: int`: Current iteration of the algorithm.

        ## Returns
        `Optional[Solution]`: An instance of solution generated from LHNS-TS mapping onto the self.current_solution, provided LLM generated
        a valid solution.
        """
        print("TS Mutation....")
        current = self.current_solution
        if iteration_number % 10 == 9:
            destroyed_code = self.get_destroyed_code(0.5, current)
            taboo_element = self.taboo_table.get_distinct_entry(current)
            destruction_count = len(current.code.split("\n")) - len(
                destroyed_code.split("\n")
            )
            if taboo_element:
                prompt = self.prompt_generator.get_prompt_taboo_search(
                    current, destroyed_code, taboo_element
                )
            else:
                prompt = self.prompt_generator.get_prompt_destroy_repair(
                    current, destroyed_code, destruction_count
                )
                for i in range(5):
                    try:
                        new = self.llm.sample_solution(
                            [{"role": "user", "content": prompt}]
                        )
                        return new
                    except Exception as e:
                        if i == 4:
                            raise e
        else:
            current = self.current_solution
            destroyed_code = self.get_destroyed_code(0.5, current)
            destruction_count = len(current.code.split("\n")) - len(
                destroyed_code.split("\n")
            )
            ils_prompt = self.prompt_generator.get_prompt_destroy_repair(
                current, destroyed_code, destruction_count
            )
            for i in range(5):
                try:
                    new = self.llm.sample_solution(
                        [{"role": "user", "content": ils_prompt}]
                    )
                    return new
                except Exception as e:
                    if i == 4:
                        raise e
        return current

    def run(self) -> Solution:
        """
        Run the algorithm.
        """
        self.initialise()
        current = self.evaluate(self.current_solution)
        self.current_solution = current
        for iteration_number in range(1, self.budget + 1):
            print(
                f"Gen {iteration_number}: Current: ({current.id}, {current.fitness}, {current.feedback}); Best: ({self.best_solution.id}, {self.best_solution.fitness}, {self.best_solution.feedback})"
            )
            if self.method == "ts":
                print(f"\tTT: {self.taboo_table.get_fitness_series()}")
            match self.method:
                case "vns":
                    next = self.mutate_lhns_vns(iteration_number)
                    self.evaluate(next)
                case "ils":
                    next = self.mutate_lhns_ils(iteration_number)
                    self.evaluate(next)
                case "ts":
                    current = self.current_solution
                    next = self.mutate_lhns_ts(iteration_number)
                    self.evaluate(next)
                    self.taboo_table.update_taboo_search_table(current, next)
                case _:
                    raise ValueError(
                        f"Expected method to be in ['vns', 'ils', 'ts'], got {self.method}"
                    )
            self.simulated_annealing(next, iteration_number)
        return self.best_solution


class LHNS_Method(Method):
    def __init__(self, llm: LLM, budget, method, name="LHNS", minimisation=True):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        ## Args:
            `problem: Problem`: The problem instance to optimize.
            `llm: LLM`: The LLM instance to use for solution generation.
            `budget: int`: The maximum number of evaluations.
            `name: str`: The name of the method.
            `method: str` The method to be used in lhns, a string literal in ['vns', 'ils', 'ts']
            `minimisation: bool`: Objective direction, minimisation if true, maximisation if false.
        """
        super().__init__(llm, budget, f"{name}:{method}")
        self.method = method
        self.minimisation = minimisation

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        self.lhns_instance = LHNS(
            problem=problem,
            llm=self.llm,
            method=self.method,
            budget=self.budget,
            minimisation=self.minimisation,
        )
        return self.lhns_instance.run()

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "LHNS",
            "budget": self.budget,
            "kwargs": {"method": self.method, "minimisation": self.minimisation},
        }
