import numpy as np
import random
from typing import Any

from iohblade.llm import Dummy_LLM
from iohblade.solution import Solution
from iohblade.methods import MoEH_Method
from iohblade.problem import Problem

from iohblade.methods.moeh_method.prompts import MoEH_Prompts

#region Helper Classes.
class DummyProblem(Problem):
    def evaluate(self, solution: Solution):
        solution.set_scores(float('nan'), 'Invalid solution for invalid problem.')
        return solution
    
    def __call__(self, solution: Solution, logger=None) -> Solution:
            solution = self.evaluate(solution)
            return solution
    
    def test(self, solution: Solution):
        return self.evaluate(solution)
    
    def to_dict(self):
        dictionary = self.__dict__.copy()
        return dictionary

    def get_config(self) -> dict[str, Any]:
        return {
            'tags': 'bruh',
            'name': 'Baka Mendo',
            'prompt': self.get_prompt(),
            'minimisation': True,
            'evaluator': '''    def evaluate(self, solution: Solution):
        solution.set_scores(float('nan'), 'Invalid solution for invalid problem.')
        return solution''',
            'config': {}
        }
    
class DummyProblemWorks(Problem):
    def __init__(self):
        super().__init__()
        self.iteration = 0

    def evaluate(self, solution: Solution):
        score = 1 / (1 + np.e ** ((-self.iteration / 4) + 10)) * random.random()
        solution.set_scores(score, f'Scored {score}, best known 1.0.')
        self.iteration += 1
        return solution
    
    def __call__(self, solution: Solution, logger=None) -> Solution:
            solution = self.evaluate(solution)
            return solution
    
    def test(self, solution: Solution):
        return self.evaluate(solution)
    
    def to_dict(self):
        dictionary = self.__dict__.copy()
        return dictionary
    
    def get_config(self) -> dict[str, Any]:
        return {
            'tags': 'bruh',
            'name': 'Baka Mendo',
            'prompt': self.get_prompt(),
            'minimisation': True,
            'evaluator':"""    def evaluate(self, solution: Solution):
        score = 1 / (np.e ** ((-self.iteration / 10) - 10))
        solution.set_scores(score, f'Scored {score}, best known 1.0.')
        self.iteration += 1
        return solution""",
            'config': {}
        }
#endregion

#region Prompt:
    
def test_i1_prompt():
    task_prompt = 'Write a simple black box optimiser class, that takes in a function, and optimises it. Tha class must contain __call__ method, that is in budget, and returns best result on execution.'
    example_prompt = """
class Optimiser:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, budget = 5000) -> Solution:
        self.budget = budget
        self.initialise()
        self.evaluate()
        while(not self.budget_exhauseted()):
            self.evolve_population()
            self.evaluate()
            self.selection()
        return self.best
"""

    output_prompt = """
Always replay in following format:
# Description : <short description of the approach.>
# Code:
```python
<code>
```
"""

    prompt = MoEH_Prompts.get_prompt_i1(task_prompt, example_prompt, output_prompt)
    assert prompt == task_prompt + '\n' + example_prompt + '\n' + output_prompt

def test_e2_prompt():
    solution = Solution()
    solution.code = '''
class BBOBSolver:
    def __init__(self, func, budget, dim, rng=None):
        """
        Parameters
        ----------
        func : callable
            Objective function taking a 1D numpy array.
        budget : int
            Maximum number of function evaluations.
        dim : int
            Problem dimension.
        rng : np.random.Generator, optional
            Random number generator.
        """
        self.func = func
        self.budget = budget
        self.dim = dim
        self.rng = rng or np.random.default_rng()

        self.best_x = None
        self.best_f = np.inf
        self.evaluations = 0

    def ask(self):
        """
        Generate a candidate solution.
        Override this method for your own algorithm.
        """
        return self.rng.uniform(-5, 5, self.dim)

    def tell(self, x, fx):
        """
        Update the algorithm state.
        Override this method if needed.
        """
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()

    def run(self):
        """
        Run the optimization until the evaluation budget is exhausted.

        Returns
        -------
        best_x : ndarray
            Best solution found.
        best_f : float
            Objective value of the best solution.
        """
        while self.evaluations < self.budget:
            x = self.ask()
            fx = self.func(x)

            self.evaluations += 1
            self.tell(x, fx)

        return self.best_x, self.best_f'''
    solution.description = 'a minimal Python template for optimizing a BBOB function. It accepts a function and an evaluation budget, provides a run() method, and returns the best solution found.'

    solution2 = Solution()
    solution2.code = '''
class Solver:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim

    def run(self):
        best_x, best_f = None, float("inf")

        for _ in range(self.budget):
            x = np.random.uniform(-5, 5, self.dim)
            f = self.func(x)

            if f < best_f:
                best_x, best_f = x, f

        return best_x, best_f
'''

    solution2.description = 'A simple random search optimiser for the bbob problem.'

    solution.set_scores(
        0.83,
        'Scored 0.83, higher is better.'
    )

    solution2.set_scores(
        0.12,
        'Scored 0.12, higher is better.'
    )

    example_prompt = """
class Optimiser:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, budget = 5000) -> Solution:
        self.budget = budget
        self.initialise()
        self.evaluate()
        while(not self.budget_exhauseted()):
            self.evolve_population()
            self.evaluate()
            self.selection()
        return self.best
"""

    output_prompt = """
Always replay in following format:
# Description : <short description of the approach.>
# Code:
```python
<code>
```
"""
    prompt = MoEH_Prompts.get_prompt_e2(example_prompt, output_prompt, [solution, solution2])

    assert solution.code in prompt
    assert solution2.code in prompt

    assert solution.description in prompt
    assert solution2.description in prompt

    assert example_prompt in prompt
    assert output_prompt in prompt

def test_m_prompts():
    solution = Solution()
    solution.code = '''
class BBOBSolver:
    def __init__(self, func, budget, dim, rng=None):
        """
        Parameters
        ----------
        func : callable
            Objective function taking a 1D numpy array.
        budget : int
            Maximum number of function evaluations.
        dim : int
            Problem dimension.
        rng : np.random.Generator, optional
            Random number generator.
        """
        self.func = func
        self.budget = budget
        self.dim = dim
        self.rng = rng or np.random.default_rng()

        self.best_x = None
        self.best_f = np.inf
        self.evaluations = 0

    def ask(self):
        """
        Generate a candidate solution.
        Override this method for your own algorithm.
        """
        return self.rng.uniform(-5, 5, self.dim)

    def tell(self, x, fx):
        """
        Update the algorithm state.
        Override this method if needed.
        """
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()

    def run(self):
        """
        Run the optimization until the evaluation budget is exhausted.

        Returns
        -------
        best_x : ndarray
            Best solution found.
        best_f : float
            Objective value of the best solution.
        """
        while self.evaluations < self.budget:
            x = self.ask()
            fx = self.func(x)

            self.evaluations += 1
            self.tell(x, fx)

        return self.best_x, self.best_f'''
    solution.description = 'A minimal Python template for optimizing a BBOB function. It accepts a function and an evaluation budget, provides a run() method, and returns the best solution found.'
    
    example_prompt = """
class Optimiser:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, budget = 5000) -> Solution:
        self.budget = budget
        self.initialise()
        self.evaluate()
        while(not self.budget_exhauseted()):
            self.evolve_population()
            self.evaluate()
            self.selection()
        return self.best
"""

    output_prompt = """
Always replay in following format:
# Description : <short description of the approach.>
# Code:
```python
<code>
```
"""
    prompt = MoEH_Prompts.get_prompt_m1(example_prompt, output_prompt, solution)
    assert solution.code in prompt
    assert solution.description in prompt
    assert output_prompt in prompt
    assert example_prompt in prompt
    assert "modified version of the provided algorithm" in prompt #M1 is considered mutation.

    prompt = MoEH_Prompts.get_prompt_m2(example_prompt, output_prompt, solution)
    assert solution.code in prompt
    assert solution.description in prompt
    assert output_prompt in prompt
    assert example_prompt in prompt
    assert " a new algorithm that has different parameter settings" in prompt #M1 is considered as hyper parameter optimisation.
#endregion

#region Population

#endregion

def test_initialise_ok():
    llm = Dummy_LLM()
    moeh = MoEH_Method(
        llm=llm,
        budget=10,
        population_size=2,
        iterations=2,
        use_e2_operator=True,
        use_m1_operator=False,
        use_m2_operator=False,
        minimisation=True
    )

    assert moeh.llm == llm
    assert moeh.budget == 10
    assert moeh.population_size == 2
    assert moeh.iterations == 2
    assert moeh.use_e2_operator == True
    assert moeh.use_m1_operator == False
    assert moeh.use_m2_operator == False
    assert moeh.minimisation == True

def test_initialisation_fails_gracefully():
    problem = DummyProblem()
    llm = Dummy_LLM()

    moeh = MoEH_Method(
        llm=llm,
        budget=10,
        population_size=2,
        iterations=2,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        minimisation=True
    )



    out = moeh(problem)

    assert len(out) == 0

def test_initialiation_succeeds():
    problem = DummyProblemWorks()
    llm = Dummy_LLM()

    moeh = MoEH_Method(
        llm,
        20,
        2,
        4,
        True
    )

    solution = moeh(problem)

    assert len(solution) == 1