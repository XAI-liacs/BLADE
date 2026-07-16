import numpy as np
import random
from typing import Any, Optional

from iohblade.llm import Dummy_LLM
from iohblade.fitness import Fitness
from iohblade.solution import Solution
from iohblade.methods import MoEH_Method
from iohblade.problem import Problem

from iohblade.methods.moeh_method.prompts import MoEH_Prompts
from iohblade.methods.moeh_method.population import Population

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
    def __init__(self, objectives:Optional[list[str]]=None, minimisation = True):
        super().__init__()
        self.iteration = 0
        self.objectives = objectives
        self.minimisation = minimisation
        self.all_scores = []

    def evaluate(self, solution: Solution):
        if self.objectives:
            score = {}
            for obj in self.objectives:
                score[obj] = 1 / (1 + np.e ** ((-self.iteration / 4) + 10)) * random.random()
            score = Fitness(score)
            self.all_scores.append(score)
            solution.set_scores(score, f'Scored {score}.')
            self.iteration += 1
            return solution
        score = 1 / (1 + np.e ** ((-self.iteration / 4) + 10)) * random.random()
        self.all_scores.append(score)
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
            'minimisation': self.minimisation,
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
def test_population_matrix_returns_empty_array_when_empty():
    p = Population(10, True)
    x = p._get_delta_domination_matrix()
    assert x.shape == (0, 0)

def test_population_matrix_returns_appropriate_array():
    p = Population(4, True)
    s1 = Solution('ABC')
    s1.set_scores(8, 'test')
    s2 = Solution("ACD")
    s2.set_scores(2, 'test')
    s3 = Solution("DEF")
    s3.set_scores(4, 'test')
    s4 = Solution("FPG")
    s4.set_scores(1, 'test')
    
    p.append(s1)
    p.append(s2)
    p.append(s3)
    p.append(s4)

    m = p._get_delta_domination_matrix()
    print([x.fitness for x in p._population])
    expected = [[0, 0, 0, 0], [-1, 0, -1, 0], [-1, 0, 0, 0], [-1, -1, -1, 0]]
    print('\n', m)
    for i, row in enumerate(m):
        for j, cell in enumerate(row):
            assert expected[i][j] == cell

    p.minimisation = False
    expected = [[0, -1, -1, -1], [0, 0, 0, -1], [0, -1, 0, -1], [0, 0, 0, 0]]

    m = p._get_delta_domination_matrix()
    print('\n', m)
    for i, row in enumerate(m):
        for j, cell in enumerate(row):
            assert expected[i][j] == cell

def test_parent_selection_gracefully_exits_selection_on_empty():
    p = Population(10, True)
    assert p.parent_selection(4) == []

def test_parent_selection_works_properly():
    p = Population(4, True)
    s1 = Solution('ABC')
    s1.set_scores(8, 'test')
    s2 = Solution("ACD")
    s2.set_scores(2, 'test')
    s3 = Solution("DEF")
    s3.set_scores(4, 'test')
    s4 = Solution("FPG")
    s4.set_scores(1, 'test')
    
    p.append(s1)
    p.append(s2)
    p.append(s3)
    p.append(s4)

    selected_population = p.parent_selection(2, True)
    assert len(selected_population) == 2
    for idx1, parent1 in enumerate(selected_population):
        assert parent1 in selected_population
        for idx2, parent2 in enumerate(selected_population):
            if idx1 != idx2:
                assert parent1.code != parent2.code

def test_population_management_fails_gracefully():
    p = Population(10, False)
    p.population_management(test=True)

def test_population_management_sorts_properly():
    p = Population(10, False)
    np.random.seed(420)
    choices = range(ord('a'), ord('z') + 1)
    for i in range(10):
        code = ''
        code = "".join([chr(np.random.choice(choices)) for _ in range(4)])
        print(code)
        s = Solution(code=code)
        s.set_scores(2 ** (10 - i))
        p.append(s)
    
    new_population = p.population_management(test=True)
    for i in range(len(new_population) - 1):
        print(new_population[i].fitness, new_population[i + 1].fitness)
        assert new_population[i].fitness > new_population[i + 1].fitness

    p.minimisation = True
    new_population = p.population_management(True)
    for i in range(len(new_population) - 1):
        print(new_population[i].fitness, new_population[i + 1].fitness)
        assert new_population[i].fitness < new_population[i + 1].fitness

def test_get_best_returns_empty():
    p = Population(10, True)
    assert p.get_best() == []

def test_get_best_handles_scalars():
    p = Population(10, True)
    np.random.seed(420)
    choices = range(ord('a'), ord('z') + 1)
    for i in range(10):
        code = ''
        code = "".join([chr(np.random.choice(choices)) for _ in range(4)])
        print(code)
        s = Solution(code=code)
        s.set_scores(2 ** (10 - i))
        p.append(s)
    
    assert len(p.get_best()) == 1
    assert p.get_best()[0].fitness == 2
    p.minimisation = False
    
    assert len(p.get_best()) == 1
    assert p.get_best()[0].fitness == 1024

def test_get_best_handles_scalars():
    p = Population(10, True)
    np.random.seed(420)
    choices = range(ord('a'), ord('z') + 1)
    for i in range(10):
        code = "".join([chr(np.random.choice(choices)) for _ in range(4)])
        s = Solution(code=code)
        s.set_scores(2 ** (10 - i))
        p.append(s)
    
    assert len(p.get_best()) == 1
    assert p.get_best()[0].fitness == 2
    p.minimisation = False
    
    assert len(p.get_best()) == 1
    assert p.get_best()[0].fitness == 1024


def test_get_best_handles_vectors():
    p = Population(10, True)
    problem = DummyProblemWorks(objectives=['Distance', 'Fitness'])
    np.random.seed(69)
    choices = range(ord('a'), ord('z') + 1)
    for i in range(10):
        code = "".join([chr(np.random.choice(choices)) for _ in range(4)])
        s = Solution(code=code)
        s = problem(s)
        p.append(s)
    
    front = p.get_best()

    for front_member in front:
        for ordinary_member in p._population:
            assert front_member.fitness <= ordinary_member.fitness
    
    p.minimisation = False

    front = p.get_best()

    for front_member in front:
        for ordinary_member in p._population:
            assert front_member.fitness >= ordinary_member.fitness


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

def test_algorithm_fails_gracefully():
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
