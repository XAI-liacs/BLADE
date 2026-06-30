import numpy as np
from typing import Any

from iohblade.llm import Dummy_LLM
from iohblade.solution import Solution
from iohblade.methods import MoEH, MoEH_Method
from iohblade.problem import Problem

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
        score = 1 / (1 + np.e ** ((-self.iteration / 10) - 10))
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

    moeh = MoEH(
        llm=llm,
        problem=problem,
        max_sample_nums=10,
        population_size=2,
        iterations=2,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        minimisation=True
    )



    out = moeh.run()

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