from typing import Any
import numpy as np

from iohblade.llm import Dummy_LLM
from iohblade.methods import MoEH_Method, MutationType
from iohblade.problem import Problem
from iohblade.solution import Solution


class DummyProblem(Problem):
    def __init__(self, logger=None, training_instances=None, test_instances=None, name="Problem", eval_timeout=6000, dependencies=None, imports=None):
        super().__init__(logger, training_instances, test_instances, name, eval_timeout, dependencies, imports)
    
    def evaluate(self, solution: Solution):
        solution.set_scores(float('nan'), 'Invalid solution for invalid problem.')
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
            'prompt': self.task_prompt + self.example_prompt + self.format_prompt,
            'minimisation': True,
            'evaluator': '''    def evaluate(self, solution: Solution):
        solution.set_scores(float('nan'), 'Invalid solution for invalid problem.')
        return solution''',
            'config': {}
        }

class DummyProblemWorks(Problem):
    def __init__(self, logger=None, training_instances=None, test_instances=None, name="Problem", eval_timeout=6000, dependencies=None, imports=None):
        super().__init__(logger, training_instances, test_instances, name, eval_timeout, dependencies, imports)
        self.iteration = 0
    
    def evaluate(self, solution: Solution):
        score = 1 / (np.e ** ((-self.iteration / 10) - 10))
        print(score)
        solution.set_scores(score, f'Scored {score}, best known 1.0.')
        self.iteration += 1
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
            'prompt': self.task_prompt + self.example_prompt + self.format_prompt,
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
    problem = DummyProblem()
    moeh = MoEH_Method(
        llm=llm,
        budget=10,
        population_size=2,
        iterations=2,
        use_e2_operator=True,
        use_m1_operator=False,
        use_m2_operator=False
    )

    _ = moeh(problem)

    assert moeh.llm == llm
    assert moeh.moeh_instance.max_sample_nums == 10
    assert moeh.moeh_instance.population_size == 2
    assert moeh.moeh_instance.iterations == 2
    assert MutationType.E2 in moeh.moeh_instance.allowed_mutation_types
    assert MutationType.M1 not in moeh.moeh_instance.allowed_mutation_types
    assert MutationType.M2 not in moeh.moeh_instance.allowed_mutation_types

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
        use_m2_operator=True
    )

    out = moeh(problem)

    assert len(out) == 0

def test_initialiation_succeeds():
    problem = DummyProblemWorks()
    llm = Dummy_LLM()

    moeh = MoEH_Method(
        llm,
        10,
        1,
        4
    )

    solution = moeh(problem)

    assert len(solution) == 1