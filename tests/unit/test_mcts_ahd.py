import random

from iohblade.llm import LLM
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.methods.mcts_ahd import MCTS

class DummyLLM(LLM):
    def __init__(self) -> None:
        pass

    def sample_solution(self, session_messages: list[dict[str, str]],
        parent_ids=[],
        HPO=False,
        base_code: str | None = None,
        diff_mode: bool = False,
        **kwargs,) -> Solution:
        code = '''
import random

class RandomSearchMock:
    """
    Simple random search mock program.
    Each call returns a random (x, y) within given bounds.
    """

    def __call__(self, x_bounds, y_bounds, last_fitness: float = None):
        """
        Returns a random point (x, y) within the given bounds.
        
        Args:
            x_bounds (tuple): (min_x, max_x)
            y_bounds (tuple): (min_y, max_y)
            last_fitness (float): fitness of last point (ignored in this mock)
        
        Returns:
            tuple: (x, y) random point
        """
        x = random.uniform(*x_bounds)
        y = random.uniform(*y_bounds)
        return (x, y)
'''
        solution = Solution(
            code=code,
            name="Random Search",
            description="Random Seach Algorithm in 2-D Space.",
        )
        return solution
    
    def _query(self, session, **kwargs):
        pass
    
    def query(self, session):
        return 'The algorithm uses random search using uniform random number generator, within the provided bounds. It applies the random ' \
        'selection each time __call__ is called, and qualifies which then the framework returns fitness of last evaluation, however algorithm ' \
        'doesn\'t use that info at all.'

class DummyProblem(Problem):
    def evaluate(self, solution):
        solution.fitness = random.random() * 100
    
    def test(self, solution):
        return self.evaluate(solution)
    
    def to_dict(self):
        return super().to_dict()

def test_initialise_works_correctly():
    """
    Paper states there should be 1 i1 and N - 1 e1 nodes.
    """
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, budget=10)
    mcts_instance.initialise(5)
    assert len(mcts_instance.root.children) == 5

    i1_count = 0
    e1_count = 0

    for child in mcts_instance.root.children:
        if child.approach == 'i1':
            i1_count += 1
        elif child.approach == 'e1':
            e1_count += 1
        else:
            ValueError(f"Got approach {child.approach} which is not in ['i1', 'e1'], violates paper specification.")
    assert i1_count == 1
    assert e1_count == 4


def test_expansion_generates_2kp2_nodes():
    """
    Each expansion must generate 2 (expantion factor) + 2 nodes.
    expansion factor number of nodes in m1, m2.
    1 s1 node.
    1 e2 node.
    """
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, budget=10, expansion_factor=2)
    mcts_instance.initialise(5)
    for child in mcts_instance.root.children:
        mcts_instance.expansion(child)
    
    for intitalised_nodes in mcts_instance.root.children:
        m1_nodes = m2_nodes = e2_nodes = s1_nodes = 0
        assert len(intitalised_nodes.children) == 6
        for child in intitalised_nodes.children:
            match child.approach:
                case 'm1': m1_nodes += 1
                case 'm2': m2_nodes += 1
                case 's1': s1_nodes += 1
                case 'e2': e2_nodes += 1
                case x: raise ValueError(f"Got approach {x}, which is not in m1, m2, s1, e2.")
        assert m1_nodes == mcts_instance.expansion_factor
        assert m2_nodes == mcts_instance.expansion_factor
        assert s1_nodes == 1
        assert e2_nodes == 1
    
    mcts_instance = MCTS(llm, problem, budget=10, expansion_factor=6)
    mcts_instance.initialise(5)
    for child in mcts_instance.root.children:
        mcts_instance.expansion(child)
    
    for intitalised_nodes in mcts_instance.root.children:
        m1_nodes = m2_nodes = e2_nodes = s1_nodes = 0
        assert len(intitalised_nodes.children) == 14
        for child in intitalised_nodes.children:
            match child.approach:
                case 'm1': m1_nodes += 1
                case 'm2': m2_nodes += 1
                case 's1': s1_nodes += 1
                case 'e2': e2_nodes += 1
                case x: raise ValueError(f"Got approach {x}, which is not in m1, m2, s1, e2.")
        assert m1_nodes == mcts_instance.expansion_factor
        assert m2_nodes == mcts_instance.expansion_factor
        assert s1_nodes == 1
        assert e2_nodes == 1

