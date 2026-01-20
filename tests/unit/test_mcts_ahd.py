import random
import pytest

from iohblade.llm import LLM
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.methods.mcts_ahd.mcts import MCTS, safe_max, safe_min
from iohblade.mcts_node import MCTS_Node

#region test helper functuins
def test_safemax():
    arr1 = [None, None, None, None]
    assert safe_max(arr1) is None
    arr1 += [10]
    assert safe_max(arr1) == 10
    arr1 += [15]
    assert safe_max(arr1) == 15

def test_safemin():
    arr1 = [None, None, None, None]
    assert safe_min(arr1) is None
    arr1 += [10]
    assert safe_min(arr1) == 10
    arr1 += [8]
    assert safe_min(arr1) == 8
#endregion

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
    def __init__(self, maximise: bool = True):
        super().__init__()
        self.count = 0
        self.maximise = maximise

    def evaluate(self, solution):
        self.count += 1
        if self.count % 10 == 5:
            solution.fitness = float('-inf') if self.maximise else float('inf')
            return solution
        solution.fitness = random.random() * 100
        return solution

    def __call__(self, solution):
        return self.evaluate(solution)
    
    def test(self, solution):
        return self.evaluate(solution)
    
    def to_dict(self):
        return super().to_dict()

def test_mcts_node_computed_properties_are_correct():
    r = MCTS_Node(Solution(), 'r')
    
    s1 = MCTS_Node(Solution(), 'i1')
    s2 = MCTS_Node(Solution(), 'e1')
    s3 = MCTS_Node(Solution(), 'e1')

    t1 = MCTS_Node(Solution(), 'e2')
    t2 = MCTS_Node(Solution(), 'm1')
    t3 = MCTS_Node(Solution(), 'm2')
    t4 = MCTS_Node(Solution(), 's1')

    t5 = MCTS_Node(Solution(), 'e2')
    t6 = MCTS_Node(Solution(), 'm1')
    t7 = MCTS_Node(Solution(), 'm2')
    t8 = MCTS_Node(Solution(), 's1')

    t9 = MCTS_Node(Solution(), 'e2')
    t10 = MCTS_Node(Solution(), 'm1')
    t11 = MCTS_Node(Solution(), 'm2')
    t12 = MCTS_Node(Solution(), 's1')

    assert r.is_root == True
    assert r.is_leaf == True

    for next_layer in [s1, s2, s3]:
        r.add_child(next_layer)
    
    assert s1.is_root == False
    assert s2.is_root == False
    assert s3.is_root == False
    assert s1.is_leaf == True
    assert s2.is_leaf == True
    assert s3.is_leaf == True

    for next_layer in [t1, t2, t3, t4]:
        s1.add_child(next_layer)
    
    for next_layer in [t5, t6, t7, t8]:
        s2.add_child(next_layer)
    
    for next_layer in [t9, t10, t11, t12]:
        s3.add_child(next_layer)

    assert r.is_root == True
    assert r.is_leaf == False

    for mid_node in [s1, s2, s3]:
        assert mid_node.is_root == False
        assert mid_node.is_leaf == False

    for leaf_nodes in [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]:
        assert leaf_nodes.is_root == False
        assert leaf_nodes.is_leaf == True

def test_mcts_node_adds_child_correctly():
    a = MCTS_Node(Solution(), 'e1')
    b = MCTS_Node(Solution(), 'e2')

    a.add_child(b)
    assert a.parent == None
    assert b.parent == a
    
    assert a.parent_ids == None
    assert b.parent_ids == a.id

    assert a.children == [b]
    assert b.children == []

def test_mcts_nodes_is_fully_expanded():
    grandparent = MCTS_Node(Solution(), 'root')
    parent = MCTS_Node(Solution(), 'i1')

    child1 = MCTS_Node(Solution(), 'e2')
    parent.Q = 10
    child1.Q = 8
    grandparent.add_child(parent)
    parent.add_child(child1)
    assert parent.is_fully_expanded(5) == False

    child2 = MCTS_Node(Solution(), 's2')
    child2.Q = 5
    parent.add_child(child2)
    assert parent.is_fully_expanded(5) == False

    child3 = MCTS_Node(Solution(), 's2')
    child3.Q = 4
    parent.add_child(child3)
    assert parent.is_fully_expanded(5) == False
    assert parent.is_fully_expanded(3) == True
    child3.Q = 12
    assert parent.is_fully_expanded(5) == True
    assert grandparent.is_fully_expanded(100) == True

def test_expansion_on_non_leaf_node_is_refuted():
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, 10)
    # raise on root expansion. Only initialise is applicable on root.
    with pytest.raises(ValueError):
        mcts_instance.expansion(mcts_instance.root)
    mcts_instance.initialise()
    n1 = MCTS_Node(Solution(), 'e2', depth=2)
    mcts_instance.root.children[0].add_child(n1)

    #raises on non-leaf, non-root, only selection can expand on non leaf nodes.
    with pytest.raises(ValueError):
        child = mcts_instance.root.children[0]
        mcts_instance.expansion(child)

def test_Q_comparison():
    a = MCTS_Node(Solution(), 'm1')
    b = MCTS_Node(Solution(), 'm2')

    # Node Q compare always return false, if either of the comaprison values evaluates to None.
    assert a.is_less_fit_than(b) == False
    assert b.is_less_fit_than(a) == False

    a.Q = 10

    assert a.is_less_fit_than(b) == False
    assert b.is_less_fit_than(a) == False

    b.Q = 15

    assert a.is_less_fit_than(b) == True
    assert b.is_less_fit_than(a) == False

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

def test_get_node_raises_error_for_wrong_node_type():
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, 10)

    with pytest.raises(ValueError):
        mcts_instance._get_new_node('brew', [], 10)

def test_method_crashes_when_llm_don_t_respond_5_times(monkeypatch):
    counter = 0
    def fake_sample_solution(message):
        print(message)
        nonlocal counter
        counter += 1
        print(counter)
        raise Exception("Mock error.")
    
    llm = DummyLLM()
    problem = DummyProblem()

    monkeypatch.setattr(llm, 'sample_solution', fake_sample_solution)
    mcts_instance = MCTS(llm, problem, 10)
    with pytest.raises(Exception):
        mcts_instance._get_new_node('i1', [], 1)

    assert counter == 5

def test_get_s1_nodes_returns_correct_trace():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    n1 = MCTS_Node(Solution(), 'i1', 1)
    n2 = MCTS_Node(Solution(), 'e2', 2)
    n3 = MCTS_Node(Solution(), 's1', 3)

    nn1 = MCTS_Node(Solution(), 'e1', 1)
    nn2 = MCTS_Node(Solution(), 'e2', 2)

    mcts_instance.root.add_child(n1)
    mcts_instance.root.add_child(nn1)
    
    n1.add_child(n2)
    n1.add_child(nn2)

    n2.add_child(n3)

    trace = mcts_instance._get_s1_nodes(n3)

    assert nn1 not in trace
    assert nn2 not in trace

    assert mcts_instance.root not in trace
    assert trace == [n1, n2, n3]

def test_get_e1_node_returns_siblings():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    i1 = MCTS_Node(Solution(), 'i1', 1)
    e1 = MCTS_Node(Solution(), 'e1', 1)
    e11 = MCTS_Node(Solution(), 'e1', 1)
    e2 = MCTS_Node(Solution(), 'e2', 2)
    mcts_instance.root.add_child(i1)
    mcts_instance.root.add_child(e1)
    mcts_instance.root.add_child(e11)
    e1.add_child(e2)

    relevant_nodes = mcts_instance._get_e1_nodes(mcts_instance.root)
    assert i1 in relevant_nodes
    assert e1 in relevant_nodes
    assert e11 in relevant_nodes
    assert e2 not in relevant_nodes

def test_get_e2_nodes_returns_random_sample_from_best_solutions(monkeypatch):

    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    n1 = MCTS_Node(Solution(), 'i1', 1)

    mcts_instance.root.add_child(n1)
    assert mcts_instance._get_e2_nodes(as_child_of_node=n1) == []

    # Add candidates progressively
    f1 = MCTS_Node(Solution(), 'e2', 2)
    mcts_instance.e2_candidates.append(f1)
    result = mcts_instance._get_e2_nodes(as_child_of_node=n1)
    assert result == [f1]

    f2 = MCTS_Node(Solution(), 's1', 2)
    mcts_instance.e2_candidates.append(f2)
    result = mcts_instance._get_e2_nodes(as_child_of_node=n1)
    assert set(result) == {f1, f2}
    assert len(result) == 2

    f3 = MCTS_Node(Solution(), 'm1', 3)
    mcts_instance.e2_candidates.append(f3)
    result = mcts_instance._get_e2_nodes(as_child_of_node=n1)
    assert set(result) <= {f1, f2, f3}
    assert 1 <= len(result) <= 3

    f4 = MCTS_Node(Solution(), 'm2', 4)
    mcts_instance.e2_candidates.append(f4)
    result = mcts_instance._get_e2_nodes(as_child_of_node=n1)
    assert all(x in mcts_instance.e2_candidates for x in result)
    assert len(result) <= 5  # sample up to 5

    f5 = MCTS_Node(Solution(), 'm2', 2)
    mcts_instance.e2_candidates.append(f5)

    mcts_instance.best_solution = f5
    result = mcts_instance._get_e2_nodes(as_child_of_node=n1)

    # Verify all results are from the deduplicated candidate pool
    expected_pool = list(dict.fromkeys(
        mcts_instance.e2_candidates +
        ([mcts_instance.best_solution] if not mcts_instance.best_solution.is_root else [])
    ))

    assert all(x in expected_pool for x in result)
    assert len(result) <= min(5, len(expected_pool))
    # Ensure no duplicate references
    assert len(result) == len(set(map(id, result)))

#region Test get node function raises appropriate errors:
def test_m1_node_generator_raises_when_expected_parent_is_root():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    with pytest.raises(ValueError):
        mcts_instance._get_m1_nodes(mcts_instance.root)

def test_m2_node_generator_raises_when_expected_parent_is_root():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    with pytest.raises(ValueError):
        mcts_instance._get_m2_nodes(mcts_instance.root)

def test_s1_node_generator_raises_when_expected_parent_is_root():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    with pytest.raises(ValueError):
        mcts_instance._get_s1_nodes(mcts_instance.root)

def test_e1_node_generator_raises_when_not_used_to_generate_for_root():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    n1 = MCTS_Node(Solution(), 'e1', 1)
    mcts_instance.root.add_child(n1)
    with pytest.raises(ValueError):
        mcts_instance._get_e1_nodes(n1)

def test_e2_node_generator_raises_when_expected_parent_is_root():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    with pytest.raises(ValueError):
        mcts_instance._get_e2_nodes(mcts_instance.root)

#endregion

def test_simulate_updates_eval_remaining():
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)

    assert mcts_instance.eval_remain == 10

    mcts_instance.initialise()
    for (index, node) in enumerate(mcts_instance.root.children):
        mcts_instance.simulate(node)
        assert mcts_instance.eval_remain == 9 - index
    
def test_simulate_updates_Q_val_appropriately(monkeypatch):
    llm = DummyLLM()
    problem = DummyProblem()
    mcts_instance = MCTS(llm, problem, 10)
    mcts_instance.initialise()

    def invalid_evaluator(node: MCTS_Node):
        node.fitness = float('inf')
        return node

    def invalid_evaluator_minus(node: MCTS_Node):
        node.fitness = float('-inf')
        return node

    monkeypatch.setattr(problem, 'evaluate', invalid_evaluator)
    for node in mcts_instance.root.children:
        mcts_instance.simulate(node)
        assert node.fitness == float('inf')
        assert node.Q is None
    assert mcts_instance.q_max is None
    assert mcts_instance.q_min is None
    
    mcts_instance = MCTS(llm, problem, 10)
    monkeypatch.setattr(problem, 'evaluate', invalid_evaluator_minus)
    mcts_instance.initialise()
    for node in mcts_instance.root.children:
        mcts_instance.simulate(node)
        assert node.fitness == float('-inf')
        assert node.Q is None
    assert mcts_instance.q_max is None
    assert mcts_instance.q_min is None

def test_simulate_updates_best_fitness_values_correctly():
    llm = DummyLLM()
    problem = DummyProblem() # maximisation problem.

    mcts_instance_maximisation = MCTS(llm, problem, 10)
    mcts_instance_maximisation.initialise(10)
    q_min = None
    q_max = None
    for node in mcts_instance_maximisation.root.children:
        mcts_instance_maximisation.simulate(node)
        q_min = safe_min([q_min, node.Q])
        q_max = safe_max([q_max, node.Q])
        assert q_min == mcts_instance_maximisation.q_min
        assert q_max == mcts_instance_maximisation.q_max
    assert mcts_instance_maximisation.best_solution.fitness == q_max
    
    llm = DummyLLM()
    problem = DummyProblem(maximise=False) # Minimisation problem.

    mcts_instance_minimisation  = MCTS(llm, problem, 10, maximisation=False)
    mcts_instance_minimisation.initialise(10)
    q_min = None
    q_max = None
    for node in mcts_instance_minimisation.root.children:
        mcts_instance_minimisation.simulate(node)
        q_min = safe_min([q_min, node.Q])
        q_max = safe_max([q_max, node.Q])
        print(f'\tNode(id:{node.id}, fitness:{node.fitness}, method:{node.approach})')
        assert q_min == mcts_instance_minimisation.q_min
        assert q_max == mcts_instance_minimisation.q_max
    print(f"MCTS Best solution: Node(id:{mcts_instance_minimisation.best_solution.id}, fitness:{mcts_instance_minimisation.best_solution.fitness}, method:{mcts_instance_minimisation.best_solution.approach})")
    assert mcts_instance_minimisation.best_solution.fitness == q_min

def test_selection_exapnds_root_nodes(monkeypatch):
    llm = DummyLLM()
    problem = DummyProblem()

    def is_never_fully_expanded(max_children: int):
        return False
    
    mcts_instance = MCTS(llm, problem, 10)
    mcts_instance.initialise()
    for node in mcts_instance.root.children:
        mcts_instance.simulate(node)
    
    monkeypatch.setattr(mcts_instance.root, 'is_fully_expanded', is_never_fully_expanded)

    expanded_node, selected_node = mcts_instance.selection()
    assert len(expanded_node) == 1
    assert selected_node == mcts_instance.best_solution
    assert expanded_node[0].parent == mcts_instance.root

def test_selection_exapnds_non_root(monkeypatch):
    llm = DummyLLM()
    problem = DummyProblem()

    def is_never_fully_expanded(max_children: int):
        return False
    
    def is_always_fully_expanded(max_children: int):
        return True
    
    mcts_instance = MCTS(llm, problem, 40)
    n1 = MCTS_Node(Solution(), 'i1', 1)
    mcts_instance.root.add_child(n1)

    n2 = MCTS_Node(Solution(), 'm1', 2)
    n1.add_child(n2)

    n3 = MCTS_Node(Solution(), 'm2', 3)
    n2.add_child(n3)

    for node in [n1, n2, n3]:
        mcts_instance.simulate(node)

    mcts_instance.backpropogate(n3)

    monkeypatch.setattr(mcts_instance.root, 'is_fully_expanded', is_always_fully_expanded)
    monkeypatch.setattr(n1, 'is_fully_expanded', is_always_fully_expanded)
    monkeypatch.setattr(n2, 'is_fully_expanded', is_never_fully_expanded)
    monkeypatch.setattr(n3, 'is_fully_expanded', is_always_fully_expanded)

    expanded, _ = mcts_instance.selection()

    assert len(expanded) == 1
    assert expanded[0].approach == 'e2'
    assert expanded[0].parent == n2

def test_selection_always_picks_better_child():
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, 10)
    
    mcts_instance.initialise(10)
    
    [mcts_instance.simulate(node) for node in mcts_instance.root.children]

    _, selected = mcts_instance.selection()
    assert selected.Q == safe_max([node.Q for node in mcts_instance.root.children])
    
    problem = DummyProblem(maximise=False)

    mcts_instance = MCTS(llm, problem, 10, maximisation=False)
    
    mcts_instance.initialise(10)
    
    [mcts_instance.simulate(node) for node in mcts_instance.root.children]

    _, selected = mcts_instance.selection()
    assert selected.Q == safe_min([node.Q for node in mcts_instance.root.children])

def test_back_propagation_logs_rank_list_properly():
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, 100)
    mcts_instance.initialise(10)
    [mcts_instance.simulate(node) for node in mcts_instance.root.children]
    [mcts_instance.backpropogate(node) for node in mcts_instance.root.children]
    assert None not in mcts_instance.rank_list
    assert len(mcts_instance.rank_list) == len(set(
        [node.Q for node in mcts_instance.root.children if node.Q is not None]
    ))

def test_backpropagate_logs_e2_elements():
    llm = DummyLLM()
    problem = DummyProblem()

    mcts_instance = MCTS(llm, problem, 100)
    n1 = MCTS_Node(Solution(), 'i1', 1)
    n2 = MCTS_Node(Solution(), 'e2', 2)
    n3 = MCTS_Node(Solution(), 'm1', 3)
    n4 = MCTS_Node(Solution(), 'm2', 4)

    mcts_instance.root.add_child(n1)
    n1.add_child(n2)
    n2.add_child(n3)
    n3.add_child(n4)

    mcts_instance.simulate(n1)
    mcts_instance.backpropogate(n1)
    mcts_instance.simulate(n2)
    mcts_instance.backpropogate(n2)
    mcts_instance.simulate(n3)
    mcts_instance.backpropogate(n3)
    mcts_instance.simulate(n4)
    mcts_instance.backpropogate(n4)

    assert n1 in mcts_instance.e2_candidates
    assert n2 in mcts_instance.e2_candidates
    assert n3 not in mcts_instance.e2_candidates
    assert n4 not in mcts_instance.e2_candidates
    












