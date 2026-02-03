import math
import pytest
import random
from decimal import Decimal

from iohblade.problems.bbob_sboxcost import BBOB_SBOX
from iohblade.solution import Solution
from iohblade.llm import Dummy_LLM, NoCodeException
from iohblade.problem import Problem

from iohblade.methods.lhns.lhns import LHNS
from iohblade.methods.lhns import lhns as lhns_module
from iohblade.methods.lhns.taboo_table import TabooTable, TabooElement

def test_lhns_init_refutes_bad_method_name():
    with pytest.raises(ValueError):
        LHNS(problem=BBOB_SBOX(), llm=Dummy_LLM(), method="Why")

def test_simulated_annealing_rejects_unfit():
    test_cases = [
        {"old": Solution("inf1", "inf1", "inf1"), "new": Solution("inf2", "inf2", "inf2")},
        {"old": Solution("fin", "fin", "fin"), "new": Solution("inf", "inf", "inf")},
        {"old": Solution("inf", "inf", "inf"), "new": Solution("fin", "fin", "fin")},
    ]
    test_cases[0]["old"].fitness = float('inf')
    test_cases[0]["new"].fitness = float('inf')
    
    test_cases[1]["old"].fitness = float('inf')
    test_cases[1]["new"].fitness = 5.0
    
    test_cases[2]["old"].fitness = 10.0
    test_cases[2]["new"].fitness = float('inf')
    

    dllm = Dummy_LLM()
    problem = BBOB_SBOX()
    lhns = LHNS(problem, dllm, method='vns')

    for index, test in enumerate(test_cases):
        lhns.current_solution = test["old"]
        print(f"Old Fitness: {test['old'].fitness}, new fitness: {test['new'].fitness}")
        match index:
            case 0:
                lhns.simulated_annealing(test['new'], 10)
                assert lhns.current_solution.id in [test["new"].id, test['old'].id]      # Randomly selects when both have fitness infinity.
            case 1:
                lhns.simulated_annealing(test["new"], 10)
                assert lhns.current_solution.id == test["new"].id          #Always picks valid fitness over invalid fitness.
            case 2:
                lhns.simulated_annealing(test["new"], 10)
                assert lhns.current_solution.id == test["old"].id          #Always picks valid fitness over invalid fitness.

def test_taboo_table_rejects_unfit_solutions():
    tt = TabooTable(size=10, minimisation=True)
    solution1 = Solution("import Foundation", "Library Import", "Imports foundation Library")
    solution1.set_scores(float('-inf'), feedback="Nothing implemented")
    solution2 = Solution("import Foundations", "Library Import", "Imports foundation Library")
    solution2.set_scores(float('inf'), feedback="Nothing implemented 'Foundations\' library not found.")
    tt.update_taboo_search_table(solution1, solution2)
    assert len(tt.taboo_table) == 0

def test_taboo_table_maintains_order():
    # Ascending in minimisation.
    tt = TabooTable(size=10, minimisation=True)
    prev = Solution()
    prev.set_scores(6)
    for _ in range(10):
        next = Solution()
        next.set_scores(random.random() * 10)
        tt.update_taboo_search_table(next, prev)
        prev = next

    for i in range(1, 10):
        assert tt.taboo_table[i - 1].fitness < tt.taboo_table[i].fitness

    for _ in range(11):
        next = Solution()
        next.set_scores(-10 + (random.random() * 10))
        tt.update_taboo_search_table(next, prev)
        prev = next
    
    print(f"{tt.taboo_table[0].fitness:.2f}", end="\t")
    for i in range(1, 10):
        print(f"{tt.taboo_table[i].fitness:.2f}", end="\t")
        assert tt.taboo_table[i - 1].fitness < tt.taboo_table[i].fitness
        assert -10 < tt.taboo_table[i].fitness < 0
    
    # Descending in maximisation.
    tt = TabooTable(size=10, minimisation=False)
    prev = Solution()
    prev.set_scores(6)
    for _ in range(10):
        next = Solution()
        next.set_scores(random.random() * 10)
        tt.update_taboo_search_table(next, prev)
        prev = next

    for i in range(1, 10):
        assert tt.taboo_table[i - 1].fitness > tt.taboo_table[i].fitness

    for _ in range(11):
        next = Solution()
        next.set_scores(10 + (random.random() * 10))
        tt.update_taboo_search_table(next, prev)
        prev = next
    
    print(f"{tt.taboo_table[0].fitness:.2f}", end="\t")
    for i in range(1, 10):
        print(f"{tt.taboo_table[i].fitness:.2f}", end="\t")
        assert tt.taboo_table[i - 1].fitness > tt.taboo_table[i].fitness
        assert 10 < tt.taboo_table[i].fitness < 20

def test_taboo_feature_works():
    sol1 = Solution(
        '''

import numpy as np
from scipy.special import hermite
from scipy.optimize import minimize
import math

class FourierCandidate:
    def __init__(self, n_terms: int, best_known_configuration: list[float] | None = None):
        """
        Initializes the FourierCandidate with the number of terms and an optional initial configuration.
        """
        self.n_terms = n_terms
        self.best_known_configuration = best_known_configuration
        self.bounds = self._define_bounds()  # Define bounds for each coefficient

    def _define_bounds(self):
         """
         Defines bounds for each coefficient to guide the optimization.
         The first coefficient is encouraged to be negative.
         The last coefficient is forced to be positive.
         Other coefficients have relatively wide bounds centered around zero.
         """
         bounds = [(-1, 1)]  # c[0] has a negative bias

         for _ in range(1, self.n_terms - 1):
             bounds.append((-0.5, 0.5))  # Other coefficients are close to zero

         bounds.append((0.0001, 1))  # Last coefficient is positive and small

         return bounds

    def __call__(self):
        """
        Generates K coefficients for Hermite polynomials and optimizes them to minimize the target function.
        """
        initial_guess = self._generate_initial_guess()

        # Optimization using minimize with constraints
        result = minimize(self._objective_function, initial_guess,
                        method='SLSQP',  # or 'trust-constr'
                        bounds=self.bounds,
                        constraints=({'type': 'ineq', 'fun': lambda x: self._p_at_large_positive(x)}),
                        options={'maxiter': 1000})  # Adjust maxiter as needed

        if result.success:
            return result.x.tolist()
        else:
            print("Warning: Optimization failed. Returning initial guess.")
            return initial_guess.tolist()

    def _generate_initial_guess(self):
        """
        Generates an initial guess for the coefficients, potentially using the best-known configuration if available.
        """
        if self.best_known_configuration is not None and len(self.best_known_configuration) == self.n_terms:
            return np.array(self.best_known_configuration)
        else:
            # Sample random coefficients within the defined bounds
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            return initial_guess


    def _objective_function(self, coefficients):
        """
        Defines the objective function to minimize: r_max^2 / (2*pi).
        """
        r_max = self._find_r_max(coefficients)
        return r_max**2 / (2 * np.pi)

    def _p_x(self, x, coefficients):
        """
        Calculates P(x) = sum_{k=0..K-1} c[k] * H_{4k}(x).
        """
        p_x = 0
        for k, c in enumerate(coefficients):
            p_x += c * hermite(4 * k)(x)
        return p_x

    def _f_x(self, x, coefficients):
        """
        Calculates f(x) = P(x) * exp(-pi * x^2).
        """
        return self._p_x(x, coefficients) * np.exp(-np.pi * x**2)

    def _find_r_max(self, coefficients):
        """
        Finds r_max such that f(r_max) = max(f(x)) for x in [0, inf).
        """
        # Use a simple grid search to find r_max (can be replaced with a more efficient method)
        x_values = np.linspace(0, 5, 500)  # Search up to x=5
        f_values = self._f_x(x_values, coefficients)
        r_max_index = np.argmax(f_values)
        r_max = x_values[r_max_index]
        return r_max

    def _p_at_large_positive(self, coefficients):
        """
        Constraint: Enforces P(x) >= 0 for large |x| (e.g., x=10).
        This ensures that the polynomial doesn't become negative for large x.
        """
        x_large = 10  # Large x value
        return self._p_x(x_large, coefficients)
''', "FourierCandidate",
    "Optimizes Hermite polynomial coefficients using a combination of constrained random sampling and gradient descent to minimize\
         the target function while satisfying constraints.")
    sol1.set_scores(10.6, "Fitness score is 10.6, try improving the solution.")

    sol2 = Solution('''
import numpy as np
from scipy.special import hermite
from scipy.optimize import minimize
import math

class FourierCandidate:
    def __init__(self, n_terms: int, best_known_configuration: list[float] | None = None):
        """
        Initializes the FourierCandidate with the number of terms and an optional initial configuration.
        """
        self.n_terms = n_terms
        self.best_known_configuration = best_known_configuration
        self.bounds = self._define_bounds()  # Define bounds for each coefficient

    def _define_bounds(self):
         """
         Defines bounds for each coefficient to guide the optimization.
         The first coefficient is encouraged to be negative.
         The last coefficient is forced to be positive.
         Other coefficients have relatively wide bounds centered around zero.
         """
         bounds = [(-1, -0.0001)]  # c[0] is negative

         for _ in range(1, self.n_terms - 1):
             bounds.append((-0.5, 0.5))  # Other coefficients are close to zero

         bounds.append((0.0001, 1))  # Last coefficient is positive and small

         return bounds

    def __call__(self):
        """
        Generates K coefficients for Hermite polynomials and optimizes them to minimize the target function.
        """
        initial_guess = self._generate_initial_guess()

        # Optimization using minimize with constraints
        result = minimize(self._objective_function, initial_guess,
                        method='Nelder-Mead',  # Gradient-free method
                        bounds=self.bounds,
                        options={'maxiter': 1000, 'adaptive': True})  # Adjust maxiter as needed

        if result.success:
            return result.x.tolist()
        else:
            print("Warning: Optimization failed. Returning initial guess.")
            return initial_guess.tolist()

    def _generate_initial_guess(self):
        """
        Generates an initial guess for the coefficients, potentially using the best-known configuration if available.
        """
        if self.best_known_configuration is not None and len(self.best_known_configuration) == self.n_terms:
            return np.array(self.best_known_configuration)
        else:
            # Sample random coefficients within the defined bounds
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            return initial_guess


    def _objective_function(self, coefficients):
        """
        Defines the objective function to minimize: r_max^2 / (2*pi).
        Adds a penalty if P(0) is not negative.
        """
        r_max = self._find_r_max(coefficients)
        objective_value = r_max**2 / (2 * np.pi)

        # Add penalty if P(0) is not negative
        p_0 = self._p_x(0, coefficients)
        if p_0 >= 0:
            penalty = 1000 * p_0  # Large penalty
            objective_value += penalty

        # Add penalty if P(x) is negative for large x
        p_large = self._p_at_large_positive(coefficients)
        if p_large < 0:
            penalty = 1000 * abs(p_large)
            objective_value += penalty
        return objective_value

    def _p_x(self, x, coefficients):
        """
        Calculates P(x) = sum_{k=0..K-1} c[k] * H_{4k}(x).
        """
        p_x = 0
        for k, c in enumerate(coefficients):
            p_x += c * hermite(4 * k)(x)
        return p_x

    def _f_x(self, x, coefficients):
        """
        Calculates f(x) = P(x) * exp(-pi * x^2).
        """
        return self._p_x(x, coefficients) * np.exp(-np.pi * x**2)

    def _find_r_max(self, coefficients):
        """
        Finds r_max such that f(r_max) = max(f(x)) for x in [0, inf).
        """
        # Use a simple grid search to find r_max (can be replaced with a more efficient method)
        x_values = np.linspace(0, 5, 500)  # Search up to x=5
        f_values = self._f_x(x_values, coefficients)
        r_max_index = np.argmax(f_values)
        r_max = x_values[r_max_index]
        return r_max

    def _p_at_large_positive(self, coefficients):
        """
        Constraint: Enforces P(x) >= 0 for large |x| (e.g., x=10).
        This ensures that the polynomial doesn't become negative for large x.
        """
        x_large = 10  # Large x value
        return self._p_x(x_large, coefficients)
''', 'FourierCandidate',
'Optimizes Hermite polynomial coefficients using a gradient-free method with adaptive bounds and a penalty for violating P(0) < 0 to minimize the target function while satisfying constraints.')
    sol2.set_scores(8.8, 'Fitness is 8.8, try bigger mutations.')

    tt = TabooTable(10, minimisation=True)
    tt.update_taboo_search_table(sol1, sol2)
    print(tt.taboo_table[0].code_feature)
    assert tt.taboo_table[0].code_feature != ''

class DummyProblem(Problem):
    def evaluate(self, solution: Solution):
        fitness = random.random() * 80 + 9
        solution.set_scores(fitness, f'Fitness {fitness}, best known 92.')
        return solution

    def __call__(self, solution: Solution):
        return self.evaluate(solution)
    
    def test(self, solution):
        return self.evaluate(solution)
    
    def to_dict(self):
        return super().to_dict()

def test_log_best_solution():
    # Maximisation
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns')
    a = Solution()
    
    old = lhns.best_solution
    new = a

    ## log_best_solutions blindly updates to latest solution when both new and old solutions have |fitness| = ∞
    lhns._log_best_solution(a)
    assert lhns.best_solution == new
    assert lhns.best_solution.id != old

    b = Solution()
    b.set_scores(10, 'nada')

    old = new
    new = b

    ## log_best_solutions selects solution with any score, over ∞ score.
    lhns._log_best_solution(b)
    assert lhns.best_solution == new
    assert lhns.best_solution != old

    ## log_best_solution selects better solution even when current solution has |fitness| < ∞.
    c = Solution()
    c.set_scores(15, 'nada')

    old = b
    new = c

    lhns._log_best_solution(c)
    assert lhns.best_solution == new
    assert lhns.best_solution != old
    
    # Minimisation
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns', minimisation=True)
    a = Solution()
    
    old = lhns.best_solution
    new = a

    ## log_best_solutions blindly updates to latest solution when both new and old solutions have |fitness| = ∞
    lhns._log_best_solution(a)
    assert lhns.best_solution == new
    assert lhns.best_solution.id != old

    b = Solution()
    b.set_scores(10, 'nada')

    old = new
    new = b

    ## log_best_solutions selects solution with any score, over ∞ score.
    lhns._log_best_solution(b)
    assert lhns.best_solution == new
    assert lhns.best_solution != old

    ## log_best_solution selects better solution even when current solution has |fitness| < ∞.
    c = Solution()
    c.set_scores(5, 'nada')

    old = b
    new = c

    lhns._log_best_solution(c)
    assert lhns.best_solution == new
    assert lhns.best_solution != old
    
    # log_best_solution doesn't degenerate into wrong direction of optimality.
    d = Solution()
    d.set_scores(15, 'nada')

    old = new
    new = d

    lhns._log_best_solution(c)
    assert lhns.best_solution != new
    assert lhns.best_solution == old

def test_simulated_annealing_picks_optimal_solution():
    # Maximisation
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns')

    lhns.current_solution.set_scores(10)
    next_solution = Solution()
    next_solution.set_scores(15)

    old_id = lhns.current_solution.id
    new_id = next_solution.id

    lhns.simulated_annealing(next_solution, 1)

    assert new_id == lhns.current_solution.id
    assert old_id != lhns.current_solution.id
    
    # Minimisation
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns', minimisation=True)

    lhns.current_solution.set_scores(10)
    next_solution = Solution()
    next_solution.set_scores(5)

    old_id = lhns.current_solution.id
    new_id = next_solution.id

    lhns.simulated_annealing(next_solution, 1)

    assert new_id == lhns.current_solution.id
    assert old_id != lhns.current_solution.id

def test_simulated_annealing_accepts_with_probability(monkeypatch):
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'ts', cooling_rate=2)
    monkeypatch.setattr(lhns_module.random, "random", lambda: 0.1)
    lhns.current_solution.set_scores(100)

    next_solution = Solution()
    next_solution.set_scores(98)
    current_solution = lhns.current_solution

    iteration = 50
    temperature = lhns.alpha * iteration / lhns.budget
    delta = abs(next_solution.fitness - lhns.current_solution.fitness)
    p = math.e ** (-1 * delta / temperature)

    lhns.simulated_annealing(next_solution, iteration_number=iteration)
    if 0.1 <= p:
        assert lhns.current_solution != current_solution
    else:
        assert lhns.current_solution == next_solution
    
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'ts', minimisation=True, cooling_rate=2)
    monkeypatch.setattr(lhns_module.random, "random", lambda: 0.1)
    lhns.current_solution.set_scores(80)

    next_solution = Solution()
    next_solution.set_scores(82)
    current_solution = lhns.current_solution


    iteration = 50
    temperature = lhns.alpha * iteration / lhns.budget
    delta = abs(next_solution.fitness - lhns.current_solution.fitness)
    p = math.e ** (-1 * delta / temperature)

    lhns.simulated_annealing(next_solution, iteration_number=iteration)
    if 0.1 <= p:
        assert lhns.current_solution != current_solution
    else:
        assert lhns.current_solution == next_solution

def test_initialise_stops_execution_on_5x_failure(monkeypatch):
    def failure_of_a_sample_solution():
        raise TypeError

    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'ts')
    monkeypatch.setattr(Dummy_LLM, 'sample_solution', failure_of_a_sample_solution)

    with pytest.raises(Exception):
        lhns.initialise()

def test_initialise_returns_intial_operation(monkeypatch):
    code = """
def HelloWorld():
    print("Hello, there.")
"""
    def fake_sample_solution(self, messages):
        return Solution(code, "HelloWorld", "Greets you.")
    
    monkeypatch.setattr(Dummy_LLM, 
                        'sample_solution', 
                        fake_sample_solution
                        )

    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns')

    lhns.initialise()

    assert lhns.current_solution.code == code
    assert lhns.current_solution.name == "HelloWorld"
    assert lhns.current_solution.description == "Greets you."

def test_evaluate_returns_on_success_and_logs_best_solution(monkeypatch):

    def evaluate(solution: Solution):
        solution.set_scores(10.6, 'The algorithm scored 10.6, best known score is 9.8')
        return solution
    
    lhns = LHNS(DummyProblem(), Dummy_LLM(), 'vns')
    monkeypatch.setattr(lhns.problem, "evaluate", evaluate)

    solution = Solution()
    solution = lhns.evaluate(solution)

    assert solution.fitness == 10.6
    assert solution.feedback == 'The algorithm scored 10.6, best known score is 9.8'
    
    assert lhns.best_solution.fitness == 10.6
    assert lhns.best_solution.feedback == 'The algorithm scored 10.6, best known score is 9.8'

def test_evaluate_returns_unmutated_on_failure(monkeypatch):

    def evaluate(solution: Solution):
        return None
    
    lhns = LHNS(DummyProblem(), Dummy_LLM(), 'vns')
    monkeypatch.setattr(lhns.problem, "evaluate", evaluate)

    solution = Solution()
    new_solution = lhns.evaluate(solution)

    assert solution == new_solution

def test_extract_executable_lines():

    code = '''
def get_roots_of_quadratic_with_parameters(a: float, b: float, c: float) -> list[float]:
    """Given a, b, c for ax^2 + bx + c = 0, equation, returns the root of the solution:

    ## Args:
    `a: float`: Coefficient of x^2.
    `b: float`: Coefficient of x.
    `c: float`: Final Constant.

    ## Returns:
    `roots: [float]`: Returns a list of roots for the solution, may be of length 0, 1 or 2.
    """

    determinant = (b ** 2) - 4*a*c
    if determinant < 0:
        return []
    elif determinant == 0:
        return [-b/(2 * a)]
    else:
        return [(-b - (determinant) ** 0.5)/(2 * a), (-b + (determinant) ** 0.5)/(2 * a)]
'''
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns')

    data = lhns._extract_executable_lines_with_indices(code)

    expected_data = [(13, '    determinant = (b ** 2) - 4*a*c'),
                     (14, '    if determinant < 0:'),
                     (15, '        return []'),
                     (16, '    elif determinant == 0:'),
                     (17, '        return [-b/(2 * a)]'),
                     (18, '    else:'),
                     (19, '        return [(-b - (determinant) ** 0.5)/(2 * a), (-b + (determinant) ** 0.5)/(2 * a)]')]
    
    assert data == expected_data

def test_get_destroyed_code_only_destroys_executable_non_defs():
    code = '''
def get_roots_of_quadratic_with_parameters(a: float, b: float, c: float) -> list[float]:
    """Given a, b, c for ax^2 + bx + c = 0, equation, returns the root of the solution:

    ## Args:
    `a: float`: Coefficient of x^2.
    `b: float`: Coefficient of x.
    `c: float`: Final Constant.

    ## Returns:
    `roots: [float]`: Returns a list of roots for the solution, may be of length 0, 1 or 2.
    """

    determinant = (b ** 2) - 4*a*c
    if determinant < 0:
        return []
    elif determinant == 0:
        return [-b/(2 * a)]
    else:
        return [(-b - (determinant) ** 0.5)/(2 * a), (-b + (determinant) ** 0.5)/(2 * a)]
'''

    expected_output = '''
def get_roots_of_quadratic_with_parameters(a: float, b: float, c: float) -> list[float]:
    """Given a, b, c for ax^2 + bx + c = 0, equation, returns the root of the solution:

    ## Args:
    `a: float`: Coefficient of x^2.
    `b: float`: Coefficient of x.
    `c: float`: Final Constant.

    ## Returns:
    `roots: [float]`: Returns a list of roots for the solution, may be of length 0, 1 or 2.
    """

'''
    
    lhns = LHNS(BBOB_SBOX(), Dummy_LLM(), 'vns')
    soln = Solution(code=code)
    output = lhns.get_destroyed_code(1.0, soln)
    print("----------------------------Acutal Output----------------------------")
    for index, line in enumerate(output.split("\n")):
        print(index + 1, ":", line, "|")
    print("----------------------------Expected Output----------------------------")
    for index, line in enumerate(expected_output.split("\n")):
        print(index + 1, ":", line, "|")
    assert expected_output == output



def test_mutate_lhns_vns_calls_methods_properly(monkeypatch):
    
    destruction_ratio = []
    destroyed_code_arr = []
    destruction_prompt = []
    def fake_destroy_code(r, current):
        destruction_ratio.append(r)
        code = f'destroyed code r={r}'
        destroyed_code_arr.append(code)
        return code
    
    def fake_destruction_prompt(individual: Solution, destroyed_code: str, deleted_line_count: int) -> str:
        proompt = f'destruction prompt with code {destroyed_code}.'
        destruction_prompt.append(proompt)
        return proompt


    problem = DummyProblem()
    llm = Dummy_LLM()
    lhns_instance = LHNS(problem, llm, 'vns', budget=10)
    monkeypatch.setattr(lhns_instance, 'get_destroyed_code', fake_destroy_code)
    monkeypatch.setattr(lhns_instance.prompt_generator, 'get_prompt_destroy_repair', fake_destruction_prompt)
    _ = lhns_instance.run()
    for index in range(lhns_instance.budget):
        assert f'{destruction_ratio[index]:2}' == f'{0.1 * (1 + ((index + 1) % 10)):2}'
        assert destroyed_code_arr[index] == f'destroyed code r={0.1 * (1 + ((index + 1) % 10))}'
        assert destruction_prompt[index] == f'destruction prompt with code {destroyed_code_arr[index]}.'

def test_lhns_vns_raises_after_5x_tries(monkeypatch):
    counter = 0
    def fake_sample_solution(message):
        nonlocal counter
        counter += 1
        raise NoCodeException
    
    problem = DummyProblem()
    llm = Dummy_LLM()
    lhns_instance = LHNS(problem, llm, 'vns', budget=10)
    monkeypatch.setattr(lhns_instance.llm, 'sample_solution', fake_sample_solution)

    with pytest.raises(NoCodeException):
        lhns_instance.mutate_lhns_vns(4)
    assert counter == 5

def test_mutate_lhns_ils_calls_methods_properly(monkeypatch):
    
    destruction_ratio = []
    destroyed_code_arr = []
    destruction_prompt = []
    def fake_destroy_code(r, current):
        destruction_ratio.append(r)
        code = f'destroyed code r={r}'
        destroyed_code_arr.append(code)
        return code
    
    def fake_destruction_prompt(individual: Solution, destroyed_code: str, deleted_line_count: int) -> str:
        proompt = f'destruction prompt with code {destroyed_code}.'
        destruction_prompt.append(proompt)
        return proompt
    

    def fake_initialisation_prompt():
        destruction_ratio.append(None)
        destroyed_code_arr.append(None)
        destruction_prompt.append('hi')
        return 'hi'


    problem = DummyProblem()
    llm = Dummy_LLM()
    lhns_instance = LHNS(problem, llm, 'ils', budget=10)
    monkeypatch.setattr(lhns_instance, 'get_destroyed_code', fake_destroy_code)
    monkeypatch.setattr(lhns_instance.prompt_generator, 'get_prompt_destroy_repair', fake_destruction_prompt)
    monkeypatch.setattr(lhns_instance.prompt_generator, 'get_prompt_i1', fake_initialisation_prompt)
    _ = lhns_instance.run()

    for index in range(lhns_instance.budget):
        print(f"{index}| {destruction_ratio[index]} | {destruction_prompt[index]} | {destroyed_code_arr[index]} | {destruction_ratio[index]}")
        if destruction_ratio[index] is not None:
            assert destroyed_code_arr[index] == 'destroyed code r=0.5'
            assert destruction_prompt[index] == 'destruction prompt with code destroyed code r=0.5.'
        else:
            assert destroyed_code_arr[index] is None
            assert destruction_prompt[index] == 'hi'


def test_lhns_ils_raises_after_5x_tries(monkeypatch):
    counter = 0
    def fake_sample_solution(message):
        nonlocal counter
        counter += 1
        raise NoCodeException
    
    problem = DummyProblem()
    llm = Dummy_LLM()
    lhns_instance = LHNS(problem, llm, 'ils', budget=10)
    monkeypatch.setattr(lhns_instance.llm, 'sample_solution', fake_sample_solution)

    with pytest.raises(NoCodeException):
        lhns_instance.mutate_lhns_ils(4)
    assert counter == 5

def test_taboo_search_calls_functions_appripriately(monkeypatch):
    destruction_ratio = []
    destroyed_code_arr = []  # No padding.
    taboo_elements = []      # padding.
    prompt = []              # No padding.

    def fake_destroy_code(r, current):
        destruction_ratio.append(r)
        code = f'destroyed code r={r}'
        return code

    def fake_destruction_prompt(individual: Solution, destroyed_code: str, deleted_line_count: int) -> str:
        proompt = f'destruction prompt with code {destroyed_code}.'
        destroyed_code_arr.append(proompt)
        taboo_elements.append(None)
        prompt.append(proompt)
        return proompt

    def fake_taboo_prompt(individual: Solution, destroyed_code: str, taboo_element) -> str:
        proompt = f'taboo prompt with code {destroyed_code}. Taboo Element {taboo_element}'
        destroyed_code_arr.append(destroyed_code)
        taboo_elements.append(taboo_element)
        prompt.append(proompt)
        return proompt

    def fake_get_distinct_entry(for_individual):
        taboo_element = TabooElement('taboo desc', 'taboo_element', random.random(), ['feature.'])
        return taboo_element

    def fake_initialisation_prompt():
        destruction_ratio.append(None)
        destroyed_code_arr.append(None)
        taboo_elements.append(None)
        prompt.append('Init')
        return 'Init'

    problem = DummyProblem()
    llm = Dummy_LLM()
    lhns_instance = LHNS(problem, llm, 'ts', budget=10)

    monkeypatch.setattr(lhns_instance.prompt_generator, 'get_prompt_i1', fake_initialisation_prompt)
    monkeypatch.setattr(lhns_instance, 'get_destroyed_code', fake_destroy_code)
    monkeypatch.setattr(lhns_instance.taboo_table, 'get_distinct_entry', fake_get_distinct_entry)
    monkeypatch.setattr(lhns_instance.prompt_generator, 'get_prompt_taboo_search', fake_taboo_prompt)
    monkeypatch.setattr(lhns_instance.prompt_generator, 'get_prompt_destroy_repair', fake_destruction_prompt)

    _ = lhns_instance.run()

    print(f"Length of destruction ratio={len(destruction_ratio)}, destroyed code={len(destroyed_code_arr)}, taboo elements={len(taboo_elements)}, prompt={len(prompt)}")
    
    for index in range(lhns_instance.budget):
        print(f"{index}| {destruction_ratio[index]} | {prompt[index]} | {destroyed_code_arr[index]} | {taboo_elements[index]}")
        if destruction_ratio[index] is not None:
            assert destroyed_code_arr[index] == 'destruction prompt with code destroyed code r=0.5.' if index % 10 != 9 else 'taboo prompt with code destroyed code r=0.5.'
        else:
            # Initialisation step
            assert destroyed_code_arr[index] is None
            assert prompt[index] == 'Init'


def test_lhns_ts_raises_after_5x_tries(monkeypatch):
    counter = 0
    def fake_sample_solution(message):
        nonlocal counter
        counter += 1
        raise NoCodeException
    
    problem = DummyProblem()
    llm = Dummy_LLM()
    lhns_instance = LHNS(problem, llm, 'ts', budget=10)
    monkeypatch.setattr(lhns_instance.llm, 'sample_solution', fake_sample_solution)

    with pytest.raises(NoCodeException):
        lhns_instance.mutate_lhns_ts(4)
    
    assert counter == 5
    
    counter = 0
    with pytest.raises(NoCodeException):
        lhns_instance.mutate_lhns_ts(9)
    assert counter == 5
