import random

from iohblade.problems.bbob_sboxcost import BBOB_SBOX
from iohblade.solution import Solution
from iohblade.llm import Dummy_LLM

from iohblade.methods.lhns.lhns import LHNS
from iohblade.methods.lhns.taboo_table import TabooTable

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
    lhns = LHNS(problem, dllm, 0.2)

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