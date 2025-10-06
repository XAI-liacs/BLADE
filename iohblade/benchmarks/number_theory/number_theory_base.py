class NumberTheoryBase:
    """
    Base spec for number-theory benchmarks.

    This spec is written for the “sums vs differences” task as implemented in the
    paper via a *single-set* U formulation (Appendix B.6, eq. (3)):
        c(U) = 1 + log(|U−U| / |U+U|) / log(2*max(U) + 1),
    which lower-bounds C6.

    Candidates must implement a class whose __call__() returns a finite U ⊂ ℤ≥0
    with 0 ∈ U and |U| ≤ max_size.
    """

    def __init__(self, name, best_soluton: list[int] | None = None) -> None:
        self.task_name = name
        self.best_solution = best_soluton

    def make_task_prompt(self, formula: str) -> str:
        # Single-set U specification; matches Appendix B.6, eq. (3).
        return f"""
- Write a Python class, with `__call__` method that CONSTRUCTS a finite integer set U.
- Constraints:
    - U is non negative integer.
    - 0 is in U
    - |U| <= max_size passed to the class constructor.
- Objective:
    - Maximize {formula}
        - where |U±U| are the cardinalities of the sum/difference sets and max(U) is the largest element of U.
        - The evaluator computes:
            c(U) = 1 + log(|U−U| / |U+U|) / log(2*max(U) + 1)
            - which lower-bounds C6 per Gyarmati–Hennecart–Ruzsa.
- Output contract:
    - The `__call__(self) -> iterable[int]` must yield U (list or set is fine).
-   Notes/tips for search:
    - Favor constructions with large |U−U| and small |U+U| while keeping max(U) modest.
    - Ensure 0 is included.
    - Keep integers non-negative.
    - Respect max_size.
    - Useful families to try:
        - greedy/Sidon-like sets,
        - Golomb-ruler style, block + gaps,
        - digital/bit-pattern constructions, unions of progressions, randomized hill-climbing.
"""

    def make_example_prompt(self, class_name: str) -> str:
        # Minimal, valid example that returns a simple U.
        best_known_initialiser = """
    def __init__(self, max_size=100):
        self.max_size = max_size
        """
        if self.best_solution is not None:
            best_known_initialiser = """
    def __init__(self, max_size=100, best_known_configuration: list[float] | None = None):
        self.max_size = max_size
        self.best_known_configuration = best_known_configuration
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        # optimised for better results.
"""
        return f"""
An example template of the program is:
```python
class {class_name}:
    {best_known_initialiser}

    def __call__(self):
        # Must return U ⊂ ℤ≥0 with 0 ∈ U and |U| ≤ self.max_size.
        # Simple constructive baseline: start with 0, then add increasing gaps.
        import random
        U = [0]
        gap = 1
        while len(U) < self.max_size:
            U.append(U[-1] + gap)
            gap = gap * 2 if random.random() < 0.3 else gap + 1
        return U
```
"""

    def make_format_prompt(self) -> str:
        return """
Give an excellent and novel algorithm to solve this task and also give it a
one-line description, describing the main idea. Give the response in the format:

# Description: <Short one line description of the program.>
# Code:
```python
<your class here>
```
"""
