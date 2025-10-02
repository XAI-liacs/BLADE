class FourierBase:
    """
    Base spec for the uncertainty inequality (Appendix B.4).
    Candidates return coefficients c[0..K-1] for an even polynomial

        P(x) = sum_{k=0..K-1} c[k] * H_{4k}(x)

    and the test function is f(x) = P(x) * exp(-pi * x^2).
    The evaluator computes an upper bound on C4 as r_max^2 / (2*pi),
    where r_max is the largest positive root beyond which P(x) >= 0.

    Notes:
      - H_n are physicists' Hermite polynomials.
      - Constraints: P(0) < 0; leading coefficient > 0; P(x) >= 0 for large |x|.
      - Scale invariance: multiply c by s>0 ⇒ same score.
    """

    def __init__(
        self,
        task_name: str,
        n_terms: int = 3,
        tolerance: float = 1e-9,
        x_max: float = 12.0,
        grid_step: float = 1e-2,
        check_points: int = 800,
        best_known_configuration: list[float] | None = None,
    ):
        self.task_name = task_name
        self.n_terms = n_terms  # H0, H4, H8 by default
        self.tolerance = tolerance
        self.x_max = x_max  # search window for roots/positivity
        self.grid_step = grid_step  # coarse scan step for root bracketing
        self.check_points = check_points  # positivity checks beyond r_max
        self.best_known_configuration = (
            best_known_configuration  # Best known configuation.
        )

        if self.n_terms < 1:
            raise ValueError("n_terms must be >= 1")

    def problem_description(self) -> str:
        return f"{self.task_name} | Hermite degrees 0..{4*(self.n_terms-1)} step 4"

    def make_task_prompt(self, formula: str) -> str:
        return (
            """
- Write a class whose __call__() yields a list of floats c of length K (= n_terms).
- Define P(x) = sum_{k=0..K-1} c[k] * H_{4k}(x) using physicists' Hermite H_n.
- Test function: f(x) = P(x) * exp(-pi * x^2). Evenness holds by construction.
- Constraints:
    - P(0) < 0
    - the leading coefficient c[K-1] > 0
    - P(x) >= 0 for large |x|.
- Objective (minimize):
    - """
            + formula
            + """
- Tip: enforce structure (e.g., small |c|, P(0)≈0) to aid root placement.
K = {self.n_terms}."
"""
        )

    def make_example_prompt(self, class_name: str) -> str:
        accept_best_configuration = ""
        if self.best_known_configuration is not None:
            accept_best_configuration = """
    def __init__(self, best_known_configuration: list[float] | None):
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        optimised for better results.
"""
        return f"""
Here is an example template of program to solve the problem:
```python
class {class_name}:
    {accept_best_configuration}
    def __call__(self):
        # Return K={self.n_terms} coefficients for H_0, H_4, H_8, ...
        return [0.33, -0.01, -9e-05][: {self.n_terms}]
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
