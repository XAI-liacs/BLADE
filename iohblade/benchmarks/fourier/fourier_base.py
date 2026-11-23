import textwrap


class FourierBase:
    """
    Base spec for the uncertainty inequality (Appendix B.4).
    Candidates return K = n_terms coefficients c[0..K-1]; the evaluator
    appends c[K] so that P(0) = 0, giving the even polynomial

        P(x) = sum_{k=0..K} c[k] * H_{4k}(x)

    and the test function is f(x) = P(x) * exp(-pi * x^2).
    The evaluator computes an upper bound on C4 as r_max^2 / (2*pi),
    where r_max is the largest positive sign change of P(x) / x^2.

    Notes:
      - H_n are physicists' Hermite polynomials (unscaled).
      - c[K] = -sum_k c[k] H_{4k}(0) / H_{4K}(0); for K = 3,
        c[3] = -(c0 + 12 c1 + 1680 c2) / 665280.
      - If the leading coefficient of P is negative, P is flipped.
      - Scale invariance: multiply c by s != 0 ⇒ same score.
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
        return textwrap.dedent(
            """
- Write a class whose __call__() yields a list of floats c of length K (= n_terms).
- The evaluator appends one more coefficient c[K] = -(sum_k c[k] * H_{4k}(0)) / H_{4K}(0),
  chosen so that P(0) = 0 (for K = 3: c[3] = -(c[0] + 12*c[1] + 1680*c[2]) / 665280).
- Define P(x) = sum_{k=0..K} c[k] * H_{4k}(x) using physicists' Hermite H_n (unscaled),
  i.e. your K coefficients are for H_0, H_4, ..., H_{4(K-1)} and the evaluator adds H_{4K}.
- Test function: f(x) = P(x) * exp(-pi * x^2). Evenness holds by construction.
- Scoring:
    - if the leading coefficient c[K] is negative, P is replaced by -P;
    - r_max is the largest positive x at which P(x) / x^2 changes sign
      (a candidate without such a sign change is invalid);
    - scaling c by any nonzero factor leaves the score unchanged.
- Objective (minimize):
    - """
            + formula
            + textwrap.dedent(f"""
- Tip: keep the coefficients small and structured to control where the roots of P(x) / x^2 lie.
K = {self.n_terms}.
""")
        )

    def make_example_prompt(self, class_name: str) -> str:
        initialiser = """
    def __init__(self, n_terms: int):
        # Accepts number of terms K for the problem.
        self.n_terms = n_terms
"""
        if self.best_known_configuration is not None:
            initialiser = """
    def __init__(self, n_terms: int, best_known_configuration=None):
        # Accepts a number of terms K and best known configuration (if
        # available) for the problem, as an initial configuration, which
        # can then be optimised further.
        self.n_terms = n_terms
        self.best_known_configuration = best_known_configuration
"""
        return f"""
Here is an example template of program to solve the problem:
```python
class {class_name}:{initialiser}
    def __call__(self):
        # Return K={self.n_terms} coefficients for H_0, H_4, H_8, ...; the
        # evaluator appends the H_{{4K}} coefficient that makes P(0) = 0.
        # Simple baseline: c = [-1, 0, ..., 0, 1e-4].
        c = [0.0] * self.n_terms
        c[0] = -1.0
        c[-1] = 1e-4
        return c
```
"""

    def make_format_prompt(self) -> str:
        return textwrap.dedent("""
Give an excellent and novel algorithm to solve this task and also give it a
one-line description, describing the main idea. Give the response in the format:

# Description: <Short one line description of the program.>
# Code:
```python
<your class here>
```
""")
