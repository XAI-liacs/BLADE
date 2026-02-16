from typing import Optional


class PackingBase:
    """Base class for circle packing optimization problems (geometry/packing).

    Contract for candidates:
      - Return a NumPy array of shape (n, 3), each row [x, y, r].
      - Coordinates are in the ambient box of the problem.
      - Radii must be positive.
    """

    def __init__(self, name: str, best_solution):
        self.task_name = name
        self.best_solution = best_solution

    ## Prompt helpers:
    def make_task_prompt(self, headline: str, contract: str, objective: str) -> str:
        return f"""
- Write a Python class with `_call_` method that returns a numpy array of shape (n, 3) with rows [x, y, r], denoting a set of n disjoint circles, for solving the problem of:
   - {headline}
- The constranits on the circles are:
    - Circles must be pairwise disjoint.
    - Circles must lie fully inside the specified region.
    - Radii must be strictly positive.
- Objective: {objective}
- Output contract:
    - {contract}"""

    def make_hexagon_task_prompt(self, tolerance) -> str:
        return f"""
- Write a Python class with a __call__ method that generates and returns a NumPy array of shape (n, 3). Each row of the array represents one inner hexagon that is fitted inside a larger outer hexagon, with the following values:
    - x: the x-coordinate of the hexagon’s center.
    - y: the y-coordinate of the hexagon’s center.
    - theta: the rotation angle of the hexagon (in radians)
- The constranits on the hexagons are:
    - No two hexagons must overlap.
    - Hexagon must lie fully inside outer regular hexagon, with side s, and theta 0.
    - Each hexagons are assumed to be regular, with side 1.
    - The tolerance for evaluation in given by {tolerance}.
- Objective is to minimise s; the side of outer hexagon.
"""

    def make_hexagon_example_prompt(self, class_name: str, n_hexagon: int) -> str:
        best_known_initialiser = f"""
    def __init__(self, n_hexagons: int = {n_hexagon}):
        self.n_hexagons = int(n_hexagons)
"""
        if self.best_solution is not None:
            best_known_initialiser = f"""
    def __init__(self, n_hexagons: int={n_hexagon}, best_known_configuration: list[float] | None = None):
        self.best_known_configuration = best_known_configuration
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        # optimised for better results.
        self.n_hexagons = int(n_hexagons)
"""
        return f"""
```python
class {class_name}:
    {best_known_initialiser}
    def __call__(self):
        return [(0, ) * 3] * {n_hexagon}      # {n_hexagon} disjoint hexagons, with (x, y, theta).
```
"""

    def make_example_prompt(
        self,
        class_name: str,
        n_circles: int,
        body_hint: Optional[str] = None,
    ) -> str:
        hint = (
            body_hint
            or """import numpy as np

rng = np.random.default_rng(0)
n = getattr(self, 'n_circles', 8)

# naive jittered grid with small equal radii that surely fit
g = int(np.ceil(np.sqrt(n)))
r = 0.5/(g+1)
pts = []
for i in range(n):
    row, col = divmod(i, g)
    x = (col+1)/(g+1)
    y = (row+1)/(g+1)
    x += (rng.random()-0.5)*r*0.2
    y += (rng.random()-0.5)*r*0.2
    pts.append([x, y, r])
return np.array(pts, dtype=float)
"""
        )

        best_known_initialiser = f"""
    def __init__(self, n_circles: int = {n_circles}):
        self.n_circles = int(n_circles)
"""
        if self.best_solution is not None:
            best_known_initialiser = f"""
    def __init__(self, n_circles: int={n_circles}, best_known_configuration: list[float] | None):
        self.best_known_configuration = best_known_configuration
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        # optimised for better results.
        self.n_circles = int(n_circles)
"""
        stringified_hint = "\n\t".join(hint.split("\n"))
        return f"""

```python
class {class_name}:
    {best_known_initialiser}

    def __call__(self):
        {stringified_hint}
```
"""

    def make_format_prompt(self) -> str:
        return """

Give an excellent and novel algorithm to solve this task and also give it a
one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```

"""
