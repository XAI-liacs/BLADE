import textwrap
from typing import Any, Optional

from iohblade.tags import NoiseType, ObjectiveType, PrimaryCategories


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

        self.tags: list[Any] = [
            PrimaryCategories.CO,
            NoiseType.NOISELESS,
            ObjectiveType.SINGLE_OBJECTIVE,
        ]

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
    - theta: the rotation angle of the hexagon (in radians); its vertices are at (x + cos(theta + k*pi/3), y + sin(theta + k*pi/3)) for k = 0..5.
- The constranits on the hexagons are:
    - No two hexagons must overlap.
    - Hexagons must lie fully inside an outer regular hexagon with side s and a fixed orientation: its vertices are at angles pi/6 + k*pi/3 (30, 90, 150, ... degrees) from its centre, i.e. it has two vertical edges and is rotated by 30 degrees relative to an inner hexagon with theta 0.
    - The evaluator places the outer hexagon's centre optimally, so only the relative positions of the inner hexagons matter.
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
    def __init__(self, n_hexagons: int = {n_hexagon}, best_known_configuration: list[list[float]] | None = None):
        # Accepts a best known configuration (if available) for the problem, as an initial configuration, which can then
        # be optimised for better results.
        self.best_known_configuration = best_known_configuration
        self.n_hexagons = int(n_hexagons)
"""
            return f"""
# Description: Start from the best known packing.
# Code:
```python
import numpy as np


class {class_name}:
    {best_known_initialiser.strip()}

    def __call__(self):
        # Baseline: return the best known configuration unchanged.
        return np.array(self.best_known_configuration, dtype=float)
```
"""
        return f"""
# Description: Place the hexagons on a honeycomb grid, which tiles the plane without overlaps.
# Code:
```python
import math

import numpy as np


class {class_name}:
    {best_known_initialiser.strip()}

    def __call__(self):
        n = self.n_hexagons
        cols = int(math.ceil(math.sqrt(n)))
        hexagons = []
        for i in range(n):
            row, col = divmod(i, cols)
            # Flat-topped unit hexagons (theta = 0) tile the plane with a
            # horizontal spacing of 1.5 and a vertical spacing of sqrt(3);
            # every other column is shifted up by half a row.
            x = 1.5 * col
            y = math.sqrt(3) * (row + 0.5 * (col % 2))
            hexagons.append([x, y, 0.0])
        return np.array(hexagons, dtype=float)  # shape (n, 3): rows (x, y, theta)
```
"""

    def make_example_prompt(
        self,
        class_name: str,
        n_circles: int,
        body_hint: Optional[str] = None,
        best_body_hint: Optional[str] = None,
    ) -> str:
        hint = body_hint or """import numpy as np

n = getattr(self, 'n_circles', 8)

# naive square grid of equal circles, each inscribed in its own grid cell
g = int(np.ceil(np.sqrt(n)))
r = 0.5 / g
pts = []
for i in range(n):
    row, col = divmod(i, g)
    x = (col + 0.5) / g
    y = (row + 0.5) / g
    pts.append([x, y, r])
return np.array(pts, dtype=float)
"""

        best_known_initialiser = f"""
    def __init__(self, n_circles: int = {n_circles}):
        self.n_circles = int(n_circles)
"""
        if self.best_solution is not None:
            best_known_initialiser = f"""
    def __init__(self, n_circles: int = {n_circles}, best_known_configuration: list[list[float]] | None = None):
        # Accepts a best known configuration (if available) for the problem, as an initial configuration, which can then
        # be optimised for better results.
        self.best_known_configuration = best_known_configuration
        self.n_circles = int(n_circles)
"""
            hint = best_body_hint or """import numpy as np

# Baseline: return the best known configuration unchanged.
return np.array(self.best_known_configuration, dtype=float)
"""
        body = textwrap.indent(textwrap.dedent(hint).strip(), " " * 8)
        return f"""

```python
class {class_name}:
    {best_known_initialiser.strip()}

    def __call__(self):
{body}
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
