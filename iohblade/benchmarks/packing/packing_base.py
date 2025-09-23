from typing import Optional

class PackingBase:
    """Base class for circle packing optimization problems (geometry/packing).

    Contract for candidates:
      - Return a NumPy array of shape (n, 3), each row [x, y, r].
      - Coordinates are in the ambient box of the problem.
      - Radii must be positive.
    """
    def __init__(self, name):
        self.task_name = name

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

    def make_example_prompt(self, class_name: str, body_hint: Optional[str] = None) -> str:
        hint = body_hint or """import numpy as np

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
        return f"""

```python
class {class_name}:
    def __init__(self, n_circles: int = 8):
        self.n_circles = int(n_circles)
    def __call__(self):
        {"\n\t".join(hint.split("\n"))}
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
