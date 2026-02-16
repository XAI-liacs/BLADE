import math
from pathlib import Path
from dataclasses import dataclass

from iohblade.solution import Solution
from iohblade.problem import Problem
from iohblade.misc.prepare_namespace import prepare_namespace


@dataclass
class Container:
    x: float
    y: float
    radius: float

    def contains_circle(self, x: float, y: float, radius: float, tolerance):
        dist = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        dist -= tolerance
        if dist <= self.radius - radius:
            return True
        return False

    def vectorise(self):
        return [self.x, self.y, self.radius]

    def __repr__(self) -> str:
        return f"Container(x: {self.x}, y: {self.y}, r={self.radius})"


class CirclePacking(Problem):
    def __init__(self, tolerance=1e-12):
        self.container = Container(0, 0, 0)
        self.candidate_radii: list[float] = []
        self.n_circles = 0
        self.tolerance = tolerance
        self._read_file()
        Problem.__init__(self, name="circle_packing_n40")
        self.minimisation = False
        ## Prompts;
        self.task_prompt = f"""
Write a python class with function `__call__`, that generate a solution for Circle Packing Area Problem.
- The class must initialise with 3 positional parameters:
    1. `container : list[float][:3]`: A list of float representing a container as [x, y, r].
    2. `candidate_radii : list[float][:{self.n_circles}]`: A list of possible radius to be placed inside the container.
    3. `tolerance: float`: A tolerance factor, for all calculations. The circles should be further than this value for feasibility.
- The `__call__` method must return:
    - `circles : list[list[float][:3]]`: A list of circles, that are packed inside the `container`.
        - Each circle is represented as [x, y, r].
        - Each circle's radius `r` must be in `candidate_radii`.
        - Each circle must be container fully in the container.
        - No two circle must intersect, witin the tolerance.
- The optimisation goal is to maximize the area of the container filled by circles. Given by:
    \\[\\omega = \\max \\sum_i^{self.n_circles} \\alpha_i \\pi R_i^2\\]
"""

        self.example_prompt = f"""
An example response can be
---
# Descripition: 
A random selection algorithm for Circle Packing Area Problem.
# Code:
```python
import random
import math

class CirclePackingSolver:
    def __init__(self, container, candidate_radii, tolerance):
        self.container = container
        self.candidate_radii = candidate_radii

    def __call__(self):
    cs=[]
    for r in self.candidate_radii:
        for _ in range(100):
            x=random.random()*self.container
            y=random.random()*self.container
            if all((x-x2)**2+(y-y2)**2>=(r+r2)**2 for x2,y2,r2 in cs):
                cs.append((x,y,r))
                break
    return cs
```
"""
        self.format_prompt = """
Give an excellent and novel algorithm to solve this task and also give it a
one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

    def _read_file(self):
        path = Path(__file__).resolve().parent.joinpath("circles.bench")
        with open(path) as f:
            for line in f.readlines():
                data = list(map(float, line.strip().split()))
                if len(data) == 4:
                    n, cx, cy, r = data
                    self.n_circles = int(n)
                    self.container = Container(cx, cy, r)
                else:
                    self.candidate_radii.append(data[0])

    def _check_validity(self, circles: list[list[float]]):
        for circle in circles:
            assert (
                len(circle) == 3
            ), f"Expected each circle to be represented by [x, y, r], got {circle}."
            assert (
                circle[2] in self.candidate_radii
            ), f"Unknown candidate with radius {circle[2]}"
            assert self.container.contains_circle(
                circle[0], circle[1], circle[2], self.tolerance
            ), f"Circle at ({circle[0]}, {circle[1]}) with radius {circle[2]} is not contained in {self.container}."

        radii = list(map(lambda x: x[2], circles))
        assert len(radii) == len(
            set(radii)
        ), f"Unexpected use of more than one circle with same radius."

        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                dist = (
                    (circles[i][0] - circles[j][0]) ** 2
                    + (circles[i][1] - circles[j][1]) ** 2
                ) ** 0.5
                assert (
                    dist - circles[i][2] - circles[j][2] >= self.tolerance
                ), f"Circles (x = {circles[i][0]}, y = {circles[i][1]}, r = {circles[i][2]}) and (x = {circles[j][0]}, y = {circles[j][1]}, r = {circles[j][2]}) overlap."

    def _calc_area(self, circles):
        return sum(list(map(lambda x: math.pi * x[2] ** 2, circles)))

    def evaluate(self, solution: Solution, logger=None):
        name = solution.name if solution.name else "CirclePacking"
        code = solution.code

        try:
            local_ns = {}
            global_ns = prepare_namespace(code, self.dependencies)
            compiled_code = compile(code, name, "exec")
            exec(compiled_code, global_ns, local_ns)
            cls = local_ns[name]
            circles = cls(
                self.container.vectorise(), self.candidate_radii, self.tolerance
            )()
            self._check_validity(circles)
            area = self._calc_area(circles)
            solution = solution.set_scores(area, f"Got sum of area = {area}")
        except Exception as e:
            solution = solution.set_scores(float("-inf"), f"Got error {e}", e)
        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    cp = CirclePacking()
    print(cp.get_prompt())
