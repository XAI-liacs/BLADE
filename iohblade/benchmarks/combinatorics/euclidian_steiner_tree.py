import math
from pathlib import Path
from statistics import mean
from dataclasses import dataclass

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace

@dataclass
class Location:
    id: int
    x: float
    y: float

    def vectorise(self) -> list[float]:
        return [self.x, self.y]

    def __repr__(self) -> str:
        return f'Point(id: {self.id}, x: {self.x}, y: {self.y})'

    def distance_to(self, other: 'Location'):
        return math.hypot(self.x - other.x, self.y - other.y)

class EuclidianSteinerTree(Problem):

    def __init__(self, benchmark_id: int, tolerance=1e-6):
        """
        ## Euclidian Steiner Tree Benchmark:
            Implements a Eucldian Steiner Tree Algorithm, which optimises the mimimum spanning tree, but with extra points.
            Adding these points allows for shorts MST connecting each of the nodes. This benchmarks takes the set of points,
            runs mst on it and on points + steiner_points, and return their ratio. Optimisation goal: min mst(steiner_points + points)/mst(points).

        ## Args:
        `benchmark_id: int` A benchmark id, selects a benchmark from the available instances in [20, 30, 40, 50, 60, 70, 80, 90, 250, 500, 1000]
        `tolerance: float (10^-6)`: A tolerance to limit how close two points can be, stops algorithms from generating optimisers that may generate float-overflow.
        """
        self.benchmark = Path(__file__).resolve().parent.joinpath(f'estein{benchmark_id}.txt')
        self.points = {}
        self.tolerance = tolerance
        self._read_file()
        Problem.__init__(self, name=f'EucildianSteinerTree-n{20}')
        self.best_so_far = [float('nan')] * len(self.points)
        self.minimisation = True

        self.task_prompt = f"""
Write a python class with function `__call__`, that generates optimal Steiner Points for a given set of point, in order to optimise their Minimum Spanning Tree distance.
- The class must initialise with 2 positional parameter:
    1. `points: list[list[float][:2]]`: A list of locations, representing 2-D coordinates of nodes that needs to be connected using Minimum Spanning Tree using Steiner Points.
    2. `tolerance: float (10^-6)`: A tolerance parameter that expects Steiner MST distance to be at a minimum `tolerance` units less than Normal MST.
- The `__call__` method must return:
    - `steiner_points : list[list[float][:2]]`: A list of steiner points.
        - Each steiner point must be 2-D vector.
- The optimisation goal is to minimise `steiner_mst/normal_mst` ratio where:
    - `steiner_mst` is minimum spanning tree found using the provided points + returned `steiner_points`
    - `normal_mst` is a mimumum spanning tree found using the only the provided points.
- The returned fitness is going to be average fitness across {len(self.points)} distinct benchmarks.
    """
        self.example_prompt = """
An example response can be
---
# Descripition:
A novel algorithm for finding steiner points to optimise MST.
# Code:
```python
import random

class SteinerPointGenerator:
    def __init__(self, points: list[list[float]], tolerance: float):
        self.points = points[:]
        self.steiner_points = []
        self.tolerance = tolerance

    def __call__(self):
        # Find steiner points.
        return self.steiner_points
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
        with open(self.benchmark) as f:
            data = f.readlines()
            count = int(data.pop(0))
            for i in range(count):
                points_count = int(data.pop(0))
                points = []
                for j in range(points_count):
                    datum = data.pop(0).split()
                    location = Location(j, float(datum[0]), float(datum[1]))
                    points.append(location)
                self.points[i] = points
        return

    def compute_mst_length(self, points: list[Location]):
        n = len(points)
        if n == 0:
            return 0.0
        in_mst = [False] * n
        min_dist = [float('inf')] * n
        min_dist[0] = 0.0
        total = 0.0
        for _ in range(n):
            u = -1
            best = float('inf')
            for i in range(n):
                if not in_mst[i] and min_dist[i] < best:
                    best = min_dist[i]
                    u = i
            if u == -1:
                break
            in_mst[u] = True
            total += best
            for v in range(n):
                if not in_mst[v]:
                    d = points[u].distance_to(points[v])
                    if d < min_dist[v]:
                        min_dist[v] = d
        return total

    def evaluate(self, solution: Solution, logger = None):
        code = solution.code
        name = solution.name if solution.name else 'EuclidianSteinerSolver'
        try:
            ns = prepare_namespace(code, self.dependencies)
            compiled_code = compile(code, name, 'exec')
            exec(compiled_code, ns, ns)
            cls = ns[name]
            fitness = {}
            for bench_id, points in self.points.items():
                steiner_pts = cls([point.vectorise() for point in points], self.tolerance)()
                steiner_points = []
                for index, steiner_point in enumerate(steiner_pts):
                    steiner_points.append(Location(len(self.points) + index + 1, steiner_point[0], steiner_point[1]))
                normal_mst = self.compute_mst_length(points)
                steiner_mst = self.compute_mst_length(points + steiner_points)

                assert steiner_mst + self.tolerance >= normal_mst, f"Steiner's MST ({steiner_mst}) was close to or geater thn normal MST ({normal_mst})"
                fitness[bench_id] = normal_mst / steiner_mst

            solution = solution.set_scores(
                mean(fitness.values()),
                f'Got fitness {fitness} across {len(self.points)} benchmarks.'
            )
        except Exception as e:
            solution = solution.set_scores(
                float('inf'),
                f'Got error {e}',
                e
            )
        return solution


    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

if __name__ == '__main__':
    est = EuclidianSteinerTree(20)
    print(est.get_prompt())
