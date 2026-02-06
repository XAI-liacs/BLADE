import json
import math
from dataclasses import dataclass

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace


@dataclass
class Location:
    id: int
    x: float
    y: float

    def vectorise(self):
        return [self.id, self.x, self.y]
    
    def distance_to(self, other: 'Location'):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class TravelingSalesmanProblem(Problem):
    def __init__(self):
        self.customers : list[Location] = []
        self.best_known = float('inf')
        self.benchmark = ''
        self.minimisation = True
        self._readfile()

        Problem.__init__(self, name=f'TSP-{self.benchmark}')

        self.task_prompt = """
Write a python class with function `__call__`, that generate a solution for Traveling Salesman Problem.
- The class must initialise with 1 positional parameter:
    1. `locations`: A list of locations, representing customers, that a traveling salesman must visit.
- `Note`: Each of the `locations` are represented by list corresonding [id, x-coordinate, y-coordinate],
- The `__call__` method must return:
    - `paths : list[int]`: A list of customer `id`s, representing the path.
        - Each customer must only be served once.
        - No customer must be left un-served.
- The optimisation goal is to minimise total distance travelled by the salesman."""
    
        self.example_prompt = f"""
An example response can be
---
# Descripition: 
A random selection algorithm for Capacited Vehicle Routing Problem.
# Code:
```python
import random

class TravelingSalesmanSolver:
    def __init__(self, customers):
        self.customers = customers

    def __call__(self):
        ids = list(map(lambda customer: customer[0], self.customers))
        random.shuffle(ids)
        return ids
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

    def _readfile(self):
        with open('iohblade/benchmarks/logistics/A-n53-k7.json', 'r') as f:
            data = '\n'.join(f.readlines())
            data = json.loads(data)
            customers = data['customers']

            for customer in customers:
                if customer['demand'] == 0:
                    pass
                else:
                    self.customers.append(
                        Location(
                            customer['id'], 
                            customer['x'], 
                            customer['y'],
                        )
                    )
            self.lookup_table = dict([(location.id, location) for location in self.customers])
            self.benchmark = data['benchmark']

    def _check_accuracy(self, path: list[int]):
        unknown = [item for item in path if item not in self.lookup_table]
        if unknown:
            raise ValueError(f'Location ids {", ".join(map(str, unknown))} unknown.')
        missing = [item for item in self.lookup_table.keys() if item not in path]
        if missing:
            raise ValueError(f'Unserved customers {", ".join(map(str, missing))}.')
        if len(path) != len(set(path)):
            raise ValueError(f'Some customers were revisited.')
    
    def _transform_to_location_list(self, paths: list[int]) -> list[Location]:
        return list(map(lambda customer: self.lookup_table[customer], paths))

    def _calculate_length(self, path: list[int]) -> float:
        self._check_accuracy(path)
        location_path = self._transform_to_location_list(path)
        distance = 0
        previous = location_path[0]
        for customer in location_path[1:] + [previous]:
            distance += previous.distance_to(customer)
            previous = customer
        return distance

    def evaluate(self, solution: Solution, logger=None):
        name = solution.name if solution.name else 'TSPSolver'
        code = solution.code
        try:
            local_ns = {}
            global_ns = prepare_namespace(code, self.dependencies)
            compiled_code = compile(code, name, 'exec')
            exec(compiled_code, global_ns, local_ns)

            cls = local_ns[name]
            paths = cls([customer.vectorise() for customer in self.customers])()
            length = self._calculate_length(paths)
            self.best_known = min(self.best_known, length if not math.isfinite(length) else self.best_known)
            solution = solution.set_scores(length, f'Got distance {length}, best known distance is {self.best_known}.')
        except Exception as e:
            solution = solution.set_scores(float('inf'), f"Got error {e}", e)
        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)
    
    def to_dict(self):
        return self.__dict__


if __name__ == '__main__':
    tsp = TravelingSalesmanProblem()
    print(tsp.get_prompt())

