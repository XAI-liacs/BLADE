import json
from dataclasses import dataclass
from pathlib import Path

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace


@dataclass
class Location:
    id: int
    x: float
    y: float
    demand: float

    def vectorise(self):
        return [self.id, self.x, self.y, self.demand]

    def distance_to(self, other: "Location"):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class VehicleRoutingProblem(Problem):
    def __init__(self, benchmark_id):
        self.customers: list[Location] = []
        self.depot: Location = Location(0, 0, 0, 0)
        self.vehicle_capacity = 0.0
        self.fleet_size = 0
        self.best_known = float("inf")
        self.benchmark = ""
        self.minimisation = True
        self._readfile(benchmark_id)

        Problem.__init__(self, name=f"CVRP-{self.benchmark}")

        self.task_prompt = """
Write a python class with function `__call__`, that generate a solution for Capacitated Vehicle Routing Problem.
- The class must initialise with 4 positional parameters:
    1. `depot`: A location from where all vehicles start and end their journey.
    2. `customers`: A list of locations, representing customers.
    3. `fleet_size`: Number of vehicles that has to be used during the routine.
    4. `vehicle_capacity`: The total capacity of vehicles (each vehicle assumed to be same).
- `Note`: Both depot and each of customers are represented by list corresonding [id, x-coordinate, y-coordinate, demand],
    demand for depot is 0.
- The `__call__` method must return:
    - `paths : list[list[int]]`: A list of `fleet_size` paths, each path is a list of customer's id.
        - The sum of demand of customers in any of the path must not exceed the `vehicle_capacity`.
        - Each customer must only be served once.
        - No customer must be left un-served.
        - Depot must not exist in the `paths`.
- The optimisation goal is to minimise total distance travelled by the fleet of vehicles."""

        self.example_prompt = f"""
An example response can be
---
# Descripition: 
A random selection algorithm for Capacited Vehicle Routing Problem.
# Code:
```python
import random
import math

class CapacitatedVehicleRoutingProblem:
    def __init__(self, depot, customers, fleet_size, vehicle_capacity):
        self.depot = depot
        self.customers = customers
        self.fleet_size = fleet_size
        self.vehicle_capacity = vehicle_capacity

    def __call__(self):
        customers = self.customers[:]
        random.shuffle(customers)

        paths = [[] for _ in range(self.fleet_size)]
        loads = [0] * self.fleet_size

        for cid, x, y, demand in customers:
            for i in range(self.fleet_size):
                if loads[i] + demand <= self.vehicle_capacity:
                    paths[i].append(cid)
                    loads[i] += demand
                    break
            else:
                raise ValueError("No feasible assignment found")

        return paths
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

    def _readfile(self, benchmark_id: str):
        path = Path(__file__).resolve().parent.joinpath(f"{benchmark_id}.json")
        with open(path, "r") as f:
            data = "\n".join(f.readlines())
            data = json.loads(data)
            customers = data["customers"]
            vehicle_capacity = data["vehicleCapacity"]
            best_known = data["optimal"]
            fleet_size = data["fleetSize"]

            for customer in customers:
                if customer["demand"] == 0:
                    self.depot = Location(
                        customer["id"], customer["x"], customer["y"], customer["demand"]
                    )
                else:
                    self.customers.append(
                        Location(
                            customer["id"],
                            customer["x"],
                            customer["y"],
                            customer["demand"],
                        )
                    )
            self.vehicle_capacity = vehicle_capacity
            self.best_known = best_known
            self.fleet_size = fleet_size
            self.lookup_table = dict(
                [
                    (location.id, location)
                    for location in (self.customers + [self.depot])
                ]
            )
            self.benchmark = data["benchmark"]

    def _check_accuracy(self, paths: list[list[int]]):
        if len(paths) != self.fleet_size:
            raise ValueError(f"Expected {self.fleet_size} paths, got {len(paths)}.")
        flatten_list = sum(paths, [])
        unknown = [item for item in flatten_list if item not in self.lookup_table]
        if unknown:
            raise ValueError(f'Location ids {", ".join(map(str, unknown))} unknown.')
        missing = [
            item
            for item in self.lookup_table.keys()
            if item not in flatten_list and item != self.depot.id
        ]
        if missing:
            raise ValueError(f'Unserved customers {", ".join(map(str, missing))}.')
        if self.depot.id in flatten_list:
            raise ValueError(f"Depot inexpectedly found in path.")
        if len(flatten_list) != len(set(flatten_list)):
            raise ValueError(f"Some customers were revisited.")

    def _transform_to_location_list(
        self, paths: list[list[int]]
    ) -> list[list[Location]]:
        routine = []
        for path in paths:
            location_path = list(
                map(lambda customer: self.lookup_table[customer], path)
            )
            routine.append(location_path)
            filled = sum(map(lambda x: x.demand, location_path))
            if filled > self.vehicle_capacity:
                raise ValueError(
                    f'Path {", ".join(map(str, path))}, exceeds vehicle capacity {self.vehicle_capacity}'
                )
        return routine

    def _calculate_length(self, paths: list[list[int]]) -> float:
        self._check_accuracy(paths)
        location_paths = self._transform_to_location_list(paths)
        distance = 0
        for path in location_paths:
            previous = self.depot
            for location in path + [self.depot]:
                distance += previous.distance_to(location)
                previous = location
        return distance

    def evaluate(self, solution: Solution, logger=None):
        name = solution.name
        code = solution.code
        try:
            local_ns = {}
            global_ns = prepare_namespace(code, self.dependencies)
            compiled_code = compile(code, name, "exec")
            exec(compiled_code, global_ns, local_ns)

            cls = local_ns[name]
            paths = cls(
                self.depot.vectorise(),
                [customer.vectorise() for customer in self.customers],
                self.fleet_size,
                self.vehicle_capacity,
            )()
            length = self._calculate_length(paths)
            solution = solution.set_scores(
                length,
                f"Got distance {length}, best known distance is {self.best_known}.",
            )
        except Exception as e:
            solution = solution.set_scores(float("inf"), f"Got error {e}", e)
        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    vrp = VehicleRoutingProblem()
    print(vrp.get_prompt())
