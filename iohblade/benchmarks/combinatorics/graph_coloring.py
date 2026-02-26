from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace
from pathlib import Path


class GraphColoring(Problem):
    def __init__(self, benchmark_id: str, logger=None):
        """
        A Graph Colouring Benchmark focuses on assigning colours to all the nodes in undirected graph G(V,E), such that no two connecting
        nodes have the same colour.

        ## Args:
        - `benchmark_id: str` A string defining which benchmark to use from `./Graph_Coloring_Benchmarks`.
        - `logger: object`: An instance of one of the logger classes from `iohblade/loggers`.
        """
        self.nodes: list[int] = []
        self.edges: list[tuple[int, int]] = []
        self.benchmark = ""
        self._load_data(benchmark_id)
        Problem.__init__(self, name=self.benchmark, logger=logger)
        self.minimisation = True

        self.task_prompt = f"""
Write a python class with function `__call__`, that generates optimal Graph Coloring for a given set of vertices and edges.
- The class must initialise with 2 positional parameter:
    1. `nodes: list[int]`: A list of nodes id'd 1...{self.nodes}.
    2. `edges: list[tuple[int, int]]`: A list of edges in the undirected graph denotes as (u, v) where u and v are one of the `nodes`.
- The `__call__` method must return:
    - `color_mapping : dict[int, int]`: A node -> color mapping, i.e. `color_mapping[node_id] = color_id.
        - Where color_id is `int`, in 1...n.
- The optimisation goal is to minimise `n`, number of colours used, such that:
    - The assertion of color_mapping[u] != color_mapping[v] for all (u,v) pairs in `edges` stands.
- The returned fitness is going to be `len(set(color_mapping.values()))`.
    """
        self.example_prompt = """
An example response can be
---
# Descripition:
A novel algorithm for minimising the distinct colour count in graph colouring.
# Code:
```python
import random

class GraphColoring:
    def __init__(self, nodes: list[int], edges: list[tuple[int, int]]):
        self.nodes = nodes
        self.edges = edges
        self.color_mapping = {}
        self.color_mapping[self.nodes[0]] = 1

    def __call__(self):
        # Find steiner points.
        return self.color_mapping
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

    def _load_data(self, benchmark_id: str):
        path = (
            Path(__file__)
            .resolve()
            .parent.joinpath(f"Graph_Coloring_Benchmarks/gcol{benchmark_id}.txt")
        )
        self.benchmark = f"Graph-Coloring-{benchmark_id}"
        data = []
        with open(path) as f:
            data = f.readlines()
        for datum in data:
            datum = datum.split()
            if len(datum) == 4:
                self.nodes = list(range(1, int(datum[2]) + 1))
            if len(datum) == 3:
                assert datum[0] == "e"
                u = int(datum[1])
                v = int(datum[2])
                self.edges.append((u, v))

    def evaluate(self, solution: Solution, logger=None):
        code = solution.code
        name = solution.name if solution.name else "GraphColoring"
        try:
            compiled_code = compile(code, name, "exec")
            ns = prepare_namespace(code, self.dependencies)
            exec(compiled_code, ns, ns)
            cls = ns[name]
            coloring = cls(self.nodes, self.edges)()

            for u, v in self.edges:
                colorU = coloring[u]
                colorV = coloring[v]
                assert (
                    colorU != colorV
                ), f"Colours on nodes {u}, and {v} are same, while edge ({u}, {v}) exists."

            score = float(len(set(coloring.values())))
            solution = solution.set_scores(
                score, f"Got score {score}, try minimising further."
            )
        except Exception as e:
            solution = solution.set_scores(float("inf"), f"Encountered error {e}", e)
        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    GraphColoring(benchmark_id=1)
