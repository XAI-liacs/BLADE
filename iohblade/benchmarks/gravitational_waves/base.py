import ast
import textwrap
from typing import Optional
from iohblade.problem import Problem
from iohblade.solution import Solution

class OptimizationAlgorithmFixer(ast.NodeTransformer):
    """LLMs are not following instructions properly; implemented AST fixed to find most common mistakes and fix them:
        1) Unvalid or no Class inheritance of OptimizationAlgorithm.
        2) Overloading of non-abstract function `prepare`.
    """
    def __init__(self, class_name="name"):
        self.class_name = class_name
        self.found = False
        self.prepare_found = False

    def visit_ClassDef(self, node):
        if node.name == 'Objective' or node.name == 'OptimizationAlgorithm':
            return None
        else:
            self.class_name = node.name # Update class name.
        if node.name == self.class_name:
            self.found = True

            node.bases = [
                ast.Name(id="OptimizationAlgorithm", ctx=ast.Load())
            ]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == 'prepare':
            self.prepare_found = True
            return None
        return self.generic_visit(node)


class GravitationalWaveBase(Problem):
    def __init__(self):
        super().__init__(
            name="GravitationalWave",
            dependencies=[
                "git+https://github.com/artificial-scientist-lab/Learn2Design-2026.git@main",
            ],
        )

        self.role_prompt = "You are an excellent junior researcher at the CERN."
        self.task_prompt = textwrap.dedent("""
            You are an expert numerical optimization researcher. Design and implement a high-performance optimizer for the
            Learn2Design-2026 gravitational-wave detector design benchmark.

            Your goal is to minimize the objective while finding a feasible solution. The problem has ~200 continuous parameters,
            heterogeneous scales, nonlinear constraints, and expensive JAX-based objective evaluations. The optimizer is evaluated on
            unseen topologies, so robustness/generalization matters more than tuning for one instance.

            The optimizer has access to:

            * objective.value(params) → loss
            * objective.grad(params) → exact gradient
            * objective.hessian(params) → exact Hessian
            * objective.value_aux(params) → (loss, aux), where aux contains sensitivity_loss, penalty, is_feasible, violations, power_values
            * objective.random_params(n_samples=1) → random valid parameters
            * objective.warmup_value() → free JIT warmup
            * objective.start_logging() → starts the timed/evaluation budget
            * objective.budget_exceeded → budget termination condition
            * objective.best_loss → best (least) loss found
            * objective.evals_since_improvement -> Number of evaluations since last improvement.
            * objective.n_params -> Number of parameters to optimise ( = len(objective.values(params))).

            The benchmark has a hard computational budget and rewards the best feasible solution found. Objective evaluations are expensive,
            so avoid redundant evaluations and exploit batching/vectorization where possible.

            Choose the optimization strategy yourself. Do not blindly implement a textbook optimizer. Consider whether a hybrid global/local
            method, gradient-based method, second-order method, population method, adaptive restart strategy, or combination is most
            appropriate. Exact derivatives are available, but use Hessians only when their cost is justified.

            Handle parameter scaling, constraints, numerical instability, and local minima robustly. Prefer simple, computationally
            efficient strategies over unnecessary complexity.

            The class must inherit from OptimizationAlgorithm and implement:
            ```python
                def optimize(
                    self,
                    objective: Objective,
                    init_params: list[[float]] | None = None,
                    random_seed: int | None = None,
                    patience: int | None = None,
                    **kwargs,
                ) -> float:
            ```

            The first executable line of optimize must be:

            self.prepare(objective, unbounded=False, random_seed=random_seed)

            Then call:

            objective.warmup_value()
            objective.start_logging()

            before beginning the budgeted optimization.

            Use objective.budget_exceeded as the main termination condition and return objective.best_loss.

            Use init_params when provided; otherwise initialize appropriately.

            The implementation must be self-contained, syntactically valid, compatible with the provided APIs, and contain no placeholders or invented Objective methods.

            Return only the Python code for the optimizer class and its necessary imports. No explanation or pseudocode.
            """)

        self.example_prompt = textwrap.dedent("""
            * Here's a stub to get started working on the task:
            ```python
                class MyAlgorithm(OptimizationAlgorithm):

                algorithm_str = "MyAlgorithm"

                def __init__(self):
                    super()
                    ...


                def optimize(
                    self,
                    objective: Objective,
                    init_params=None,
                    random_seed=None,
                    patience=None,
                    **kwargs,
                ) -> float:
                    self.prepare(objective, unbounded=False, random_seed=random_seed) #Do not overload this function.

                    objective.warmup_value()
                    objective.start_logging()

                    # optimization algorithm here

                    while not objective.budget_exceeded:
                        ...

                    return objective.best_loss
            ```
            Update the `algorithm_str`, and Class Name appropriately.
        """)

        self.format_prompt = textwrap.dedent("""
            Do not import or define `OptimizationAlgorithm` or `Objective`, only write the algorithm with an `optimize` function, it will be provided in the solution harness.
            Always respond in the following format:

            # Description:
                Short description.
            # Code:
            ```python
                <code>
            ```
            """)

    def fix_code(self, individual: Solution) -> tuple[Solution, Optional[Exception]]:
        """Fix commonly found errors in LLM written code."""
        name = individual.name or '<String>'
        code = individual.code or ""
        try:
            tree = ast.parse(code)

            fixer = OptimizationAlgorithmFixer(name)
            tree = fixer.visit(tree)
            ast.fix_missing_locations(tree)

            if not fixer.found:
                raise ValueError(f"Class {name} was not found")

            if fixer.prepare_found:
                raise ValueError("Non-abstract function `prepare` was overloaded.")
        except Exception as e:
            individual.set_scores(
                float('inf'),
                e
            )
            return individual, e
        individual.code = ast.unparse(tree)
        individual.name = fixer.class_name
        return individual, None
