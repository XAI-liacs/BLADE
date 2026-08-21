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
    def __init__(self, cuda_version=13):
        # jax_handler = f"jax[cuda{cuda_version}]"
        super().__init__(
            name="GravitationalWave",
            dependencies=[
                f"Learn2Design[cuda{cuda_version}] @ git+https://github.com/artificial-scientist-lab/Learn2Design-2026.git@main",
            ],
        )

        self.role_prompt = "You are an excellent junior researcher at the CERN."
        self.task_prompt = textwrap.dedent("""
            You are an expert numerical optimization researcher and Python/JAX engineer.

            Your task is to implement a VALID, ROBUST, SELF-CONTAINED optimizer for the Learn2Design-2026 gravitational-wave detector design benchmark.

            PRIMARY OBJECTIVE

            Your first priority is producing code that:

            1. Parses successfully.
            2. Imports successfully.
            3. Instantiates successfully.
            4. Uses ONLY the APIs explicitly documented below.
            5. Implements the required optimize method exactly.
            6. Runs until the evaluation budget is exhausted or the optimizer terminates safely.
            7. Returns a valid numeric result.

            Only after satisfying all of the above should you optimize performance.

            A sophisticated optimizer that does not run is a complete failure. A simple optimizer that runs reliably is preferable.

            ⸻

            AVAILABLE API — DO NOT INVENT ANY OTHER METHODS

            The optimizer receives an objective object exposing exactly the following relevant interface:

            objective.value(params) -> loss
            objective.grad(params) -> gradient
            objective.hessian(params) -> hessian
            objective.value_aux(params) -> (loss, aux)
            objective.random_params(n_samples=1) -> random valid parameters
            objective.warmup_value() -> free JIT warmup
            objective.start_logging() -> starts the timed/evaluation budget
            objective.budget_exceeded -> bool
            objective.best_loss -> float
            objective.evals_since_improvement -> int
            objective.n_params -> int

            aux returned by value_aux may contain:

            sensitivity_loss
            penalty
            is_feasible
            violations
            power_values

            Do NOT assume any other attributes, methods, fields, or behavior.

            Do NOT use undocumented methods such as:

            objective.evaluate(...)
            objective.is_feasible(...)
            objective.project(...)
            objective.clip(...)
            objective.constraints(...)
            objective.params(...)
            objective.bounds(...)
            objective.reset(...)

            unless they are explicitly listed above.

            ⸻

            REQUIRED CLASS INTERFACE

            Your optimizer must inherit from OptimizationAlgorithm.

            Implement exactly:

            def optimize(
                self,
                objective: Objective,
                init_params: list[float] | None = None,
                random_seed: int | None = None,
                patience: int | None = None,
                **kwargs,
            ) -> float:

            The first executable line of optimize MUST be exactly:

            self.prepare(objective, unbounded=False, random_seed=random_seed)

            Immediately after that, execute:

            objective.warmup_value()
            objective.start_logging()

            The optimization loop must then use:

            while not objective.budget_exceeded:

            as its primary termination condition.

            At termination, return:

            return objective.best_loss

            Do not return a locally maintained best value instead of objective.best_loss.

            ⸻

            PARAMETER REPRESENTATION

            Treat params as a flat sequence of approximately 200 continuous floating-point parameters.

            Convert parameters to a form compatible with NumPy/JAX only when necessary.

            Do not assume that parameters have a fixed dimension. Always use:

            n = objective.n_params

            and construct vectors of that size.

            Do not hard-code the number of parameters.

            ⸻

            INITIALIZATION

            If init_params is provided, use it as the initial candidate.

            If it is not provided, obtain an initial valid candidate using:

            objective.random_params()

            Do not invent a parameter initialization scheme that assumes undocumented bounds.

            Do not assume that randomly generated parameters are represented in any particular container type beyond being compatible with the objective API.

            ⸻

            FEASIBILITY

            Feasibility is important because the benchmark rewards the best feasible solution.

            However, there is NO documented projection or constraint API.

            Therefore:

            * Do not invent one.
            * Do not assume explicit lower/upper bounds unless they are available through the provided parameter representation.
            * Do not fabricate constraint equations.
            * Prefer candidates generated by objective.random_params() when a feasible starting point is needed.
            * If using objective.value_aux, inspect aux["is_feasible"] only when that key actually exists.
            * Never crash because optional auxiliary information is missing.

            The optimizer should prefer feasible candidates when feasibility information is available.

            ⸻

            DERIVATIVES

            Exact derivatives are available:

            objective.grad(params)
            objective.hessian(params)

            Use gradients when they provide a meaningful advantage.

            Use Hessians ONLY if there is a clear computational justification.

            Do not call the Hessian repeatedly inside every iteration without justification: objective evaluations are expensive and the benchmark has a hard computational budget.

            Do not assume gradients or Hessians have a particular Python/JAX array type beyond being numerically usable.

            Handle NaN, Inf, overflow, and numerical failures gracefully.

            ⸻

            COMPUTATIONAL BUDGET

            Objective evaluations are expensive.

            Every call to:

            objective.value(...)
            objective.grad(...)
            objective.hessian(...)
            objective.value_aux(...)

            may consume significant computational resources.

            Avoid redundant evaluations.

            Never create an uncontrolled inner loop that can continue after:

            objective.budget_exceeded

            becomes true.

            Check the budget frequently, especially inside population, line-search, restart, or candidate-generation loops.

            The optimizer must always have a path to terminate.

            ⸻

            RANDOMNESS

            Use random_seed when randomness is required.

            Prefer a local random-number generator rather than modifying global random state.

            For example:

            rng = np.random.default_rng(random_seed)

            Do not rely on global random state.

            ⸻

            OPTIMIZATION STRATEGY

            You may choose the optimization algorithm.

            The problem has:

            * approximately 200 continuous parameters,
            * heterogeneous parameter scales,
            * nonlinear constraints,
            * expensive JAX-based evaluations,
            * exact gradients,
            * exact Hessians,
            * potentially difficult local minima,
            * unseen detector topologies during evaluation.

            Good strategies may include:

            * gradient-based optimization,
            * trust-region methods,
            * quasi-Newton methods,
            * adaptive coordinate search,
            * stochastic local search,
            * population-based methods,
            * random-restart local optimization,
            * hybrid global/local optimization.

            You may combine methods if useful.

            However:

            Do not implement unnecessary complexity.

            Prefer a small number of robust mechanisms over a large optimizer framework.

            Do not write a toy implementation merely to satisfy the interface, but do not introduce complicated machinery that is likely to contain bugs.

            ⸻

            NUMERICAL ROBUSTNESS

            The optimizer must survive:

            * NaN objective values,
            * Inf objective values,
            * NaN gradients,
            * Inf gradients,
            * failed candidate evaluations,
            * very large gradients,
            * very small gradients,
            * ill-conditioned curvature,
            * failed line searches,
            * stagnation.

            Use finite-value checks where appropriate.

            Never allow a single invalid candidate to crash the entire optimization.

            If a sophisticated optimization step fails, fall back to a simpler valid step.

            A robust fallback strategy is strongly preferred.

            ⸻

            PATIENCE

            If:

            patience is not None

            you may use:

            objective.evals_since_improvement

            to detect stagnation.

            Do not assume patience is always provided.

            Do not terminate merely because a fixed number of iterations has passed unless that is explicitly justified.

            The benchmark’s evaluation budget remains the primary termination condition.

            ⸻

            JAX / PERFORMANCE

            The objective is JAX-based.

            Avoid unnecessary conversions between:

            Python lists
            NumPy arrays
            JAX arrays

            Do not introduce JIT compilation for the optimizer itself unless there is a clear reason.

            Do not repeatedly compile dynamically shaped functions.

            Do not write code that assumes JAX internals that are not part of the documented API.

            Keep the optimizer compatible with ordinary NumPy/SciPy-style numerical operations where possible.

            ⸻

            IMPORTS

            Only import packages that are actually needed.

            Assume standard scientific Python packages such as:

            numpy
            scipy

            are available if appropriate.

            Do not introduce obscure third-party dependencies.

            Do not install packages.

            Do not access the filesystem, network, subprocesses, environment variables, or external services.

            ⸻

            IMPORTANT IMPLEMENTATION RULES

            The generated code MUST:

            * contain exactly one optimizer class unless additional helper classes/functions are genuinely necessary;
            * inherit from OptimizationAlgorithm;
            * define the required optimize method;
            * call self.prepare(...) as its first executable statement;
            * call objective.warmup_value();
            * call objective.start_logging();
            * use objective.budget_exceeded for termination;
            * return objective.best_loss;
            * respect init_params;
            * respect random_seed;
            * avoid undocumented Objective APIs;
            * avoid hard-coded dimensionality;
            * avoid infinite loops;
            * avoid uncontrolled evaluation loops;
            * handle numerical failures;
            * contain no TODOs;
            * contain no placeholders;
            * contain no pseudocode;
            * contain no ellipses such as ...;
            * contain no comments describing code that was not actually implemented.

            ⸻

            BEFORE RETURNING THE CODE

            Silently perform a mental compile/runtime audit.

            Check all of the following:

            1. Are every imported module and symbol valid?
            2. Is OptimizationAlgorithm imported from the correct location?
            3. Is Objective imported from the correct location?
            4. Does the class actually inherit from OptimizationAlgorithm?
            5. Does optimize have the required signature?
            6. Is self.prepare(...) the first executable statement?
            7. Are objective.warmup_value() and objective.start_logging() called?
            8. Is objective.budget_exceeded checked in every potentially long-running loop?
            9. Can any loop become infinite?
            10. Are all variables initialized before use?
            11. Are NumPy/SciPy calls given valid argument types?
            12. Are array dimensions consistent?
            13. Are gradients handled with the correct shape?
            14. Can NaN/Inf values crash the optimizer?
            15. Does the code use any Objective API that was not explicitly documented?
            16. Does init_params actually affect initialization?
            17. Does random_seed actually control optimizer randomness?
            18. Does the method always eventually return objective.best_loss?
            19. Is the code syntactically valid Python?
            20. Is there any placeholder, pseudocode, or invented implementation detail?

            If any answer is “no”, fix the implementation before returning it.

            Do not explain your reasoning or provide a design discussion.

            Return ONLY the complete Python source code for the optimizer and its necessary imports.
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
