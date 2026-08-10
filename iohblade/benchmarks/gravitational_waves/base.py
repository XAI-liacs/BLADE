import textwrap
from iohblade.problem import Problem


class GravitationalWaveBase(Problem):
    def __init__(self):
        super().__init__(
            name="GravitationalWave",
            dependencies=[
                "Learn2Design-2026 git+https://github.com/artificial-scientist-lab/Learn2Design-2026.git@main",
                "scipy",
                "jax",
                "jaxlib",
                "jaxtyping",
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
            * objective.best_loss → best loss found
            * objective.evals_since_improvement
            * objective.n_params

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
                    init_params: Float[Array, "..."] | None = None,
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

                def optimize(
                    self,
                    objective: Objective,
                    init_params=None,
                    random_seed=None,
                    patience=None,
                    **kwargs,
                ) -> float:
                    self.prepare(objective, unbounded=False, random_seed=random_seed)

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
            Do not import `OptimizationAlgorithm`, it will be provided in the solution harness.
            Always respond in the following format:

            # Description:
                Short description.
            # Code:
            ```python
                <code>
            ```
            """)
