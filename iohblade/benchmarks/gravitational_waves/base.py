import textwrap
from iohblade.problem import Problem

class GravitationalWaveBase(Problem):
    def __init__(self, dependencies='learn2Design-2026 @ git+https://github.com/artificial-scientist-lab/Learn2Design-2026.git'):
        super().__init__(name="GravitationalWave")

        self.role_prompt = "You are an excellent junior researcher at the CERN."
        self.task_prompt = textwrap.dedent("""
            * Write a python `class` that inherits from `OptimizationAlgorithm`, and has the method:
                ```python
                    def optimize(
                        self,
                        objective: Objective,
                        init_params: Float[Array, "..."] | None = None,
                        random_seed: int | None = None,
                        patience: int | None = None,
                        **kwargs: dict
                    ) -> None:
                ```
                * Where Objective is a class providing following functionality:
                    * `objective.value(params)` returns the loss of the given configuration.
                    * `objective.grad(params)` returns gradient of the given param configuation.
                    * `objective.hessian(params)` returns exact Hessian.
                    * `objective.value_aux(params)`: returns (loss, aux) dict: sensitivity_loss, penalty, is_feasible, violations, power_values.
                    * `objective.warmup_value()`: JIT-compile before the timer starts (free, not budgeted). Call before start_logging().
                    * `objective.start_logging()`:	Start the countdown on the evaluation time; call after warmup.
                    * `objective.random_params(n_samples=1)`: Returns JIT array, of random parameters.
                    * `objective.budget_exceeded`:	Main loop-termination check (time or evals exhausted).
                    * `objective.evals_since_improvement`: Number of evaluations since last improvement.
                    * `objective.best_loss`: Best fitness till now (minimisation).
                    * `objective.bounds`:   Per-parameter lows and highs arrays (±inf when unbounded) shaped [[lows], [highs]].
                    * `objective.n_params`: Number of parameters to optimise for.
                * `kwargs`: are the custom hyperparameters for algorithm you'd write, feel free to add those parameters if needed.
            * Your goal is to optimise the `objective.params` within the `objective.bounds`, so that the loss of signal measured from `objective.value(params)`
            is minimised.
            * Make sure to use self.prepare(obj, unbounded:bool, random_seed=random_seed) on line 1 of the optimize function.
            * Make sure to run objective.warmup_value() and objective.start_logging() at the start of optimisation process.
            """)

        self.example_prompt = textwrap.dedent("""
            * Here is an example program for running a UIFOProblem solver.
            ```python
                @dataclass
                class Genome:
                    params : list[float]
                    fitness: float = float('nan')

                    def reset_fitness(self):
                        fitness = float('nan')


                class EvolutionaryStrategy(OptimizationAlgorithm):

                    algorithm_str = 'EvolutionaryStrategy'

                    def __init__(self, Lambda=10, Mu=10) -> None:
                        super()
                        self.parent_population: list[Genome] = []
                        self.offspring_population: list[Genome] = []
                        self.Lambda = Lambda
                        self.Mu = Mu

                    def Initialise(self, obj: Objective):
                        parent_population = []
                        for i in range(self.Lambda):
                            params = list(obj.random_params())
                            genome = Genome(params)
                            parent_population.append(genome)
                        self.parent_population = parent_population

                    def Mutate(self):
                        for i in range(len(self.offspring_population)):
                            individual = self.mutate(self.offspring_population[i])
                            self.offspring_population[i] = individual

                    def mutate(self, individual: Genome) -> Genome:
                        params = self.update_params(individual.params)
                        return_individual = Genome(params)
                        return return_individual

                    def Crossover(self):
                        offspring_population = []
                        for i in range(len(self.parent_population)):
                            individual = self.crossover(self.parent_population[i], self.parent_population[(i + 1) % len(self.parent_population)])
                            offspring_population.append(individual)

                        self.offspring_population = offspring_population


                    def crossover(self, parent1: Genome, parent2: Genome):
                        params = []
                        for i in range(len(parent1.params)):
                            weight = random.random()
                            param = (parent1.params[i] * weight) + (parent2.params[i] * (1 - weight))
                            params.append(param)
                        return Genome(params)


                    def update_params(self, params) -> tuple[list[float], list[float]]:
                        params = deepcopy(params)
                        if isinstance(params, Array):
                            params = params.to_list()
                        for i in range(len(params)):
                            if random.random() < self.mutation_probability:
                                lo, hi = self.bounds[i]
                                new_param = lo + (random.random() * (hi - lo))
                                params[i] = new_param

                        return params

                    def Selection(self):
                        solutions = self.parent_population + self.offspring_population
                        solutions = sorted(solutions, key= lambda x: x.fitness)
                        self.parent_population = solutions[:self.Lambda]
                        self.offspring_population = []

                    def Evaluation(self, population: list[Genome], obj):
                        for genome in population:
                            x = jnp.asarray(genome.params)
                            loss = float(obj.value(x))
                            print(f'\t\tLoss: {loss}.')
                            genome.fitness = loss


                    def optimize(
                        self,
                        objective: Objective,
                        init_params: Float[Array, "..."] | None = None,
                        random_seed: int | None = None,
                        patience: int | None = None,
                        mutation_probability: float = 0.5,
                        std: float = 2
                    ) -> None:
                        obj = objective
                        self.mutation_probability = mutation_probability
                        self.std = std

                        self.prepare(obj, unbounded=False, random_seed=random_seed)

                        obj.warmup_value()
                        obj.start_logging()

                        # Repeatedly draw random params from the active (bounded) space and
                        # evaluate them until the budget is exhausted. Each `obj.value()` call
                        # is automatically logged.
                        self.bounds = list(zip(obj.bounds[0], obj.bounds[1]))

                        generation = 1
                        self.Initialise(obj)
                        self.Evaluation(self.parent_population, obj)
                        while not obj.budget_exceeded:
                            print(f"Generation {generation}....")
                            self.Crossover()
                            self.Mutate()
                            self.Evaluation(self.offspring_population, obj)
                            self.Selection()
                            generation += 1
                        return obj.best_params_bounded, obj.best_loss
            ```
        """)

        self.format_prompt = textwrap.dedent('''
            Always respond in the following format:

            # Description:
                Short description.
            # Code:
            ```python
                <code>
            ```
            ''')


if __name__ == '__main__':
    o = GravitationalWaveBase()
    print(o.get_prompt())
