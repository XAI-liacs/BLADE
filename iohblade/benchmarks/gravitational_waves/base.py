import textwrap
from iohblade.problem import Problem


class GravitationalWaveBase(Problem):
    def __init__(self):
        super().__init__(
            name="GravitationalWave",
            dependencies="Learn2Design-2026 git+https://github.com/artificial-scientist-lab/Learn2Design-2026.git@main",
        )

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
                    ) -> float:
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
                    * `objective.n_params`: Number of parameters to optimise for.
                * `kwargs`: are the custom hyperparameters for algorithm you'd write, feel free to add those parameters if needed.
            * Your goal is to optimise the `objective.params`, so that the loss of signal measured from `objective.value(params)`
            is minimised.
            * Make sure to use self.prepare(obj, unbounded:bool, random_seed=random_seed) on line 1 of the optimize function.
            * Make sure to run objective.warmup_value() and objective.start_logging() at the start of optimisation process.
            * Make sure to return objective.best_loss at the end of the loop.
            """)

        self.example_prompt = textwrap.dedent("""
            * Here is an example program for running a UIFOProblem solver.
            ```python
            @dataclass
            class Genome:
                params : list[float]
                speed : list[float]
                fitness: float = float('nan')

                def reset_fitness(self):
                    self.fitness = float('nan')


            class GeneticAlgorithm(OptimizationAlgorithm):

                algorithm_str = 'GeneticAlgorithm'

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
                        genome = Genome(params, [0] * len(params))
                        parent_population.append(genome)
                    self.parent_population = parent_population

                def Mutate(self, objective: Objective):
                    for i in range(len(self.offspring_population)):
                        individual = self.mutate(self.offspring_population[i], objective)
                        self.offspring_population[i] = individual

                def mutate(self, individual: Genome, objective: Objective) -> Genome:
                    params, speed = self.update_params(individual.params, individual.speed, objective)
                    return_individual = Genome(params, speed)
                    return return_individual

                def Crossover(self):
                    offspring_population = []
                    population = self.parent_population[:]
                    random.shuffle(population)
                    for i in range(len(population)):
                        individual = self.crossover(population[i], population[(i + 1) % len(population)])
                        offspring_population.append(individual)

                    self.offspring_population = offspring_population


                def crossover(self, parent1: Genome, parent2: Genome):
                    crossover_point = random.randint(1, len(parent1.params) // 2)
                    params = parent1.params[:crossover_point] + parent2.params[crossover_point:]
                    speed = parent1.speed[:crossover_point] + parent2.speed[crossover_point:]
                    return Genome(params, speed)

                def mutationrate(self, x):
                    return 2 * self.mutation_probability * (1 / (1 + np.e ** ((x)/(self.mutation_probability))))

                def update_params(self, params, speed, obj: Objective) -> tuple[list[float],list[float]]:
                    params = deepcopy(params)
                    speed = deepcopy(speed)
                    if isinstance(params, Array):
                        params = params.to_list()
                    std = np.std(list(map(lambda x: x.fitness, self.parent_population)))
                    mu = np.mean(list(map(lambda x: x.fitness, self.parent_population)))
                    mutation_probability = self.mutationrate(std/mu)
                    print(f'\t Mutation probability: {mutation_probability}, parent population standard deviation {std}.')
                    for i in range(len(params)):
                        if random.random() < mutation_probability:
                            speed[i] = random.uniform(speed[i], self.std)
                            params[i] *= (np.e ** speed[i])
                    return params, speed

                def Selection(self):
                    solutions = self.parent_population + self.offspring_population
                    solutions = sorted(solutions, key= lambda x: x.fitness)
                    self.parent_population = solutions[:self.Lambda]
                    self.offspring_population = []

                def Evaluation(self, population: list[Genome], obj):
                    for genome in population:
                        x = jnp.asarray(genome.params)
                        loss = float(obj.value(x))
                        print(f'\t\tLoss: {loss}{"*" if loss < max(list(map(lambda x: x.fitness, self.parent_population))) else ""}')
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

                    generation = 1
                    self.Initialise(obj)
                    self.Evaluation(self.parent_population, obj)
                    while not obj.budget_exceeded:
                        print(f"Generation {generation}....")
                        self.Crossover()
                        self.Mutate(obj)
                        self.Evaluation(self.offspring_population, obj)
                        self.Selection()
                        generation += 1
                    return obj.best_loss
            ```
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
