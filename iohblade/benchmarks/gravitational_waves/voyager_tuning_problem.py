import textwrap

from typing import Optional
from dfbench.problems import VoyagerTuningProblem
from dfbench import Objective
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace
from iohblade.benchmarks.gravitational_waves.base import GravitationalWaveBase


class VoyagerTuningProblemSolver(GravitationalWaveBase):
    """
    A gravitational wave detection problem based on Quasi-Universal Interferometer (UIFO).

    ## Params
     * `n_frequencies: int(50)`: Frequency points for sensitivity calculation.
     * `bounds_overrides: dict? (None): Overwrite the bounds of certain properties.
     * `signal_floor: float(1e-20)`: Lower floor for detector signal magnitudes before sensitivity normalization.
     * `duration: int(15 * 60)`: Budget for the evaluation of the function; officially it is set to 4 hours, set 15 mins for testing.
    """

    def __init__(
        self,
        n_frequencies=50,
        bounds_overrides=None,
        signal_floor=1e-20,
        duration: int = 15 * 60,
    ):
        super().__init__()
        self.name += f"_VoyagerTuningProblem_{duration}s"
        self.imports = textwrap.dedent("""
            from dfbench import Objective, OptimizationAlgorithm

            """)
        self.n_frequencies = n_frequencies
        self.bounds_overrides = bounds_overrides
        self.signal_floor = signal_floor
        self.duration = duration
        self.minimisation = True

    def evaluate(self, solution: Solution):
        solution, error = self.fix_code(solution)
        if error is not None:
            return solution

        code = solution.code
        name = solution.name
        ns = {}
        try:
            ns = prepare_namespace(self.imports + code, self.dependencies)
            compiled_code = compile(
                self.imports + code, filename="LLM_Code_" + name, mode="exec"
            )
            exec(compiled_code, ns, ns)

            solver = ns[name]
            voyager_problem = VoyagerTuningProblem(
                n_frequencies=self.n_frequencies,
                bounds_overrides=self.bounds_overrides,
                signal_floor=self.signal_floor,
            )

            obj =Objective(
                voyager_problem,
                unbounded=True,
                verbose=1,
                max_time=self.duration,
                print_every=50,  # Adapt this to your needs (per n evaluations)
            )

            solver_object = solver()

            _ = solver_object.optimize(obj)
            if obj.best_loss is not None:
                solution.set_scores(
                    obj.best_loss,
                    f"Got best loss of {obj.best_loss} in {obj.eval_count} evaluations; under {self.duration} s."
                )
                solution.metadata["best_solution"] = obj.best_params

            else:
                solution.set_scores(
                    float("inf"),
                    f"Got best loss of {float('inf')} under {self.duration} s; `objective.value(params)` never ran in optimisation loop.",
                )

        except Exception as e:
            solution.set_scores(float("inf"), f"Got error {e}.", e)
        return solution

    def get_config(self):
        import inspect

        evaluator = inspect.getsource(self.evaluate)
        extra_config = {
            "n_frequencies": self.n_frequencies,
            "bounds_overrides": self.bounds_overrides,
            "signal_floor": self.signal_floor,
            "duration": self.duration,
        }
        config = {
            "tags": [
                "physics",
                "gravitational waves",
                "frontier physics",
                "many dimension optimization",
            ],
            "name": "Voyager Tuning Problem",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator,
            "config": extra_config,
        }
        return config

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__.copy()


if __name__ == "__main__":
    voyager_tuning_ad = VoyagerTuningProblemSolver()
    solution = Solution(textwrap.dedent("""
        from dataclasses import dataclass
        import random
        from copy import deepcopy
        import numpy as np
        import jax.numpy as jnp
        from jaxtyping import Array, Float
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
                    self.Mutate(obj)
                    self.Evaluation(self.offspring_population, obj)
                    self.Selection()
                    generation += 1
                return obj.best_loss
        """))
    solution.name = "GeneticAlgorithm"
    solution = voyager_tuning_ad.evaluate(solution)
    print(voyager_tuning_ad.imports + solution.code)
    print(solution.fitness, solution.feedback, solution.error)
