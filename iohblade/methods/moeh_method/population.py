import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from codebleu.syntax_match import calc_syntax_match

from iohblade.fitness import Fitness
from iohblade.solution import Solution


class Population:
    def __init__(self, size, minimisation: bool) -> None:
        self.size = size
        self._population: list[Solution] = []
        self._history: list[Solution] = []
        self.minimisation = minimisation
        self.best_known: list[Solution] = []

    def append(self, element: Solution):
        if element.fitness_is_valid():
            self._population.append(element)

    def __len__(self):
        return len(self._population)

    def __getitem__(self, key):
        return self._population[key]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    def _get_delta_domination_matrix(self) -> np.ndarray:
        N = len(self._population)
        S = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i, j] = -calc_syntax_match(
                        self._population[i].code, self._population[j].code, "python"
                    )

                    if (
                        self._population[i].fitness <= self._population[j].fitness
                        and not self.minimisation
                    ):
                        S[i, j] = 0
                    elif (
                        self._population[i].fitness >= self._population[j].fitness
                        and self.minimisation
                    ):
                        S[i, j] = 0
        return S

    def parent_selection(self, d: int, test: bool = False) -> list[Solution]:
        S = self._get_delta_domination_matrix()
        if test:
            print(f"\tS pre column sum: \n{S}")
        if S.shape == (0, 0):
            if test:
                print("\tEarly exit, population empty.")
            return []

        v = np.sum(S, axis=0)

        if test:
            print(f"\tS after column sum: {v}")

        v = v - np.max(v)

        if test:
            print(f"v: {v}")
        pdf = np.exp(v)
        pdf = pdf / pdf.sum()
        if test:
            print(f"\tSoftmax: {pdf}")
        idx = np.random.choice(
            range(len(self._population)), size=d, replace=False, p=pdf
        )
        return [self._population[i] for i in idx]

    def population_management(self, test: bool = False) -> list[Solution]:
        M = self._get_delta_domination_matrix()

        if test:
            print("Selection Matrix:", M)
        v = np.sum(M, axis=0)
        if test:
            print("Column wise sum:", v)
        k = list(
            map(lambda y: y[0], sorted(enumerate(v), key=lambda x: x[1], reverse=True))
        )
        if test:
            print("Sorted Index:", k)
        new_population: list[Solution] = [self._population[i] for i in k[: self.size]]
        print(len(new_population), k[: self.size])
        self._population = new_population
        return self._population

    def get_best(self) -> list[Solution]:
        if len(self.population) == 0:
            return []

        if isinstance(self.population[0].fitness, Fitness):
            fitness = np.array(
                [individual.fitness.to_vector() for individual in self._population]
            )
            nds = NonDominatedSorting()
            indexes = nds.do(
                fitness if self.minimisation else -fitness,
                only_non_dominated_front=True,
            )
            return [self._population[index] for index in indexes]
        population = list(
            sorted(
                self._population, key=lambda x: x.fitness, reverse=not self.minimisation
            )
        )
        return [population[0]]
