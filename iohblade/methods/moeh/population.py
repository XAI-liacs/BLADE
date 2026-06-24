from codebleu.syntax_match import calc_syntax_match
import numpy as np
from iohblade.solution import Solution

class Population:
    def __init__(self, size, minimisation: bool) -> None:
        self.size = size
        self._population: list[Solution] = []
        self.minimisation = minimisation

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

    def parent_selection(self, d: int, test: bool=False) -> list[Solution]:
        N = self.size
        S = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i, j] = -calc_syntax_match(
                        self._population[i].code,
                        self._population[j].code,
                        "python"
                    )

                    if self._population[i].fitness < self._population[j].fitness and not self.minimisation:
                        S[i, j] = 0
                    elif self._population[i].fitness > self._population[j].fitness and self.minimisation:
                        S[i, j] = 0

        v = np.sum(S, axis=0)
        if test:
            print(f"S: {S}")
        v = v - np.max(v)
        if test:
            print(f'v: {v}')
        pdf = np.exp(v)
        pdf = pdf / pdf.sum()
        if test:
            print(f'Softmax: {pdf}')
        idx = np.random.choice(N, size=d, replace=False, p=pdf)
        return [self._population[i] for i in idx]
    
    def population_management(self, test: bool = False) -> list[Solution]:
        S = len(self._population)
        M = np.zeros((S, S), dtype=float)
        for i in range(S):
            for j in range(S):
                if i != j:
                    M[i][j] = -calc_syntax_match(
                        self._population[i].code,
                        self._population[j].code,
                        'python'
                    )
                if self._population[i].fitness < self._population[j].fitness and not self.minimisation:
                    M[i][j] = 0
                elif self._population[i].fitness > self._population[j].fitness and self.minimisation:
                    M[i][j] = 0
        if test:
            print('Selection Matrix:')
            print(M)
        v = np.sum(M, axis=0)
        if test:
            print('Column wise sum:')
            print(v)
        k = list(map
                    (lambda y: y[0], 
                        sorted(enumerate(v), key=lambda x: x[1], reverse=True)
                    )
                )
        if test:
            print('Sorted Index:')
            print(k)
        new_population : list[Solution] = [self._population[i] for i in k]
        self._population = new_population
        return self._population
    
    def get_best(self) -> Solution:
        population = list(sorted(self._population, key=lambda x: x.fitness))
        if self.minimisation:
            return population[0]
        return population[-1]