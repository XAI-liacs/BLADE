from iohblade.solution import Solution

class Population:
    def __init__(self, size) -> None:
        self.size = size
        self._population: list[Solution] = []

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
