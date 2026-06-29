from iohblade.solution import Solution
from iohblade.methods.moeh_method.population import Population

def test_population_accepts_valid_solution():
    s = Solution()

    s.set_scores(
        0.72,
        "Scored 0.72, best known solution is 0.912."
    )

    p = Population(10, True)
    p.append(s)

    assert len(p) == 1
    

def test_population_rejects_invalid_solution():
    s = Solution()

    p = Population(10, True)

    p.append(s)

    assert len(p) == 0

def test_population_selection_matrix():
    p = Population(3, True)
    solutions = [Solution("ABC"), Solution("ABD"), Solution("BATMAN")]
    solutions[0].set_scores(1, "Score is 1")
    solutions[1].set_scores(2, "Score is 2")
    solutions[2].set_scores(3, "Score is 3")

    for solution in solutions:
        p.append(solution)
    
    next_gen = p.parent_selection(2, True)
    for individual in next_gen:
        assert individual in p

def test_population_management():
    p = Population(3, True)
    solutions = [Solution("ABC"), Solution("ABD"), Solution("BATMAN")]
    solutions[0].set_scores(1, "Score is 1")
    solutions[1].set_scores(2, "Score is 2")
    solutions[2].set_scores(3, "Score is 3")

    for solution in solutions:
        p.append(solution)
    
    sorted_population = p.population_management(True)
    solutions = sorted(solutions, key=lambda x: x.fitness)
    for i in range(len(solutions)):
        ref_dict = solutions[i].__dict__
        source_dict = sorted_population[i].__dict__
        for key in source_dict:
            if key != 'id':
                assert source_dict[key] == ref_dict[key]