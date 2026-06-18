from iohblade.solution import Solution
from iohblade.methods.moeh.population import Population

def test_population_accepts_valid_solution():
    s = Solution()

    s.set_scores(
        0.72,
        "Scored 0.72, best known solution is 0.912."
    )

    p = Population(10)
    p.append(s)

    assert len(p) == 1
    

def test_population_rejects_invalid_solution():
    s = Solution()

    p = Population(10)

    p.append(s)

    assert len(p) == 0