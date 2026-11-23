import random

from iohblade import Problem, Solution
from iohblade.benchmarks.analysis import AutoCorrIneq1, AutoCorrIneq2
from iohblade.llm import Dummy_LLM
from iohblade.methods import EoH, LHNS_Method, LLaMEA, MCTS_Method
from iohblade.methods.reevo import ReEvo
from tests.unit.test_moeh import DummyProblemWorks


def test_llamea_works_properly(monkeypatch):

    def call_without_env(self, solution, logger=None):
        print("Call without env called.")
        return self.evaluate(solution)

    monkeypatch.setattr(Problem, "__call__", call_without_env)

    llm = Dummy_LLM()
    ac1 = AutoCorrIneq1()
    ac2 = AutoCorrIneq2()
    llamea = LLaMEA(llm, budget=10)

    _ = llamea(ac1)
    assert llamea.llamea_instance.minimization == ac1.minimisation
    _ = llamea(ac2)
    assert llamea.llamea_instance.minimization == ac2.minimisation


def test_lhns_works_properly(monkeypatch):

    def call_without_env(self, solution, logger=None):
        print("Call without env called.")
        return self.evaluate(solution)

    monkeypatch.setattr(Problem, "__call__", call_without_env)

    llm = Dummy_LLM()
    ac1 = AutoCorrIneq1()
    ac2 = AutoCorrIneq2()
    lhns = LHNS_Method(llm, 5, "vns")

    _ = lhns(ac1)
    assert lhns.lhns_instance.minimisation == ac1.minimisation
    _ = lhns(ac2)
    assert lhns.lhns_instance.minimisation == ac2.minimisation


def test_mcts_ahd_works_properly(monkeypatch):

    def call_without_env(self, solution, logger=None):
        print("Call without env called.")
        return self.evaluate(solution)

    monkeypatch.setattr(Problem, "__call__", call_without_env)

    llm = Dummy_LLM()
    ac1 = AutoCorrIneq1()
    ac2 = AutoCorrIneq2()
    mcts = MCTS_Method(llm, 5)

    _ = mcts(ac1)
    assert not mcts.mcts_instance.maximisation == ac1.minimisation
    _ = mcts(ac2)
    assert not mcts.mcts_instance.maximisation == ac2.minimisation


def test_eoh_works_properly():

    from iohblade.methods.eoh import _BladeProblemAdapter

    dp = DummyProblemWorks()
    bp = _BladeProblemAdapter(dp)

    s = Solution("""
def RandomOptimser():
    return random.random()
""")

    fitness = bp.evaluate(s.code)
    assert (fitness >= 0) and (dp.minimisation)

    dp = DummyProblemWorks(minimisation=False)
    bp = _BladeProblemAdapter(dp)
    fitness = bp.evaluate(s.code)
    assert (fitness <= 0) and (not dp.minimisation)


class _SeedSucceeds(DummyProblemWorks):
    """DummyProblemWorks whose first evaluation (ReEvo's seed) never fails.

    ReEvo stops when its seed function is invalid, and a non-finite fitness
    marks an individual invalid.
    """

    def evaluate(self, solution):
        if self.iteration == 0:
            rate, self.failure_rate = self.failure_rate, 0.0
            try:
                return super().evaluate(solution)
            finally:
                self.failure_rate = rate
        return super().evaluate(solution)


def test_revo_works_properly():

    llm = Dummy_LLM()
    reevo = ReEvo(llm, 5)

    problem = _SeedSucceeds(minimisation=True, failure_rate=0.6)
    soln = reevo(problem)

    assert soln.fitness == min(problem.all_scores)

    problem = DummyProblemWorks(minimisation=False)
    soln = reevo(problem)
    assert soln.fitness == max(problem.all_scores)
