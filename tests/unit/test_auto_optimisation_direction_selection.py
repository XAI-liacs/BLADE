from iohblade.llm import Dummy_LLM
from iohblade import Problem, Solution
from iohblade.methods import LLaMEA, LHNS_Method, MCTS_Method, EoH
from iohblade.methods.reevo import ReEvo
from iohblade.benchmarks.analysis import AutoCorrIneq1, AutoCorrIneq2
from tests.unit.test_moeh import DummyProblemWorks
import random



def test_llamea_works_properly(monkeypatch):

    def call_without_env(self, solution, logger=None):
        print('Call without env called.')
        return self.evaluate(solution)
    monkeypatch.setattr(Problem, '__call__', call_without_env)

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
        print('Call without env called.')
        return self.evaluate(solution)
    monkeypatch.setattr(Problem, '__call__', call_without_env)

    llm = Dummy_LLM()
    ac1 = AutoCorrIneq1()
    ac2 = AutoCorrIneq2()
    lhns = LHNS_Method(llm, 5, 'vns')

    _ = lhns(ac1)
    assert lhns.lhns_instance.minimisation == ac1.minimisation
    _ = lhns(ac2)
    assert lhns.lhns_instance.minimisation == ac2.minimisation

def test_mcts_ahd_works_properly(monkeypatch):

    def call_without_env(self, solution, logger=None):
        print('Call without env called.')
        return self.evaluate(solution)
    monkeypatch.setattr(Problem, '__call__', call_without_env)

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

    s = Solution('''
def RandomOptimser():
    return random.random()
''')

    fitness = bp.evaluate(s.code)
    assert (fitness >= 0) and (dp.minimisation)

    dp = DummyProblemWorks(minimisation=False)
    bp = _BladeProblemAdapter(dp)
    fitness = bp.evaluate(s.code)
    assert (fitness <= 0) and (not dp.minimisation)


def test_revo_works_properly():

    llm = Dummy_LLM()
    reevo = ReEvo(llm, 5)

    problem = DummyProblemWorks(minimisation=True)

    soln = reevo(problem)


    assert soln.fitness == min(problem.all_scores)

    problem = DummyProblemWorks(minimisation=False)
    soln = reevo(problem)
    assert soln.fitness == max(problem.all_scores)
