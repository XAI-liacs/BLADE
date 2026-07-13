from operator import imod

from iohblade.llm import Dummy_LLM
from iohblade.problem import Problem
from iohblade.methods import LLaMEA, LHNS_Method, MCTS_Method, EoH
from iohblade.methods.reevo import ReEvo
from iohblade.benchmarks.analysis import AutoCorrIneq1, AutoCorrIneq2



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
