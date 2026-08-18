from iohblade.llm import Ollama_LLM
from iohblade.experiment import Experiment, ExperimentLogger
from iohblade.methods import LLaMEA, MoEH_Method, LHNS_Method, MCTS_Method
from iohblade.benchmarks.gravitational_waves import UIFOProblemSolver, ConstrainedVoyagerSolver, VoyagerProblemSolver, VoyagerTuningProblemSolver

if __name__ == '__main__':
    llms = [
            Ollama_LLM('gemma4:latest'),
            Ollama_LLM('gpt-oss:20b'),
        ]

    logger = ExperimentLogger('results/GravitationalWavesDetectorOptimisation')

    methods = []
    problems = []

    for llm in llms:
        llamea = LLaMEA(llm=llm, budget=200, n_parents=4, n_offspring=4)
        llamea.name = f'{llamea.name}-{llm.model}'

        moeh = MoEH_Method(llm, budget=200, iterations=50, population_size=4)
        moeh.name = f'{moeh.name}-{llm.model}'

        lhns = LHNS_Method(llm, budget=200, method='vns')
        lhns.name = f'{lhns.name}-{llm.model}'

        mcts = MCTS_Method(llm, 200)
        mcts.name = f'{mcts.name}-{llm.model}'

        methods.extend([llamea, moeh, lhns, mcts])

    for problem in [UIFOProblemSolver, VoyagerProblemSolver, VoyagerTuningProblemSolver, ConstrainedVoyagerSolver]:
        problems.append(problem())

    exp = Experiment(
            methods,
            problems,
            2,
            200,
            show_stdout=True,
            exp_logger=logger,
            n_jobs=1
        )

    exp()
