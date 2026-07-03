from iohblade.methods import LLaMEA
from iohblade.methods import MCTS_Method
from iohblade.benchmarks.analysis import get_analysis_problems
from iohblade.benchmarks.analysis import AutoCorrIneq2

from iohblade.llm import Ollama_LLM
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger


if __name__ == '__main__':
    problems = get_analysis_problems(False)

    llms = [
        Ollama_LLM('gemma4:latest'),
        Ollama_LLM('gpt-oss:20b'),
        Ollama_LLM('deepcoder:14b')
    ]

    logger = ExperimentLogger('AutoCorrelation_DB')

    methods = []
    minimisation_problems = []

    for llm in llms:
        for problem in problems:
            if isinstance(problem, AutoCorrIneq2):
                llamea = LLaMEA(
                        llm=llm,
                        budget=200,
                        n_parents=4,
                        n_offspring=4,
                        minimisation=problem.minimisation
                    )
                llamea.name = llamea.name + llm.model

                mcts = MCTS_Method(
                        llm=llm,
                        budget=200,
                        maximisation=not problem.minimisation
                    )
                
                mcts.name = mcts.name + llm.model

                methods.append( 
                    mcts
                )

                minimisation_problems.append(problem)
    
    exp = Experiment(
        methods,
        minimisation_problems,
        2,
        200,
        show_stdout=True,
        exp_logger=logger,
    )

    exp()

    maximisation_problems = []
    for llm in llms:
        for problem in problems:
            if not isinstance(problem, AutoCorrIneq2):
                methods.append(
                    LLaMEA(
                        llm=llm,
                        budget=200,
                        n_parents=4,
                        n_offspring=4,
                        minimisation=problem.minimisation
                    )
                )

                methods.append( 
                    MCTS_Method(
                        llm=llm,
                        budget=200,
                        maximisation=not problem.minimisation
                    )
                )

                maximisation_problems.append(problem)
    
    exp = Experiment(
        methods,
        maximisation_problems,
        2,
        200,
        show_stdout=True,
        exp_logger=logger,
    )

    exp()

