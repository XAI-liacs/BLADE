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

    logger = ExperimentLogger('results/AutoCorrelation_2_DB')

    methods = []
    minimisation_problems = []

    problem_1 = [problem for problem in problems if isinstance(problem, AutoCorrIneq2)]

    for llm in llms:
            llamea = LLaMEA(
                    llm=llm,
                    budget=200,
                    n_parents=4,
                    n_offspring=4,
                    minimisation=problem_1[0].minimisation,
                )
            llamea.name = f'{llamea.name}-{llm.model}'

            mcts = MCTS_Method(
                    llm=llm,
                    budget=200,
                    maximisation=not problem_1[0].minimisation
                )
            
            mcts.name = f'{mcts.name}-{llm.model}'

            methods.extend( 
                [mcts, llamea]
            )

    print(f'Number of methods added: {len(methods)}.')
    
    exp = Experiment(
        methods,
        problem_1,
        2,
        200,
        show_stdout=True,
        exp_logger=logger,
    )

    exp()

    logger = ExperimentLogger('results/AutoCorrelation_1_3_DB')
    maximisation_problems = [problem for problem in problems if not isinstance(problem, AutoCorrIneq2)]

    for llm in llms:
        methods.append(
            LLaMEA(
                llm=llm,
                budget=200,
                n_parents=4,
                n_offspring=4,
                minimisation=problem2[0].minimisation
            )
        )

        methods.append( 
            MCTS_Method(
                llm=llm,
                budget=200,
                maximisation=not problem2[0].minimisation
            )
        )
    
    exp = Experiment(
        methods,
        problem2,
        2,
        200,
        show_stdout=True,
        exp_logger=logger,
    )

    exp()

