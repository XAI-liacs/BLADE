from os import getenv

from iohblade.llm import Gemini_LLM
from iohblade.methods import LHNS_Method
from iohblade.experiment import Experiment, ExperimentLogger
from iohblade.benchmarks.geometry import get_min_max_dist_ratio_problem


def main():
    api_key = getenv('GOOGLE_API_KEY')
    
    problem = get_min_max_dist_ratio_problem(use_best=False)[0]
    llm = Gemini_LLM(api_key)
    
    # method1 = LHNS_Method(llm, budget=20, method='vns', minimisation=problem.minimisation)
    # method2 = LHNS_Method(llm, budget=20, method='ils', minimisation=problem.minimisation)
    method3 = LHNS_Method(llm, budget=20, method='ts', minimisation=problem.minimisation)
    
    logger = ExperimentLogger(f"results/{problem.name}")
    experiment = Experiment(
        [method3],
        [problem],
        runs=1,
        budget=20,
        exp_logger=logger,
        show_stdout=True
    )

    experiment()


if __name__ == "__main__":
    main()