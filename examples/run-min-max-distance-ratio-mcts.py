from os import getenv

from iohblade.llm import Gemini_LLM
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.methods import MCTS_Method
from iohblade.benchmarks.geometry import get_min_max_dist_ratio_problem

def main():
    problem = get_min_max_dist_ratio_problem(use_best=False)[0]

    key = getenv("GOOGLE_API_KEY")
    print(key)
    llm = Gemini_LLM(key, "gemini-2.0-flash")
    mcts_method = MCTS_Method(llm, 100, maximisation=not problem.minimisation)
    logger = ExperimentLogger(f"results/{problem.name}")
    Experiment(
        [mcts_method],
        [problem],
        1,
        100,
        show_stdout=True,
        exp_logger=logger
    )()
    
if __name__ == "__main__":
    main()