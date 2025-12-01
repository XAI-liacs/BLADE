from os import getenv

from iohblade.llm import Ollama_LLM
from iohblade.llm import Gemini_LLM
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.methods import MCTS_Method
from iohblade.benchmarks.geometry import get_min_max_dist_ratio_problem

def main():
    problem = get_min_max_dist_ratio_problem(use_best=False)[0]

    key = getenv("GOOGLE_API_KEY")
    llm = Gemini_LLM(key, "gemini-2.0-flash")

    # llm = Ollama_LLM()
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
    best_solution = mcts_method.mcts_instance.best_solution
    print("id: ", best_solution.id)
    print("description:", best_solution.description)
    print("code:\v", best_solution.code)
    print("fitness:\v", best_solution.fitness)
    
if __name__ == "__main__":
    main()