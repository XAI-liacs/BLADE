from iohblade.experiment import MA_BBOB_Experiment, Experiment
from iohblade.problems import BBOB_SBOX
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch, MCTS_Method, LHNS_Method, ReEvo
from iohblade.loggers import ExperimentLogger
import numpy as np
import ioh
import os

from iohblade.utils import code_compare
import lizard



if __name__ == "__main__": # prevents weird restarting behaviour
    api_key_openai = os.getenv("OPENAI_API_KEY")


    llm = OpenAI_LLM(api_key_openai, "gpt-5-mini-2025-08-07", temperature=1.0)

    budget = 200 # test run (25 iterations of 8 algs)

    DEBUG = False
    if DEBUG:
        budget = 20

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]


    mcts_method = MCTS_Method(llm, maximisation=True, budget=budget)
    lhns_method = LHNS_Method(llm, minimisation=False, method="vns", budget=budget)

    methods = [mcts_method, lhns_method] 

    training_instances = range(0,10)

    if DEBUG:
        logger = ExperimentLogger("results/MABBOB_guided_baselines_debug")
    else:
        logger = ExperimentLogger("results/MABBOB_guided_baselines")

    if DEBUG:
        experiment = MA_BBOB_Experiment(methods=methods, training_instances=[0,1], runs=1, seeds=[1], dims=[2], budget_factor=200, budget=budget, eval_timeout=60, show_stdout=True, log_stdout=False, exp_logger=logger, n_jobs=5) #normal run
    else:
        experiment = MA_BBOB_Experiment(methods=methods, training_instances=training_instances, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=5000, budget=budget, eval_timeout=600, show_stdout=False, log_stdout=True, exp_logger=logger, n_jobs=5) #normal run


    experiment() #run the experiment

