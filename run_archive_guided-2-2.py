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
    api_key_google = os.getenv("GEMINI_API_KEY")
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_claude = os.getenv("CLAUDE_API_KEY")



    llm = Gemini_LLM(api_key_google, "gemini-2.0-flash-lite")
    #llm3 = Claude_LLM(api_key_claude, "claude-sonnet-4-5-20250929", temperature=1.0)

    budget = 100 # test run (25 iterations of 8 algs)

    DEBUG = False
    if DEBUG:
        budget = 20

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]


    mcts_method = MCTS_Method(llm, maximisation=True, budget=budget)
    lhns_method = LHNS_Method(llm, minimisation=False, method="vns", budget=budget)

    LLaMEA_1 = LLaMEA(llm, budget=budget, name="ES", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=16, elitism=True)
    LLaMEA_2 = LLaMEA(llm, budget=budget, name="ES-guided", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=16, elitism=True, feature_guided_mutation=True)
    

    methods = [LLaMEA_1, LLaMEA_2, mcts_method, lhns_method] 

    training_instances = range(0,10)

    if DEBUG:
        logger = ExperimentLogger("results/MABBOB_guided_gemini_debug")
    else:
        logger = ExperimentLogger("results/MABBOB_guided_gemini")

    if DEBUG:
        experiment = MA_BBOB_Experiment(methods=methods, training_instances=[0,1], runs=1, seeds=[1], dims=[2], budget_factor=200, budget=budget, eval_timeout=60, show_stdout=True, log_stdout=False, exp_logger=logger, n_jobs=5) #normal run
    else:
        experiment = MA_BBOB_Experiment(methods=methods, training_instances=training_instances, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=5000, budget=budget, eval_timeout=600, show_stdout=False, log_stdout=True, exp_logger=logger, n_jobs=10) #normal run


    #experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=2000, budget=budget, eval_timeout=270, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


