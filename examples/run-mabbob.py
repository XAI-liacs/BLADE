from llamea import Operator

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, Dummy_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
import numpy as np
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.0-flash"
    llm1 = Gemini_LLM(api_key, ai_model)
    llm2 = Ollama_LLM("codestral")
    budget = 10


    # Mutation operators (llamea >= 1.4); equal weights pick each one equally often.
    operators1 = [
        Operator(
            prompt="Refine the strategy of the selected algorithm to improve it.",
            name="refine",
            weight=1.0,
            number_of_parents=1,
        ),
    ]
    operators2 = [
        Operator(
            prompt="Generate a new algorithm that is different from the algorithms you have tried before.",
            name="new_random",
            weight=1.0,
            number_of_parents=1,
        ),
    ]
    operators3 = [
        Operator(
            prompt="Refine and simplify the selected algorithm to improve it.",
            name="simplify",
            weight=1.0,
            number_of_parents=1,
        ),
    ]
    operators4 = [
        Operator(
            prompt="Refine the strategy of the selected solution to improve it.",
            name="refine",
            weight=1.0,
            number_of_parents=1,
        ),
        Operator(
            prompt="Generate a new algorithm that is different from the algorithms you have tried before.",
            name="new_random",
            weight=1.0,
            number_of_parents=1,
        ),
    ]
    operators5 = [
        Operator(
            prompt="Refine the strategy of the selected solution to improve it.",
            name="refine",
            weight=1.0,
            number_of_parents=1,
        ),
        Operator(
            prompt="Generate a new algorithm that is different from the algorithms you have tried before.",
            name="new_random",
            weight=1.0,
            number_of_parents=1,
        ),
        Operator(
            prompt="Refine and simplify the selected algorithm to improve it.",
            name="simplify",
            weight=1.0,
            number_of_parents=1,
        ),
    ]

    for llm in [llm1]:#, llm2]:
        #RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
        LLaMEA_method1 = LLaMEA(llm, budget=budget, name="LLaMEA-1", operators=operators1, n_parents=1, n_offspring=1, elitism=False)
        # LLaMEA_method2 = LLaMEA(llm, budget=budget, name="LLaMEA-2", operators=operators2, n_parents=4, n_offspring=12, elitism=False)
        # LLaMEA_method3 = LLaMEA(llm, budget=budget, name="LLaMEA-3", operators=operators3, n_parents=4, n_offspring=12, elitism=False)
        # LLaMEA_method4 = LLaMEA(llm, budget=budget, name="LLaMEA-4", operators=operators4, n_parents=4, n_offspring=12, elitism=False)
        # LLaMEA_method5 = LLaMEA(llm, budget=budget, name="LLaMEA-5", operators=operators5, n_parents=4, n_offspring=12, elitism=False) 

        methods = [LLaMEA_method1] #, LLaMEA_method4, LLaMEA_method5]#, LLaMEA_method4, LLaMEA_method5]
        logger = ExperimentLogger("results/MA-BBOB-test")
        experiment = MA_BBOB_Experiment(methods=methods, runs=2, seeds=[4,7], dims=[5], budget_factor=200, budget=10, eval_timeout=60, show_stdout=True, exp_logger=logger) #normal run
        experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


