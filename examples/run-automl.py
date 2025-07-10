from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA, RandomSearch, EoH, ReEvo
from iohblade.loggers import ExperimentLogger
from iohblade.problems import AutoML
import numpy as np
import os
import logging

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    #llm1 = Gemini_LLM(api_key_gemini, "gemini-1.5-flash")
    llm2 = OpenAI_LLM(api_key,"gpt-4.1-nano-2025-04-14", temperature=1.0)
    budget = 10

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
        "Refine and simplify the selected algorithm to improve it.", #simplify
    ]

    for llm in [llm2]:
        LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=2, n_offspring=4, elitism=True)
        ReEvo_method = ReEvo(llm, budget=budget, name="ReEvo", output_path="results/automl-breast-cancer", pop_size=2, init_pop_size=4)
        EoH_method = EoH(llm, budget=budget, name="EoH", output_path="results/automl-breast-cancer")
        methods = [EoH_method, ReEvo_method, LLaMEA_method] 
        logger = ExperimentLogger("results/automl-breast-cancer")
        problems = [AutoML()]
        experiment = Experiment(methods=methods, problems=problems, runs=1, show_stdout=False, exp_logger=logger, budget=budget, n_jobs=3) #normal run

        experiment() #run the experiment