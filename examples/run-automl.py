from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA, RandomSearch, EoH, ReEvo
from iohblade.loggers import ExperimentLogger
from iohblade.problems import AutoML
import numpy as np
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    llm1 = Gemini_LLM(api_key_gemini, "gemini-2.0-flash")
    #llm5 = OpenAI_LLM(api_key,"o4-mini-2025-04-16", temperature=1.0)
    budget = 50

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
        "Refine and simplify the selected algorithm to improve it.", #simplify
    ]

    for llm in [llm1]:
        #LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=12, elitism=True)
        ReEvo_method = ReEvo(llm, budget=budget, name="ReEvo", output_path="results/automl-breast-cancer")
        methods = [ReEvo_method] #LLaMEA_method, EoH_method
        logger = ExperimentLogger("results/automl-breast-cancer")
        problems = [AutoML()]
        experiment = Experiment(methods=methods, problems=problems, runs=1, show_stdout=True, exp_logger=logger) #normal run
        experiment() #run the experiment