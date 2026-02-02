from iohblade.experiment import Experiment
from iohblade.problems import BenchDesign
from iohblade.llm import Gemini_LLM, Ollama_LLM, Dummy_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
import numpy as np
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    ai_model = "gpt-5-mini"
    llm = OpenAI_LLM(api_key, ai_model, temperature=1.0)
    budget = 100


    mutation_prompts = [
        "Refine one of the benchmarks to improve the diversity of optimizer rankings.", 
        "Improve the benchmarking suite by sligytly altering the functions.",
        "Replace one of the benchmark functions with a completely new one.",
        "Change the dimensionality of some functions."
    ]
    
    LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=6, elitism=True)

    methods = [LLaMEA_method] #, LLaMEA_method4, LLaMEA_method5]#, LLaMEA_method4, LLaMEA_method5]
    logger = ExperimentLogger("results/benchdesign")
    problem = BenchDesign(logger=logger,
        name="BenchDesign",
        eval_timeout=1200,
        budget=500,
        repeats=5)
    
    example_code = """
import numpy as np


def sphere(x):
    return float(np.sum(x ** 2))


def ridge(x):
    return float(np.sum(np.abs(x)) + 0.1 * np.sum(x ** 2))


problems = {
    "sphere": sphere,
    "ridge": ridge,
}

meta_dims = {
    "sphere": 5,
    "ridge": 12,
}"""

    # test problem
    # solution = Solution(example_code,"test")
    # problem.evaluate(solution)
    # print("Score:", solution.fitness)
    # print("Feedback:", solution.feedback)


    experiment = Experiment(methods=methods, problems=[problem], runs=2, seeds=[1,2], budget=budget, show_stdout=True, exp_logger=logger) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


