from iohblade.experiment import Experiment
from iohblade import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.problems import BBOB_SBOX
from iohblade.loggers import ExperimentLogger
import ioh
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    #llm1 = OpenAI_LLM(api_key,"gpt-4.1-2025-04-14") #Done
    llm1 = Gemini_LLM(api_key_gemini, "gemini-2.0-flash") #Failed partly #running 3/4 in BBOB-4, 5/6 in BBOB-5, rest is in 1/2 BBOB-1 folder.
    #llm3 = Ollama_LLM("qwen2.5-coder:32b") #Failed
    #llm4 = Ollama_LLM("gemma3:27b") #Done
    #llm5 = OpenAI_LLM(api_key,"o4-mini-2025-04-16", temperature=1.0)
    #llm2 = Ollama_LLM("codestral")
    #llm3 = Ollama_LLM("qwen2.5-coder:14b") #qwen2.5-coder:14b, deepseek-coder-v2:16b
    #llm4 = Ollama_LLM("deepseek-coder-v2:16b")
    #llm5 = Gemini_LLM(api_key, "gemini-1.5-flash")
    budget = 100 #long budgets




    mutation_prompts1 = [
        "Refine the selected algorithm to improve it.",  # simplify mutation
    ]
    mutation_prompts2 = [
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]
    mutation_prompts3 = [
        "Refine and simplify the selected solution to improve it.",  # simplify mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]


    for llm in [llm1]: #, llm2, llm3, llm4, llm5, llm2
        #RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
        LLaMEA_method1 = LLaMEA(llm, budget=budget, name=f"LLaMEA (1+1)", mutation_prompts=mutation_prompts1, n_parents=1, n_offspring=1, elitism=True) 
        LLaMEA_method2 = LLaMEA(llm, budget=budget, name=f"LLaMEA (4+16)", mutation_prompts=mutation_prompts1, n_parents=4, n_offspring=16, elitism=True) 
        #LLaMEA_method2 = LLaMEA(llm, budget=budget, name=f"LLaMEA-2", mutation_prompts=mutation_prompts2, n_parents=4, n_offspring=12, elitism=False) 

        methods =  [LLaMEA_method1, LLaMEA_method2]#, LLaMEA_method2] # 

        # List containing function IDs we consider
        fids = [
            2, 4,
            6, 8,
            12, 14,
            18, 15,
            21, 23,
        ]
        ids = [1,1,1] # 3 reps with first instance
        
        training_instances = [(f, i) for f in fids for i in ids]
        test_instances = [(f, i) for f in fids for i in range(5, 16)]
        
        logger = ExperimentLogger("results/BBOB-BO")

        problems = []
        prob = BBOB_SBOX(training_instances=training_instances, test_instances=test_instances, dims=[5], budget_factor=20, eval_timeout=1200, name=f"BBOB", problem_type=ioh.ProblemClass.BBOB, full_ioh_log=False, ioh_dir=f"{logger.dirname}/ioh")
        
        role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

        lib_prompt = "As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries."

        prob.task_prompt = f"""{role_prompt}
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of noiseless functions. Your task is to write the optimization algorithm in Python code. 
The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
"""
        prob.format_prompt = """
Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
# Description: <short-description>
# Code: 
```python
<code>
```
"""
        problems.append(prob)

        
        experiment = Experiment(methods=methods, problems=problems, runs=5, show_stdout=False, exp_logger=logger, budget=budget) #normal run
        experiment() #run the experiment
