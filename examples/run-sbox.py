from iohblade.experiment import Experiment
from iohblade import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.problems import BBOB_SBOX
from iohblade.loggers import ExperimentLogger
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.0-flash"
    llm1 = Gemini_LLM(api_key, ai_model)
    llm2 = Ollama_LLM("codestral")
    llm3 = Ollama_LLM("qwen2.5-coder:14b") #qwen2.5-coder:14b, deepseek-coder-v2:16b
    llm4 = Ollama_LLM("deepseek-coder-v2:16b")
    llm5 = Gemini_LLM(api_key, "gemini-1.5-flash")
    budget = 50 #short budgets



    mutation_prompts = [
        "Refine and simplify the selected algorithm to improve it.", #simplify
    ]

    for llm in [llm1, llm2, llm3, llm4, llm5]: #llm1 , llm3, llm4
        #RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
        LLaMEA_method = LLaMEA(llm, budget=budget, name=f"LLaMEA-{llm.model}", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=12, elitism=False) 

        methods = [LLaMEA_method] #, LLaMEA_method3, LLaMEA_method4, LLaMEA_method5

        # List containing function IDs per group
        group_functions = [
            [], #starting at 1
            [1, 2, 3, 4, 5],      # Separable Functions
            [6, 7, 8, 9],         # Functions with low or moderate conditioning
            [10, 11, 12, 13, 14], # Functions with high conditioning and unimodal
            [15, 16, 17, 18, 19], # Multi-modal functions with adequate global structure
            [20, 21, 22, 23, 24]  # Multi-modal functions with weak global structure
        ]
        
        problems = []
        for fid in [2, 5, 13, 15, 21]: # a selection of single functions
            training_instances = [(fid, i) for i in range(1, 6)]
            test_instances = [(fid, i) for i in range(5, 16)] #10 test instances
            problems.append(BBOB_SBOX(training_instances=training_instances, test_instances=test_instances, dims=[5], budget_factor=2000, name=f"SBOX_COST_fid{fid}", specific_fid=fid))
        for group in range(1,6):
            training_instances = [(f, i) for f in group_functions[group] for i in range(1, 6)]
            test_instances = [(f, i) for f in group_functions[group] for i in range(5, 16)]
            problems.append(BBOB_SBOX(training_instances=training_instances, test_instances=test_instances, dims=[5], budget_factor=2000, name=f"SBOX_COST_group{group}", specific_group=group))

        logger = ExperimentLogger("results/SBOX")
        experiment = Experiment(methods=methods, problems=problems, llm=llm, runs=5, show_stdout=True, exp_logger=logger) #normal run
        experiment() #run the experiment
