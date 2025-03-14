from blade.experiment import Experiment
from blade.llm import Gemini_LLM, Ollama_LLM
from blade.methods import LLaMEA, RandomSearch
from blade.problems import BBOB_SBOX
import os

api_key = os.getenv("GEMINI_API_KEY")
ai_model = "gemini-2.0-flash"
llm1 = Gemini_LLM(api_key, ai_model)
llm2 = Ollama_LLM("codestral")
llm3 = Ollama_LLM("gemma3:27b")
budget = 50 #short budgets



mutation_prompts = [
    "Refine and simplify the selected algorithm to improve it.", #simplify
]

for llm in [llm2, llm1, llm3]:
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

    # 10 problems, 3 llms, 5 runs, 20 llm calls per run = 30 * 5 runs = 150 runs * 20 llm calls = 3000 llm calls
    experiment = Experiment(methods=methods, problems=problems, llm=llm, runs=5, show_stdout=True, log_dir="results/SBOX") #normal run
    experiment() #run the experiment
