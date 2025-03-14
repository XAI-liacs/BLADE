from blade.experiment import MA_BBOB_Experiment
from blade.llm import Gemini_LLM, Ollama_LLM
from blade.methods import LLaMEA, RandomSearch
import os

api_key = os.getenv("GEMINI_API_KEY")
ai_model = "gemini-2.0-flash"
llm1 = Gemini_LLM(api_key, ai_model)
llm2 = Ollama_LLM("codestral")
budget = 100


mutation_prompts1 = [
    "Refine the strategy of the selected algorithm to improve it.",  # small mutation
]
mutation_prompts2 = [
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
]
mutation_prompts3 = [
    "Refine and simplify the selected algorithm to improve it.", #simplify
]
mutation_prompts4 = [
    "Refine the strategy of the selected solution to improve it.",  # small mutation
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
]
mutation_prompts5 = [
    "Refine the strategy of the selected solution to improve it.",  # small mutation
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    "Refine and simplify the selected algorithm to improve it.", #simplify
]

for llm in [llm1]:#, llm2]:
    #RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
    LLaMEA_method1 = LLaMEA(llm, budget=budget, name="LLaMEA-1", mutation_prompts=mutation_prompts1, n_parents=4, n_offspring=12, elitism=False)
    LLaMEA_method2 = LLaMEA(llm, budget=budget, name="LLaMEA-2", mutation_prompts=mutation_prompts2, n_parents=4, n_offspring=12, elitism=False)
    LLaMEA_method3 = LLaMEA(llm, budget=budget, name="LLaMEA-3", mutation_prompts=mutation_prompts3, n_parents=4, n_offspring=12, elitism=False)
    LLaMEA_method4 = LLaMEA(llm, budget=budget, name="LLaMEA-4", mutation_prompts=mutation_prompts4, n_parents=4, n_offspring=12, elitism=False)
    LLaMEA_method5 = LLaMEA(llm, budget=budget, name="LLaMEA-5", mutation_prompts=mutation_prompts5, n_parents=4, n_offspring=12, elitism=False) 

    methods = [LLaMEA_method3, LLaMEA_method3, LLaMEA_method4, LLaMEA_method5]
    experiment = MA_BBOB_Experiment(methods=methods, llm=llm, runs=5, dims=[5], budget_factor=2000, eval_timeout=60, show_stdout=True) #normal run
    experiment() #run the experiment



#MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


