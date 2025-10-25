## Set up a diversity benchmarking experiment. We can already refer to earlier experiments for setting prompts and 4+4 config.
# Try the following methods:
#   - LLaMea 4+4 with one LLM.
#   - LLaMEA 4+4 with a combination of 3 LLMs.
#   - LLaMEA with adaptive fitness sharing using AST distance metric.
#   - LLaMEA with adaptive fitness sharing using behavioural distance metric.
#   - LLaMEA with fitness clearing using AST distance metric.
#   - LLaMEA with fitness clearing using behavioural distance metric.
#   - LLaMEA with MAP-Elites using AST distance metric.
#   - LLaMEA with MAP-Elites using behavioural distance metric.
#   - LLaMEA with Novelty Search using AST distance metric.
#   - LLaMEA with Novelty Search using behavioural distance metric.

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
import numpy as np
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key_google = os.getenv("GEMINI_API_KEY")
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_claude = os.getenv("CLAUDE_API_KEY")


    #. lets first experiment with local models.
    # qwen3-coder:30b, gemma3:27b, llama3.2:3b

    # first experiments with 3x 1 llm, 2x2 llms, 3 llms

    #llm_qwen = Ollama_LLM("qwen3-coder:30b")
    #llm_gemma3 = Ollama_LLM("gemma3:27b")
    #llm_llama = Ollama_LLM("llama3.2:3b")

    #ai_model = "gemini-2.0-flash"
    llm1 = Gemini_LLM(api_key_google, "gemini-2.5-flash")
    llm2 = OpenAI_LLM(api_key_openai, "gpt-5-mini-2025-08-07", temperature=1.0)
    llm3 = Claude_LLM(api_key_claude, "claude-sonnet-4-5-20250929", temperature=1.0)

    llm4 = Multi_LLM([llm1, llm2])
    llm5 = Multi_LLM([llm1, llm2, llm3])
    budget = 200

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    #for llm in [llm1]:#, llm2]:
    #RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
    LLaMEA_gemini = LLaMEA(llm1, budget=budget, name="gemini-2.5-flash", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=4, elitism=True)
    LLaMEA_gpt = LLaMEA(llm2, budget=budget, name="gpt-5-nano", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=4, elitism=True)
    LLaMEA_claude = LLaMEA(llm3, budget=budget, name="claude-4.5", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=4, elitism=True)
    
    LLaMEA_multi1 = LLaMEA(llm4, budget=budget, name="multi-2", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=4, elitism=True)
    LLaMEA_multi2 = LLaMEA(llm5, budget=budget, name="multi-3", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=4, elitism=True)
    
    
    methods = [LLaMEA_claude, LLaMEA_gemini, LLaMEA_gpt, LLaMEA_multi1, LLaMEA_multi2]#LLaMEA_gemini, LLaMEA_gpt, LLaMEA_multi1, LLaMEA_multi2] #, LLaMEA_method4, LLaMEA_method5]#, LLaMEA_method4, LLaMEA_method5]
    logger = ExperimentLogger("results/MA-BBOB-multi-full")
    experiment = MA_BBOB_Experiment(methods=methods, runs=3, seeds=[1,2,3], dims=[5], budget_factor=2000, budget=budget, eval_timeout=180, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


