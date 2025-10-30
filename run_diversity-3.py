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

from iohblade.experiment import MA_BBOB_Experiment, Experiment
from iohblade.problems import BBOB_SBOX
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
import numpy as np
import ioh
import os

from iohblade.utils import code_compare
import lizard

def analyse_complexity(solution):
    """
    Analyzes the solution complexity of a Python code snippet using the lizard library.

    Args:
        code (str): The Python code to analyze.

    Returns:
        dict: A dictionary containing statistics such as mean and total cyclomatic complexity,
        token counts, and parameter counts.
    """
    code = solution.code
    i = lizard.analyze_file.analyze_source_code(f"algorithm.py", code)
    complexities = []
    token_counts = []
    parameter_counts = []
    nlocs = []
    for f in i.function_list:
        complexities.append(f.__dict__["cyclomatic_complexity"])
        token_counts.append(f.__dict__["token_count"])
        parameter_counts.append(len(f.__dict__["full_parameters"]))
        nlocs.append(f.__dict__["nloc"])
    return [
        np.sum(complexities),
        np.mean(token_counts),
        np.sum(token_counts),
        np.sum(parameter_counts),
        np.sum(nlocs),
    ]

def ast_distance(solution1, solution2):
    # AST distance calculation, returns a ratio between 0 and 1
    return code_compare(solution1.code, solution2.code)

def fitness_behavioral_distance(solution1, solution2):
    # Placeholder for behavioral distance calculation
    # In practice, this function would evaluate the behavior of the solutions and compute a distance metric
    return np.random.rand()  # Replace with actual behavioral distance computation

def behavior_descriptor(solution):
    # Placeholder for behavior descriptor extraction
    # In practice, this function would extract meaningful features from the solution's behavior
    return np.random.rand(10)  # Replace with actual behavior descriptor extraction

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
    #llm1 = Gemini_LLM(api_key_google, "gemini-2.5-flash")
    llm = OpenAI_LLM(api_key_openai, "gpt-5-mini-2025-08-07", temperature=1.0)
    #llm3 = Claude_LLM(api_key_claude, "claude-sonnet-4-5-20250929", temperature=1.0)

    budget = 200

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    

    #for llm in [llm1]:#, llm2]:
    #RS = RandomSearch(llm, budget=budget) 
    LLaMEA_1 = LLaMEA(llm, budget=budget, name="8+8 ES", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True)
    
    LLaMEA_novelty_ast = LLaMEA(llm, budget=budget, name="8+8 NS-code", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="novelty", distance_metric=ast_distance, novelty_archive_size=100, novelty_k=5)
    #LLaMEA_novelty_fitness = LLaMEA(llm, budget=budget, name="8+8 NS", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="novelty", distance_metric=fitness_behavioral_distance, novelty_archive_size=100, novelty_k=5)

    LLaMEA_fitness_sharing_ast = LLaMEA(llm, budget=budget, name="8+8 FS", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="sharing", distance_metric=ast_distance, adaptive_niche_radius=True)
    LLaMEA_fitness_clearing_ast = LLaMEA(llm, budget=budget, name="8+8 FC", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="clearing", distance_metric=ast_distance, adaptive_niche_radius=True)

    #LLaMEA_MAP_elites = LLaMEA(llm, budget=budget, name="8+8 MAP", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="map_elites", distance_metric=ast_distance, novelty_archive_size=100, novelty_k=5)


    methods = [LLaMEA_1, LLaMEA_novelty_ast, LLaMEA_fitness_sharing_ast, LLaMEA_fitness_clearing_ast]#LLaMEA_gemini, LLaMEA_gpt, LLaMEA_multi1, LLaMEA_multi2] #, LLaMEA_method4, LLaMEA_method5]#, LLaMEA_method4, LLaMEA_method5]
    logger = ExperimentLogger("results/SBOX-diversity")

    # List containing function IDs we consider
    training_fids = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]
    testing_fids = [2, 4, 5, 7, 9, 11, 12, 14, 16, 18, 19, 20, 22, 24]

    training_instances = [(f, i) for f in training_fids for i in range(1, 6)]
    test_instances = [(f, i) for f in testing_fids for i in range(1, 6)]

    problems = []
    problems.append(
        BBOB_SBOX(
            training_instances=training_instances,
            test_instances=test_instances,
            dims=[10],
            budget_factor=2000,
            eval_timeout=600,
            name=f"SBOX",
            problem_type=ioh.ProblemClass.SBOX,
            full_ioh_log=False,
            ioh_dir=f"{logger.dirname}/ioh",
        )
    )

    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=5,
        show_stdout=False,
        exp_logger=logger,
        budget=budget,
    )  # normal run

    experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=2000, budget=budget, eval_timeout=270, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


