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
    # take the mean pairwise difference between auc scores as behavioral distance
    aucs1 = solution1.get_metadata("aucs")
    aucs2 = solution2.get_metadata("aucs")
    if aucs1 is None or aucs2 is None:
        return 0.0  # minimum distance if no behavioral data is available (algorithm did not run)
    aucs1 = np.array(aucs1)
    aucs2 = np.array(aucs2)
    distance = np.mean(np.abs(aucs1 - aucs2))
    return distance


def behavior_descriptor(solution):
    # group the auc scores per BBOB function group as descriptors.
    aucs = solution.get_metadata("aucs")
    if aucs is None:
        return np.zeros(5)
    group1 = aucs[:6] #2 functions, 3 instances each
    group2 = aucs[6:12] #2 functions, 3 instances each
    group3 = aucs[12:18] #2 functions, 3 instances each
    group4 = aucs[18:24] #2 functions, 3 instances each
    group5 = aucs[24:] #2 functions, 3 instances each
    return [np.mean(group1), np.mean(group2), np.mean(group3), np.mean(group4), np.mean(group5)]

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

    budget = 200 # test run (25 iterations of 8 algs)

    DEBUG = False
    if DEBUG:
        budget = 10

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    #for llm in [llm1]:#, llm2]:
    #RS = RandomSearch(llm, budget=budget) 
    LLaMEA_1 = LLaMEA(llm, budget=budget, name="ES", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True)
    LLaMEA_2 = LLaMEA(llm, budget=budget, name="ES-guided", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, feature_guided_mutation=True)
    
    methods = [LLaMEA_1, LLaMEA_2]

    # List containing function IDs we consider
    training_fids = [1,2,3,4,5]
    testing_fids = [1,2,3,4,5]

    training_instances = [(f, i) for f in training_fids for i in range(1, 4)]
    test_instances = [(f, i) for f in testing_fids for i in range(5, 10)]

    if DEBUG:
        logger = ExperimentLogger("results/BBOB_guided_debug")
    else:
        logger = ExperimentLogger("results/BBOB_guided")

    problems = []
    if DEBUG:
        problems.append(
            BBOB_SBOX(
                training_instances=[(f, i) for f in [1,2] for i in range(1, 2)],
                test_instances=[(f, i) for f in [1,2] for i in range(3, 4)],
                dims=[2],
                budget_factor=200,
                eval_timeout=100,
                name=f"SBOX",
                problem_type=ioh.ProblemClass.SBOX,
                full_ioh_log=False,
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )
    else:
        problems.append(
            BBOB_SBOX(
                training_instances=training_instances,
                test_instances=test_instances,
                dims=[10],
                budget_factor=2000,
                eval_timeout=1800,
                name=f"BBOB",
                problem_type=ioh.ProblemClass.BBOB,
                full_ioh_log=False,
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )

    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=5,
        seeds=[1,2,3,4,5],
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=5
    )  # normal run

    if DEBUG:
        experiment = Experiment(
            methods=methods,
            problems=problems,
            runs=1,
            seeds=[1],
            show_stdout=True,
            exp_logger=logger,
            budget=budget,
            n_jobs=5
        )  # test run

    #experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=2000, budget=budget, eval_timeout=270, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


