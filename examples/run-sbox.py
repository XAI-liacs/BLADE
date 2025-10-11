import os

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.problems import BBOB_SBOX

if __name__ == "__main__":  # prevents weird restarting behaviour
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.5-flash"
    llm1 = Gemini_LLM(api_key, ai_model)
    llm2 = OpenAI_LLM(os.getenv("OPENAI_API_KEY"),"gpt-5-mini")
    llm3 = Ollama_LLM("qwen2.5-coder:14b")  # qwen2.5-coder:14b, deepseek-coder-v2:16b
    #llm4 = Ollama_LLM("deepseek-coder-v2:16b")
    llm4 = Gemini_LLM(api_key, "gemini-1.5-flash")
    budget = 100  # short budgets

    mutation_prompts = [
        "Refine and simplofy the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    llms = [llm1,llm2,llm3,llm4]
    llamea_methods = []
    for llm in llms:
        llamea_methods.append(
            LLaMEA(
                llm,
                budget=budget,
                name=f"Baseline-{llm.model}",
                mutation_prompts=mutation_prompts,
                n_parents=4,
                n_offspring=4,
                elitism=True,
                provide_errors=False,
                )
            )
        llamea_methods.append(
            LLaMEA(
                llm,
                budget=budget,
                name=f"Error-context-{llm.model}",
                mutation_prompts=mutation_prompts,
                n_parents=4,
                n_offspring=4,
                elitism=True,
                provide_errors=True
                )
            )
    # List containing function IDs per group
    group_functions = [
        [],  # starting at 1
        [1, 2, 3, 4, 5],  # Separable Functions
        [6, 7, 8, 9],  # Functions with low or moderate conditioning
        [10, 11, 12, 13, 14],  # Functions with high conditioning and unimodal
        [15, 16, 17, 18, 19],  # Multi-modal functions with adequate global structure
        [20, 21, 22, 23, 24],  # Multi-modal functions with weak global structure
    ]

    problems = []
    for fid in [2, 5, 13, 15, 21]:  # a selection of single functions , 5, 13, 15, 21
        training_instances = [(fid, i) for i in range(1, 6)]
        test_instances = [(fid, i) for i in range(5, 16)]  # 10 test instances
        problems.append(
            BBOB_SBOX(
                training_instances=training_instances,
                test_instances=test_instances,
                dims=[10],
                budget_factor=2000,
                name=f"SBOX_COST_fid{fid}",
                specific_fid=fid,
            )
        )

    logger = ExperimentLogger("results/SBOX-error")
    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=2,
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        n_jobs=6,
        budget=budget,
    )  # normal run
    experiment()  # run the experiment
