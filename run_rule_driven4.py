import os

# Limit native math-library threading before importing iohblade/NumPy/SciPy.
# To avoid the errors that disrupt the experiment when running multiple jobs in parallel.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, OpenAI_LLM, Ollama_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA
from iohblade.problems import CEC2005


CEC2005_FUNCTION_IDS = [1, 7, 8, 9, 10, 20, 23]

PROMPT_VARIANTS = [
    ("", False, False),
    ("info-", True, False),
    ("rules-", True, True),
]

MUTATION_PROMPTS = [
    "Refine and simplify the strategy of the selected solution to improve it, but preserve its strongest ideas. Seek a meaningful performance improvement.",
    "Generate a new algorithm that is different from the algorithms you have tried before.",
]


def make_cec_problem(
    logger,
    name,
    fid,
    budget_factor,
    eval_timeout,
    add_info=False,
    add_rules=False,
    debug=False,
):
    """Create a CEC2005 problem scoped to a single function id, with the settings shared by all variants."""
    return CEC2005(
        budget_factor=budget_factor,
        eval_timeout=eval_timeout,
        name=name,
        add_info_to_prompt=add_info,
        add_rules_to_prompt=add_rules,
        full_ioh_log=debug,
        specific_fid=fid,
        training_instances=[fid],
        test_instances=[fid],
        ioh_dir=f"{logger.dirname}/ioh",
    )


def build_problems(logger, debug=False):
    """Build the baseline, feature-info, and rule-driven CEC2005 variants, one per function id."""
    problems = []

    for fid in CEC2005_FUNCTION_IDS:
        for prefix, add_info, add_rules in PROMPT_VARIANTS:
            problem = make_cec_problem(
                logger,
                name=f"CEC2005-{prefix}f{fid}",
                fid=fid,
                budget_factor=200,
                eval_timeout=360,
                add_info=add_info,
                add_rules=add_rules,
                debug=debug,
            )
            problems.append(problem)

    return problems


def main():
    search_budget = 24
    debug = True
    llm = OpenAI_LLM(os.getenv("OPENAI_API_KEY"), "gpt-5.4-mini", temperature=1.0)
    # llm = Ollama_LLM("qwen3-coder:30b")
    method = LLaMEA(
        llm,
        budget=search_budget,
        name="LLaMEA",
        mutation_prompts=MUTATION_PROMPTS,
        n_parents=4,
        n_offspring=16,
        elitism=False,
    )

    logger = ExperimentLogger("results/rule-driven-gpt-5seed-cec2005")
    problems = build_problems(logger, debug=debug)

    experiment = Experiment(
        methods=[method],
        problems=problems,
        seeds=[1, 2, 3, 4, 5],
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        budget=search_budget,
        n_jobs=4,
    )
    experiment()


if __name__ == "__main__":
    main()
