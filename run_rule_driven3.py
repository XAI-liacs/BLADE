import os

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, OpenAI_LLM, Ollama_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA
from iohblade.problems import HLP


ALL_FEATURES = ["Separable", "GlobalLocal", "Multimodality", "Basins", "Homogeneous"]
NOT_FEATURES = ["NOT Basins", "NOT Homogeneous"]
REST_FEATURES = ["Separable", "GlobalLocal", "Multimodality"]

FEATURE_COMBINATIONS = []

for i in range(len(ALL_FEATURES)):
    for j in range(i + 1, len(ALL_FEATURES)):
        FEATURE_COMBINATIONS.append([ALL_FEATURES[i], ALL_FEATURES[j]])
    FEATURE_COMBINATIONS.append([ALL_FEATURES[i]])

for not_feature in NOT_FEATURES:
    for rest_feature in REST_FEATURES:
        FEATURE_COMBINATIONS.append([not_feature, rest_feature])
    FEATURE_COMBINATIONS.append([not_feature])

# FOR TEST
FEATURE_COMBINATIONS = [["Separable", "Multimodality"]]

PROMPT_VARIANTS = [
    ("", False, False),
    ("info-", True, False),
    ("rules-", True, True),
]

MUTATION_PROMPTS = [
    "Refine and simplify the strategy of the selected solution to improve it.",
    "Generate a new algorithm that is different from the algorithms you have tried before.",
    "Stick to using Modular CMA-ES with modcma library and try to improve the selected solution by changing its parameters.",
]

def make_hlp_problem(
    logger,
    name,
    features,
    dim,
    budget_factor,
    eval_timeout,
    add_info=False,
    add_rules=False,
    full_ioh_log=False,
):
    """Create an HLP problem with the settings shared by all variants."""
    return HLP(
        dim=dim,
        budget_factor=budget_factor,
        eval_timeout=eval_timeout,
        name=name,
        add_info_to_prompt=add_info,
        add_rules_to_prompt=add_rules,
        full_ioh_log=full_ioh_log,
        specific_high_level_features=features,
        ioh_dir=f"{logger.dirname}/ioh",
    )


def build_problems(logger):
    """Build the baseline, feature-info, and rule-driven HLP variants."""
    problems = []

    for features in FEATURE_COMBINATIONS:
        for prefix, add_info, add_rules in PROMPT_VARIANTS:
            problem = make_hlp_problem(
                logger,
                name=f"HLP-{prefix}{'-'.join(features)}",
                features=features,
                dim=30,
                budget_factor=2000,
                eval_timeout=360,
                add_info=add_info,
                add_rules=add_rules,
            )
            problems.append(problem)

    return problems


def main():
    search_budget = 24
    # llm = Gemini_LLM(os.getenv("GEMINI_API_KEY"), "gemini-3.5-flash")
    llm = Ollama_LLM("qwen3-coder:30b")
    method = LLaMEA(
        llm,
        budget=search_budget,
        name="LLaMEA",
        mutation_prompts=MUTATION_PROMPTS,
        n_parents=4,
        n_offspring=16,
        elitism=False,
    )

    logger = ExperimentLogger("results/rule-driven-3_2")
    problems = build_problems(logger)

    experiment = Experiment(
        methods=[method],
        problems=problems,
        seeds=[1, 2],
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        budget=search_budget,
        n_jobs=4,
    )
    experiment()


if __name__ == "__main__":
    main()
