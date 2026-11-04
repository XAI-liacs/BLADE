"""Configuration for experiment 1 (probing for fitness awareness).

All settings that define a data-collection campaign live here, so a ``tag``
fully identifies how a dataset was produced. Scripts select a preset with
``--config`` and may override a few fields on the command line.
"""

from dataclasses import asdict, dataclass, field

EXPERIMENT = "exp01_probing"

# LLaMEA variation operators (single-parent). The prompts are matched in the
# logged prompt text to recover which operator produced a child.
OPERATORS = {
    "refine": "Refine the strategy of the selected solution to improve it.",
    "new": "Generate a new algorithm that is different from the algorithms "
    "you have tried before.",
    "simplify": "Refine and simplify the selected algorithm to improve it.",
}


@dataclass
class ProbeConfig:
    tag: str
    model: str = "qwen3-8b"  # key in common.models.MODELS
    seeds: list[int] = field(default_factory=lambda: list(range(12)))
    budget: int = 100  # LLM-generated algorithms per LLaMEA run
    n_parents: int = 4
    n_offspring: int = 12
    elitism: bool = True
    operators: list[str] = field(default_factory=lambda: list(OPERATORS))
    # LLM sampling
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 3072
    # MA-BBOB evaluation. "budget_enforced" truncates runs that overspend
    # instead of scoring them 0 (common.problems.BudgetEnforcedMABBOB);
    # "mabbob" is BLADE's original scoring (used by the pilot).
    problem: str = "budget_enforced"
    dims: list[int] = field(default_factory=lambda: [5])
    budget_factor: int = 2000
    training_instances: list[int] = field(default_factory=lambda: list(range(10)))
    eval_timeout: int = 120
    # Sandbox packages for evaluating generated code. SMAC is left out: it
    # is only needed for HPO and its pyrfr dependency needs Python headers.
    eval_dependencies: list[str] = field(
        default_factory=lambda: ["pandas==2.2.3", "ioh==0.3.22", "configspace==1.2.1"]
    )

    # In-loop pre-screen (05_prescreen_generate.py): probe exported by
    # 04_export_probes.py from campaign ``prescreen_source``. Each seed runs
    # twice, "shadow" (flag only) and "screened" (flagged code is shown to
    # LLaMEA as failed); every algorithm is still evaluated.
    prescreen_probe: str | None = None
    prescreen_source: str = "main"

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # Pipeline check: tiny model, two short runs. Minutes, not hours.
    "smoke": ProbeConfig(
        tag="smoke",
        model="qwen2.5-0.5b",
        seeds=[0, 1],
        budget=6,
        n_parents=2,
        n_offspring=2,
        max_new_tokens=1024,
        training_instances=list(range(2)),
        eval_timeout=60,
    ),
    # Pilot with the real model: one seed per GPU, ~20 min. Checks label
    # balance and throughput before committing to the main campaign.
    # The pilot used the original scoring; 84% of working algorithms then
    # scored 0 for overspending, which motivated "budget_enforced".
    "pilot": ProbeConfig(
        tag="pilot",
        seeds=[0, 1, 2, 3],
        budget=20,
        n_offspring=4,
        problem="mabbob",
    ),
    # Main campaign: 12 runs x 100 algorithms = ~1200 labelled LLM queries.
    "main": ProbeConfig(tag="main"),
    # Held-out test of the child-failure pre-screen: new seeds, the probe
    # and threshold frozen from "main" (layer 21, 20% flagged).
    "prescreen": ProbeConfig(
        tag="prescreen",
        seeds=list(range(100, 112)),
        prescreen_probe="child_failed_gen_mean_L21",
    ),
    # 24 more seeds of the same design, for the non-inferiority test on the
    # search outcome (power analysis in 09_search_effect.py: 36 seeds give
    # 82% power for a 0.10 AOCC margin). Analysed pooled with "prescreen".
    "prescreen2": ProbeConfig(
        tag="prescreen2",
        seeds=list(range(112, 136)),
        prescreen_probe="child_failed_gen_mean_L21",
    ),
}
