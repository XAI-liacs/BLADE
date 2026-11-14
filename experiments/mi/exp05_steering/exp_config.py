"""Configuration for experiment 5 (activation steering as a variation operator).

A *direction* is the decoder column of an SAE feature found in experiment 3
(``DIRECTIONS``) or a random unit vector at the same layer (``random<L>``).
A *dose* ``c`` scales it to ``c`` times the feature's typical peak
activation (median activation of its top examples in experiment 3), so
``c = 1`` adds what the feature contributes when it fires strongly. Random
directions get the median scale of the features at their layer. Negative
doses subtract the direction.
"""

from dataclasses import dataclass, field

EXPERIMENT = "exp05_steering"
SOURCE_EXPERIMENT = "exp01_probing"  # prompts and evaluation settings
CONCEPT_EXPERIMENT = "exp03_concept_features"  # features and their scales
SOURCE_TAGS = ("main", "prescreen", "prescreen2")
CONCEPT_TAG = "k100"

MODEL = "qwen3-8b"
SAE = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"

# name -> (layer, feature, concept label it should add or None)
DIRECTIONS = {
    "covariance": (15, 5776, "covariance"),
    "gradient": (15, 6720, "gradient"),
    "swarm": (27, 38837, "particle_swarm"),
    "local_search": (27, 26058, "local_search"),
    "undef_9410": (21, 9410, None),
    "undef_7944": (21, 7944, None),
}

# Screen for more levers: the rank-0 detector of each concept at L21 and
# L27 (experiment 3, keyword_masking.csv). "population" and
# "step_size_adaptation" are left out (nearly every algorithm has a
# population; the step-size labels are weak, AUROC < 0.76).
_SCREEN = {
    "differential_evolution": {21: 1323, 27: 40999},
    "crossover": {21: 24722, 27: 6049},
    "particle_swarm": {21: 21412, 27: 46928},
    "covariance": {21: 13556, 27: 11916},
    "restart": {21: 43600, 27: 20316},
    "local_search": {21: 52337, 27: 39575},
    "gradient": {21: 23754, 27: 49680},
    "elitism": {21: 56831, 27: 29867},
    "surrogate": {21: 25145, 27: 63904},
    "adaptive_parameters": {21: 35776, 27: 61505},
}
SCREEN_DIRECTIONS = [
    f"{c}{layer}" for c, by_layer in _SCREEN.items() for layer in by_layer
]
for _c, _by_layer in _SCREEN.items():
    for _layer, _f in _by_layer.items():
        DIRECTIONS[f"{_c}{_layer}"] = (_layer, _f, _c)


# Extended sandbox, used when a campaign sets ``extra_deps``: these packages
# are added to experiment 1's evaluation dependencies (pinned to the host
# versions) and the code runs in one namespace
# (``BudgetEnforcedMABBOB.shared_namespace``). Quasi-random designs
# (``scipy.stats.qmc``) and surrogate models (``sklearn``) otherwise fail:
# MA_BBOB's exec hides module-level imports from the class's methods.
EXTRA_EVAL_DEPENDENCIES = ["scipy==1.18.0", "scikit-learn==1.9.0"]


@dataclass
class Campaign:
    tag: str
    directions: list[str]
    random_layers: list[int]
    doses: list[float]
    queries: str = "main"  # "main": init + mutation prompts; "undefined"
    n_init: int = 8
    n_mut: int = 32
    k: int = 1  # replicates (common random numbers across conditions)
    positions: str = "generation"
    # "peak": dose x the feature's peak activation; "residual": dose x the
    # mean residual norm at the direction's layer (comparable across
    # features, needs 01_prepare.py --device)
    dose_unit: str = "peak"
    # "sae": DIRECTIONS (SAE decoder columns); "contrast": directions of
    # step 5b (``08_contrast.py``), named "<c_concept|b_metric>@L<layer>"
    direction_source: str = "sae"
    # evaluate with ``TracedMABBOB`` (adds search-behaviour metrics)
    traced: bool = False
    # extended sandbox (EXTRA_EVAL_DEPENDENCIES + shared namespace)
    extra_deps: bool = False
    max_new_tokens: int = 2048
    seed: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)


CAMPAIGNS = {
    # pick the dose range: where do concepts appear and the code still work?
    "pilot": Campaign(
        tag="pilot",
        directions=["gradient", "covariance", "swarm"],
        random_layers=[15, 27],
        doses=[0.25, 0.5, 1, 2],
        n_init=4,
        n_mut=4,
    ),
    # dose-response for the four line-mask-robust concept features
    "dose": Campaign(
        tag="dose",
        directions=["covariance", "gradient", "swarm", "local_search"],
        random_layers=[15, 27],
        doses=[0.5, 1, 1.5],
        n_init=8,
        n_mut=32,
    ),
    # undefined-name features on prompts that produced an undefined name
    "scope": Campaign(
        tag="scope",
        directions=["undef_9410", "undef_7944"],
        random_layers=[21],
        doses=[-4, 4],
        queries="undefined",
        k=2,
    ),
    # which concept features are levers? Steer norm as a fraction of the
    # residual norm (swarm dose 1 = 0.33, dose 1.5 = 0.49; L15/L21 broke
    # at 0.56-0.85), plus the known swarm lever as a positive control.
    "screen": Campaign(
        tag="screen",
        directions=SCREEN_DIRECTIONS + ["swarm"],
        random_layers=[21, 27],
        doses=[0.3, 0.5],
        dose_unit="residual",
        n_init=4,
        n_mut=12,
        seed=1,
    ),
    # L21 features broke the code at 0.3-0.5 (random21 did not): the same
    # prompts at lower doses, for every L21 feature and the L27 features
    # that failed at 0.3
    "screen2": Campaign(
        tag="screen2",
        directions=[d for d in SCREEN_DIRECTIONS if d.endswith("21")]
        + ["surrogate27", "crossover27"],
        random_layers=[21],
        doses=[0.1, 0.2],
        dose_unit="residual",
        n_init=4,
        n_mut=12,
        seed=1,
    ),
    # Step 5b: contrastive directions (08_contrast.py) on the screen's
    # prompts. Concepts: the rare and elicited ones, positive doses.
    # Behaviour: the exploration axis, raw and with the concept labels
    # regressed out, at signed doses (a dial in both directions), at the
    # best-validated layer and at L27.
    "contrast-c": Campaign(
        tag="contrast-c",
        directions=[
            f"c_{c}@L15"
            for c in (
                "levy_flight",
                "opposition",
                "quasi_random_init",
                "archive",
                "simulated_annealing",
                "surrogate",
                "restart",
                "tournament_selection",
                "local_search",
                "covariance",
            )
        ],
        random_layers=[15],
        doses=[0.15, 0.3, 0.5],
        dose_unit="residual",
        direction_source="contrast",
        traced=True,
        n_init=4,
        n_mut=12,
        seed=1,
    ),
    "contrast-b": Campaign(
        tag="contrast-b",
        directions=[
            f"b_{m}@L{l}"
            for m in ("exploration_pct", "exploration_pct_resid")
            for l in (15, 27)
        ],
        random_layers=[15, 27],
        doses=[-0.5, -0.3, -0.15, 0.15, 0.3, 0.5],
        dose_unit="residual",
        direction_source="contrast",
        traced=True,
        n_init=4,
        n_mut=12,
        seed=1,
    ),
    # confirmation of the exploration dial (concept-residualised directions)
    # on new prompts, 4x the samples per condition
    "dial": Campaign(
        tag="dial",
        directions=[f"b_exploration_pct_resid@L{l}" for l in (15, 27)],
        random_layers=[15, 27],
        doses=[-0.4, -0.2, 0.2, 0.4],
        dose_unit="residual",
        direction_source="contrast",
        traced=True,
        n_init=8,
        n_mut=24,
        k=2,
        seed=2,
    ),
}


# ----------------------------------------------------------------------
# Step 5: steering as a LLaMEA variation operator
# ----------------------------------------------------------------------
@dataclass
class LLaMEAArm:
    """Every arm runs LLaMEA with experiment 1's operators (refine, new,
    simplify) plus one extra operator ``extra`` with equal weight:

    * ``refine``: the refine prompt again (control: more refinement);
    * ``new``: the "new algorithm" prompt again (prompt-based diversity);
    * ``levers``: the refine prompt, generated with one lever from
      ``levers.json`` (``04_select_levers.py``) added, drawn at random;
    * ``random``: as ``levers``, but each lever replaced by a random unit
      direction at the same layer and norm.

    ``extra=None`` adds no operator. ``dial`` steers *every* generated
    algorithm (initial population included) along the campaign's
    exploration dial (``LLaMEACampaign.dial_direction``) with a signed
    dose (x the residual norm):

    * ``explore``: +``dose``; ``exploit``: -``dose``;
    * ``anneal``: from +``dose`` to -``dose``, linear in the share of the
      budget used (explore early, exploit late);
    * ``random``: a random direction of the same norm (one of 8 fixed
      ones, random sign) per algorithm;
    * ``adaptive``: no steering of the initial population; afterwards a
      dose in [-``dose``, +``dose``] (starting at 0) controlled by a
      success rule: after every child that improves the run's best
      fitness the dose moves ``step`` towards exploitation,
      after every other child ``step * target / (1 - target)`` towards
      exploration, so the dose is stationary at a success rate of
      ``target`` (``adaptive``: step 0.1, target 1/5 as in the 1/5th
      rule, though with the opposite sign: stagnation -> explore).
      ``adaptive_cal``: the same stagnation logic, calibrated: the target
      is the control's measured rate of children that improve the run's
      best (0.055 MA-BBOB, 0.064 Bragg, 0.038 packing in step 5c/5d), so
      a run progressing at the usual pace sits near dose 0, a stagnating
      one drifts to explore, a fast one to exploit.

    ``prompt`` (a key of ``ELICIT``, e.g. ``explore``) appends that
    instruction (``ELICIT_TEMPLATE``) to the prompt of every generated
    algorithm, initial population included: the prompt baseline of the
    dial (stage 1 of ``paper_gecco2027/PLAN.md``).
    """

    name: str
    extra: str | None = None
    dial: str = ""
    dose: float = 0.3
    prompt: str = ""
    step: float = 0.1  # adaptive: dose change after a success
    target: float = 0.2  # adaptive: success rate at which the dose is stationary


ADAPTIVE_CAL_TARGET = 0.05  # control's success rate (step 5c/5d, all problems)


@dataclass
class LLaMEACampaign:
    tag: str
    arms: list[LLaMEAArm]
    seeds: list[int]
    lever_source: str = "screen"  # campaign whose levers.json is used
    budget: int = 100
    n_parents: int = 4
    n_offspring: int = 12
    max_new_tokens: int = 3072  # as experiment 1
    # step 5c: exploration dial (a direction of the step 5b corpus)
    dial_direction: str = "b_exploration_pct_resid@L27"
    traced: bool = False  # TracedMABBOB: behaviour of every algorithm
    extra_deps: bool = False  # extended sandbox (see EXTRA_EVAL_DEPENDENCIES)
    # step 5d: "mabbob", or a problem of ``10_transfer.make_problem``
    # ("bragg", "packing26"; always traced where possible, extended sandbox)
    problem: str = "mabbob"
    # warm start (``05_llamea.py``): arm name -> campaign whose finished run
    # of that arm and seed is continued up to ``budget`` (the restored
    # algorithms are copied into the new run's log)
    warm_from: dict[str, str] = field(default_factory=dict)


ARMS = [
    LLaMEAArm("refine2", "refine"),
    LLaMEAArm("new2", "new"),
    LLaMEAArm("levers", "levers"),
    LLaMEAArm("random", "random"),
]
LLAMEA_CAMPAIGNS = {
    "llamea-smoke": LLaMEACampaign(
        tag="llamea-smoke", arms=ARMS, seeds=[999], budget=6, n_offspring=2
    ),
    "llamea": LLaMEACampaign(tag="llamea", arms=ARMS, seeds=list(range(200, 208))),
}
# Step 5c: LLaMEA with the exploration dial on every generated algorithm.
DIAL_ARMS = [
    LLaMEAArm("control"),
    LLaMEAArm("explore", dial="explore"),
    LLaMEAArm("exploit", dial="exploit"),
    LLaMEAArm("anneal", dial="anneal"),
    LLaMEAArm("randomdir", dial="random"),
]
LLAMEA_CAMPAIGNS["llamea-dial-smoke"] = LLaMEACampaign(
    tag="llamea-dial-smoke",
    arms=DIAL_ARMS,
    seeds=[998],
    budget=6,
    n_offspring=2,
    traced=True,
    extra_deps=True,
)
LLAMEA_CAMPAIGNS["llamea-dial"] = LLaMEACampaign(
    tag="llamea-dial",
    arms=DIAL_ARMS,
    seeds=list(range(300, 308)),
    traced=True,
    extra_deps=True,
)


# Stage 1 of paper_gecco2027/PLAN.md: the prompt baseline and the
# adaptive dial, paired with the step 5c/5d campaigns (same seeds), and
# more seeds for the main MA-BBOB arms.
STAGE1_ARMS = [
    LLaMEAArm("adaptive", dial="adaptive"),
    LLaMEAArm("prompt_explore", prompt="explore"),
    LLaMEAArm("prompt_exploit", prompt="exploit"),
]
LLAMEA_CAMPAIGNS["llamea-stage1-smoke"] = LLaMEACampaign(
    tag="llamea-stage1-smoke",
    arms=STAGE1_ARMS,
    seeds=[996],
    budget=12,
    n_offspring=2,
    traced=True,
    extra_deps=True,
)
LLAMEA_CAMPAIGNS["llamea-stage1"] = LLaMEACampaign(
    tag="llamea-stage1",
    arms=STAGE1_ARMS,
    seeds=list(range(300, 308)),  # as llamea-dial
    traced=True,
    extra_deps=True,
)
# The calibrated stagnation dial (decided 2026-11-14): replaces
# ``adaptive`` on later problems if it does better.
ADAPTIVE_CAL_ARM = LLaMEAArm(
    "adaptive_cal", dial="adaptive", step=0.15, target=ADAPTIVE_CAL_TARGET
)
LLAMEA_CAMPAIGNS["llamea-stage1-cal"] = LLaMEACampaign(
    tag="llamea-stage1-cal",
    arms=[ADAPTIVE_CAL_ARM],
    seeds=list(range(300, 308)),  # as llamea-dial
    traced=True,
    extra_deps=True,
)
# Budget 100 -> 200 for control and adaptive_cal (decided 2026-11-14): the
# MA-BBOB runs are continued from their 100-evaluation state.
LLAMEA_CAMPAIGNS["llamea-ext200"] = LLaMEACampaign(
    tag="llamea-ext200",
    arms=[LLaMEAArm("control"), ADAPTIVE_CAL_ARM],
    seeds=list(range(300, 308)),
    budget=200,
    traced=True,
    extra_deps=True,
    warm_from={"control": "llamea-dial", "adaptive_cal": "llamea-stage1-cal"},
)
LLAMEA_CAMPAIGNS["llamea-ext200-adaptive"] = LLaMEACampaign(
    tag="llamea-ext200-adaptive",
    arms=[LLaMEAArm("adaptive", dial="adaptive")],
    seeds=list(range(300, 308)),
    budget=200,
    traced=True,
    extra_deps=True,
    warm_from={"adaptive": "llamea-stage1"},
)
LLAMEA_CAMPAIGNS["llamea-ext-smoke"] = LLaMEACampaign(
    tag="llamea-ext-smoke",
    arms=[LLaMEAArm("adaptive", dial="adaptive")],
    seeds=[996],
    budget=16,
    n_offspring=2,
    traced=True,
    extra_deps=True,
    warm_from={"adaptive": "llamea-stage1-smoke"},
)
LLAMEA_CAMPAIGNS["llamea-stage1-seeds"] = LLaMEACampaign(
    tag="llamea-stage1-seeds",
    arms=[a for a in DIAL_ARMS if a.name in ("control", "explore", "exploit")]
    + STAGE1_ARMS,
    seeds=list(range(308, 316)),
    traced=True,
    extra_deps=True,
)


# Step 5b: contrastive levers. A labelled corpus of code (natural: every
# algorithm of experiment 1's main campaign and of the `llamea` campaign;
# elicited: initial prompts with one extra instruction), with concept
# labels (experiment 3's regexes) and search-behaviour labels (traced
# evaluation, ``common.problems.TracedMABBOB``). Directions are differences
# of mean residual states (teacher forced, one neutral prompt) between code
# with and without a concept, or between the most and least exploratory
# code.
ELICIT = {
    "levy_flight": "Levy flights: heavy-tailed step lengths drawn with "
    "Mantegna's algorithm",
    "opposition": "opposition-based learning: also evaluate the opposite "
    "point lb + ub - x of candidate solutions",
    "quasi_random_init": "a quasi-random initial design (Latin hypercube or "
    "Sobol sampling)",
    "archive": "an archive of good or recently replaced solutions that new "
    "candidates are drawn from",
    "simulated_annealing": "simulated annealing: accept worse moves with a "
    "probability that depends on a decreasing temperature",
    "surrogate": "a cheap surrogate model of the objective to pre-select "
    "candidates before evaluating them",
    "restart": "restarts from new random points when the search stagnates",
    "tournament_selection": "tournament selection of parents",
    "local_search": "a local search (pattern search or Nelder-Mead) around "
    "the best solution",
    "covariance": "a covariance matrix that is adapted to shape the "
    "sampling distribution (as in CMA-ES)",
    "explore": "strong global exploration: it must keep spreading its "
    "evaluations over the whole search space for as long as possible",
    "exploit": "strong exploitation: it must quickly concentrate its "
    "evaluations around the best solution found and refine it",
}
ELICIT_TEMPLATE = "\nThe algorithm must use {}."
CORPUS = {
    "tag": "corpus",
    "n_per_target": 24,
    "natural": {"exp01_probing": "main", "exp05_steering": "llamea"},
    "trace_instances": list(range(10)),  # all of experiment 1's instances
}
CONTRAST_LAYERS = [9, 15, 21, 27, 33]
BEHAVIOUR_METRICS = [
    "exploration_pct",
    "dist_to_best",
    "intensification",
    "last_improvement",
    "success_rate",
]


# Step 5d: does the exploration dial transfer to other BLADE problems?
# Screen on the problem's own LLaMEA prompts: the initial prompt (k_init
# samples per condition) and refine prompts on n_parents working unsteered
# initial algorithms (k_refine each). Conditions: none, the dial at each
# dose, and a random unit direction (one of 8, random sign) at
# random_dose. Common random numbers: the sampling seed depends on the
# prompt and the replicate only.
@dataclass
class TransferCampaign:
    tag: str
    problem: str  # "bragg" (TracedPhotonics) or "packing26"
    doses: list[float] = field(default_factory=lambda: [-0.4, -0.2, 0.2, 0.4])
    random_dose: float = 0.4
    direction: str = "b_exploration_pct_resid@L27"
    k_init: int = 32
    # stage 1 (E5): extra initial-prompt replicates k_init .. k_init +
    # k_init_extra - 1 for these conditions only (names as 10_transfer)
    k_init_extra: int = 0
    extra_conditions: list[str] = field(default_factory=list)
    n_parents: int = 16
    k_refine: int = 2
    max_new_tokens: int = 2048


TRANSFER_CAMPAIGNS = {
    t.tag: t
    for t in [
        TransferCampaign("transfer-smoke-bragg", "bragg", k_init=2, n_parents=1),
        TransferCampaign("transfer-smoke-packing", "packing26", k_init=2, n_parents=1),
        TransferCampaign(
            "transfer-bragg",
            "bragg",
            k_init_extra=32,
            extra_conditions=["none", "dial@-0.4", "dial@+0.4"],
        ),
        TransferCampaign("transfer-packing", "packing26"),
    ]
}

# Step 5d, part 2: LLaMEA with the dial on the transfer problems.
for _p, _name in [("bragg", "bragg"), ("packing26", "packing")]:
    LLAMEA_CAMPAIGNS[f"llamea-dial-{_name}-smoke"] = LLaMEACampaign(
        tag=f"llamea-dial-{_name}-smoke",
        arms=DIAL_ARMS,
        seeds=[997],
        budget=6,
        n_offspring=2,
        problem=_p,
    )
    LLAMEA_CAMPAIGNS[f"llamea-dial-{_name}"] = LLaMEACampaign(
        tag=f"llamea-dial-{_name}",
        arms=DIAL_ARMS,
        seeds=list(range(400, 406)),
        problem=_p,
    )
    LLAMEA_CAMPAIGNS[f"llamea-stage1-{_name}"] = LLaMEACampaign(
        tag=f"llamea-stage1-{_name}",
        arms=STAGE1_ARMS,
        seeds=list(range(400, 406)),  # as llamea-dial-{_name}
        problem=_p,
    )
