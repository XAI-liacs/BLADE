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
    """

    name: str
    extra: str


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
