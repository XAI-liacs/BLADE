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
}
