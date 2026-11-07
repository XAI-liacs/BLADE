"""Configuration for experiment 3 (algorithmic concept features)."""

EXPERIMENT = "exp03_concept_features"
SOURCE_EXPERIMENT = "exp01_probing"  # recorded LLaMEA runs to analyse

MODEL = "qwen3-8b"  # the only model with public residual SAEs
SAES = {
    "k50": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    "k100": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
}
# Layers to analyse: around the experiment 1 probe layers (9: over budget,
# 15: parent fitness, 21: child failure) and one late layer.
LAYERS = (9, 15, 21, 27)
# Qwen3-8B campaigns of experiment 1 (12 + 24 + 48 runs of 100 queries).
SOURCE_TAGS = ("main", "prescreen", "prescreen2")
