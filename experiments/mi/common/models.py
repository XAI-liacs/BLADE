"""Model presets shared by all MI experiments.

Keep experiments 1, 2, 3 and 5 on the same model so that probes, attribution
graphs, SAE features and steering vectors live in one representation space.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPreset:
    hf_id: str
    chat_template_kwargs: dict = field(default_factory=dict)
    gated: bool = False
    transcoders: str | None = None  # circuit-tracer (experiment 2)
    saes: tuple[str, ...] = ()  # residual SAEs (experiment 3)
    store_dtype: str = "float16"  # activations on disk (see HookedLLM)
    notes: str = ""


MODELS: dict[str, ModelPreset] = {
    # Default: ungated, decent code quality, and both circuit-tracer
    # transcoders and official Qwen residual SAEs are public.
    "qwen3-8b": ModelPreset(
        hf_id="Qwen/Qwen3-8B",
        chat_template_kwargs={"enable_thinking": False},
        transcoders="mwhanna/qwen3-8b-transcoders",
        saes=(
            "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
            "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
        ),
        notes="SAEs are trained on the base model; check transfer to the "
        "post-trained model (reconstruction loss) before relying on them.",
    ),
    # Replication models (experiments 1 and 2 repeated on a bigger model of
    # the same family and on another family; both have transcoders).
    "qwen3-14b": ModelPreset(
        hf_id="Qwen/Qwen3-14B",
        chat_template_kwargs={"enable_thinking": False},
        transcoders="mwhanna/qwen3-14b-transcoders",
        notes="40 layers, d_model 5120.",
    ),
    "gemma-3-12b": ModelPreset(
        hf_id="google/gemma-3-12b-it",
        gated=True,
        transcoders="mwhanna/gemma-scope-2-12b-it",
        store_dtype="float32",  # residual norms overflow float16
        notes="Gated: needs HF_TOKEN with an accepted licence. 48 layers, "
        "d_model 3840. Transcoders: Gemma Scope 2 in circuit-tracer format.",
    ),
    "qwen3-4b": ModelPreset(
        hf_id="Qwen/Qwen3-4B",
        chat_template_kwargs={"enable_thinking": False},
        transcoders="mwhanna/qwen3-4b-transcoders",
        notes="Cheaper variant for attribution graphs.",
    ),
    "qwen2.5-0.5b": ModelPreset(
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        notes="Smoke tests only; code quality is too low for real runs.",
    ),
    "llama-3.1-8b": ModelPreset(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        gated=True,
        saes=("Goodfire/Llama-3.1-8B-Instruct-SAE-l19",),
        notes="Gated: needs HF_TOKEN with an accepted licence.",
    ),
    "gemma-2-9b": ModelPreset(
        hf_id="google/gemma-2-9b-it",
        gated=True,
        saes=("google/gemma-scope-9b-it-res",),
        notes="Gated. Best MI tooling (Gemma Scope), weaker code.",
    ),
}
