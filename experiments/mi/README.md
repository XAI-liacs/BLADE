# Mechanistic Interpretability experiments (branch `experiments/mi`)

Research notes for opening up the *generator* inside LLaMEA / BLADE: the
LLM that writes and mutates the optimisation heuristics. So far BLADE treats
the LLM as a black box (prompt in, code out). The goal of this branch is to
look at its internal activations while it designs algorithms.

> Status: planning only. No code changes yet.

## Prerequisite: an open-weights, hookable LLM backend

MI needs access to the model's residual stream, attention and MLP activations.
`Claude_LLM`, `OpenAI_LLM`, `Gemini_LLM` and even `Ollama_LLM` /
`LMStudio_LLM` (quantised GGUF behind an HTTP API) do not expose them. Step 0
is a new provider in `iohblade/llm.py`, for example `HookedLLM`:

- wraps a HuggingFace model through `transformer_lens.HookedTransformer` or
  `nnsight.LanguageModel`;
- implements `_query()` as usual, so LLaMEA and the other methods work
  unchanged;
- optionally caches selected activations per query (mean-pooled or
  per-token at chosen layers/hooks, or SAE feature activations) and
  attaches them to the `Solution` (metadata or a side file keyed by
  `solution.id`), so the activations sit next to fitness, feedback,
  parent ids and mutation operator in the BLADE logs;
- supports interventions (steering vector, feature clamping, ablation)
  passed in through kwargs, so an intervention can act as a search operator.

Candidate models (they need public SAEs/transcoders *and* acceptable code
quality):

| Model | Public dictionaries | Notes |
|-------|---------------------|-------|
| Gemma-2-2B / 9B | Gemma Scope (SAEs + transcoders), circuit-tracer support | Best tooling; 2B code quality is weak |
| Gemma-3 family | Gemma Scope 2 | Newer; check SAELens support |
| Llama-3.1-8B (-Instruct) | Llama Scope SAEs, Goodfire SAEs | Reasonable code quality |
| Llama-3.2-1B | circuit-tracer transcoders | Toy-scale attribution graphs |
| Qwen3 (small) / Qwen2.5-Coder-7B | some circuit-tracer transcoders for Qwen3; train your own for Coder | Best code quality per parameter |

LLaMEA has produced large corpora of prompts and generated code. We can use
them to train our own task-specific SAEs/transcoders (SAELens, `sparsify`)
on a code model.

## Research directions (tied to the LLaMEA line)

1. **Does the model know if its mutation will work?** Train linear probes on
   activations at the end of the mutation prompt (after the parent code and
   its feedback) to predict (a) parent fitness, (b) fitness improvement of
   the child, (c) whether the child will crash or time out. If the probes
   work, we get a free surrogate for pre-screening candidates, which saves
   evaluation budget. Connects to the LLaMEA surrogate and budget work.

2. **Is the feedback actually used?** Use attribution (circuit-tracer
   attribution graphs, integrated gradients via `inseq`/`captum`, attention
   knock-out) from the generated code back to the feedback tokens (fitness
   value, error trace, behaviour metrics). Measure how much of the edit is
   driven by feedback vs. by the parent code vs. by the task prompt. Compare
   mutation prompts ("refine", "new", "simplify"), and test whether richer
   feedback (BLADE behaviour metrics) is used or ignored.

3. **Algorithmic concept features and code structure.** Find SAE features
   for metaheuristic concepts (population, step-size adaptation, restart,
   covariance, crossover, elitism, "CMA-ES", "DE", ...). Correlate them with
   the AST/code-evolution-graph features and `behaviour_metrics.py` already
   in BLADE. This gives a three-way map between latent intent, code
   structure and search behaviour, which extends the explainable-AAD work
   into the model itself.

4. **Why does it converge to DE/CMA-ES-like designs? Mode collapse and
   priors.** Find the features or directions behind the "default" algorithm
   families. Ablating or suppressing them during generation is a test of
   whether population diversity and novelty go up, and whether fitness goes
   down.

5. **Activation steering as a variation operator.** Use steering vectors
   (contrastive, e.g. "exploratory" vs "exploitative" algorithm
   descriptions) or SAE-feature clamping as a mutation operator, as an
   alternative to temperature or prompt engineering for controlling
   exploration. Natural next step for this group: run **CMA-ES in the
   steering/feature space** to search latent directions that maximise
   generated-algorithm fitness on BBOB/MA-BBOB. That gives a two-level
   search: an optimiser in activation space on top of LLaMEA in code space.

6. **Recall or invention?** Use logit lens/tuned lens, token entropy and
   feature signatures to tell apart code that is reproduced from known
   algorithms in the training data from genuinely recombined code. Relate
   this to the novelty of the final elite algorithms.

7. **Model diffing.** Use crosscoders (`dictionary_learning`) or SAE
   diffing between a base code model and the same model fine-tuned on
   LLaMEA-generated elites, or between base and instruct models. The
   question is what the fine-tuning adds: new algorithmic concepts, or only
   formatting and style.

8. **Error and timeout circuits.** Find the internal signals that come
   before code that crashes or exceeds the budget, and use them to
   early-stop generation or reject candidates without executing them.

Suggested order: 0 → 1 → 2 → 3, then 5 (steering) as the main contribution.
Directions 1 and 8 have the most immediate practical payoff for BLADE.

## Python libraries to consider

**Model access and hooking**
- `transformer_lens`: HookedTransformer with clean hook points; the standard for
  small/medium models, built-in caching, logit lens, activation patching.
- `nnsight` (+ NDIF): tracing and interventions on any HF model; NDIF gives
  remote access to large open models (e.g. Llama-3.1-70B/405B) without
  local GPUs.
- `nnterp`: unified nnsight-based interface across architectures.
- `pyvene` (Stanford): declarative interventions, causal abstraction and
  interchange interventions.
- `baukit`: lightweight tracing/editing helpers.

**Sparse dictionaries (SAEs, transcoders, crosscoders)**
- `sae_lens`: load pretrained SAEs (Gemma Scope, Llama Scope, ...) and train
  new ones.
- `sparsify` (EleutherAI): train SAEs/transcoders at scale on custom data
  (e.g. LLaMEA corpora).
- `dictionary_learning` (Sam Marks et al.): SAEs and crosscoders for model
  diffing.
- `delphi` (EleutherAI): automated interpretation (LLM-written
  explanations and scoring) of features.
- Neuronpedia (web + API): browse, search and explain features; hosts
  attribution graphs.

**Anthropic-originated open-source**
- `circuit-tracer` (github.com/safety-research/circuit-tracer): the
  open-source release of Anthropic's attribution graph / circuit tracing
  method ("Biology of an LLM"). Builds attribution graphs with
  cross-layer/per-layer transcoders on Gemma-2-2B, Llama-3.2-1B and Qwen3
  models, supports interventions, and integrates with Neuronpedia for
  visualisation.
- Methodology to reimplement rather than install: Towards
  Monosemanticity / Scaling Monosemanticity (SAEs), Crosscoders, Circuit
  Tracing, and the Transformer Circuits thread (transformer-circuits.pub).
  Note: Claude's internals are **not** accessible via the API, so
  `Claude_LLM` cannot be used for MI experiments. It can still serve as the
  automated interpreter (feature labelling) or as a judge.

**Attribution / classic XAI on LLMs** (closest to existing XAI work)
- `inseq`: feature attribution for generative LMs (IG, attention, gradient x
  input, contrastive attribution).
- `captum` (`LLMAttribution`): perturbation and gradient attribution for
  LLMs.

**Steering**
- `steering-vectors`, `repeng`: contrastive steering vectors / control
  vectors on HF models.
- Goodfire Ember API: hosted SAE-based steering on Llama models (commercial;
  useful for fast prototyping).

**Probing and analysis**
- `scikit-learn` (linear/logistic probes), `torch`, `einops`, `jaxtyping`,
  `plotly`/`circuitsvis` for attention and feature visualisation.

Proposed dependency group in `pyproject.toml` (to add once we start
implementing; the heavy `torch` stack must stay optional):

```toml
mi = [
    "torch",
    "transformers",
    "transformer-lens",
    "nnsight",
    "sae-lens",
    "inseq",
    "scikit-learn",
    "circuitsvis",
]
# circuit-tracer is installed from git:
#   uv add --group mi "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer"
```

Check versions and model support before pinning; this ecosystem changes
quickly.
