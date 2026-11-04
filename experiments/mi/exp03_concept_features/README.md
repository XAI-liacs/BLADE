# Experiment 3: Algorithmic concept features

**Question.** Which sparse features of the model encode metaheuristic
concepts (population, step-size adaptation, restarts, covariance learning,
crossover, elitism, "CMA-ES", "DE", ...)? Do these features match the
structure of the generated code and the search behaviour of the resulting
algorithms?

**Builds on.**
- Experiment 1's recorded runs (`HookedLLM.replay` for per-token states).
- BLADE's code features: AST and code-evolution-graph features.
- `iohblade/behaviour_metrics.py`.

## Design (draft)

1. **Validate the SAEs.** Load `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` /
   `L0_100` with SAELens and measure reconstruction (explained variance,
   downstream loss) on our activations. They were trained on the base model.
2. **Encode** the generated code tokens. Aggregate each feature per
   solution (max and mean over code tokens).
3. **Find concept features.** Contrast feature activations between
   algorithms with and without each concept. Labels come from AST
   heuristics and, optionally, an LLM judge. Use `delphi` /
   Neuronpedia-style auto-interpretation for the top features.
4. **Three-way map.** Correlate features ↔ AST features ↔ behaviour
   metrics (exploration/exploitation, convergence speed, diversity). The
   aim is to show where the latent "intent" and the actual behaviour
   diverge.
5. **Fallback.** If transfer from the base-model SAEs is poor, train an SAE
   on LLaMEA-generated code (`sparsify`) using the activations of
   experiments 1 and 3.

The resulting concept features are the candidate steering directions for
experiment 5.

## Libraries to add

`sae-lens`, `delphi` (optional), `sparsify` (fallback).

## Status

Planned.
