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

## How to run

```bash
source experiments/mi/env.sh
S=experiments/mi/exp03_concept_features/scripts
# 1. SAE transfer: post-trained vs base model (48 queries, ~2 min each)
uv run python $S/01_validate_saes.py --tag qwen3-8b
uv run python $S/01_validate_saes.py --tag qwen3-8b-base --hf-id Qwen/Qwen3-8B-Base
# 2. encode the 8400 recorded algorithms (4 shards on 2 GPUs: ~12 min)
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((2 + k / 2)) \
  uv run python $S/02_encode.py --tag k100 --sae k100 --shard $k --n-shards 4 & done
uv run python $S/03_concept_labels.py      # regex labels on the code
uv run python $S/04_concept_features.py    # ~20 min (CPU)
uv run python $S/05_feature_contexts.py    # top-activating contexts
uv run python $S/06_concept_intent.py      # last prompt token -> child concepts
# 7. keyword masking (2 shards on GPUs 2/3: ~10 min, then CPU analysis)
for k in 0 1; do CUDA_VISIBLE_DEVICES=$((2 + k)) \
  uv run python $S/07_keyword_masking.py --shard $k --n-shards 2 & done
uv run python $S/07_keyword_masking.py --analyze
```

Data: the Qwen3-8B campaigns of experiment 1 (`main`, `prescreen`,
`prescreen2`; 84 runs, 8400 algorithms). Layers 9, 15, 21, 27. SAE:
Qwen-Scope `W64K-L0_100` (TopK, k = 100, 65,536 features); `common/sae.py`
loads it without SAELens.

## Status

Steps 1–4 of the design and the keyword-masking test done (2026-11-07);
the three-way map with behaviour metrics is still open.

1. **The base-model SAEs transfer.** On the generated code of the
   post-trained model, k100 leaves 21–31% of the variance unexplained
   (base model: 18–26%). Splicing in the reconstruction raises the code
   loss from 0.043 to 0.08–0.14 nats/token: it recovers more than 99% of
   the loss relative to mean ablation. k100 beats k50 at every layer
   (`results/qwen3-8b*/sae_validation.csv`).
2. **Every concept has a feature that detects it**
   (`results/k100/concept_decoding.csv`). Features are chosen on half the
   runs and tested on the other half (validation AUROC):
   - one feature: 0.89–1.00, except step-size adaptation (0.72–0.77);
   - an L1 combination: 0.97–1.00 (step-size adaptation 0.88–0.90).
3. **Most of these features are keyword detectors.** In 34 of 48 best
   features, at least 30% of the top examples peak on the concept's
   identifier (`mutant`, `restart`, `.velocity`, `elite`;
   `results/k100/feature_tokens.csv`, `feature_contexts.md`). The
   exceptions are closer to concepts:
   - finite-difference gradients (L15 f6720, L21 f23754): peak on
     `x_plus`, `x_minus`, `(2 * eps)`;
   - a Gaussian-process surrogate context (L15 f55511): generic tokens
     inside GP code;
   - covariance-matrix construction (L15 f5776);
   - velocity/momentum across families (L27 f46928): PSO velocity and
     gradient momentum.
4. **Concepts are not planned visibly at the last prompt token**
   (`results/k100/concept_intent.csv`).
   - Whether the child contains a concept is predicted by the parent's
     concepts (AUROC 0.81–0.95). The last-token SAE code adds nothing.
   - For children that *introduce* a concept the parent lacks, the SAE
     code adds a little for step-size adaptation (0.61 → 0.69),
     adaptive parameters (+0.05), gradients (+0.045) and swarms (+0.03).
     There are no CIs yet.
   - The code-side concept features are never among the 100 active
     features at the last prompt token. Whatever is planned there is not
     carried by the same features.

5. **Masking the keywords separates three groups**
   (`results/k100/keyword_masking.csv`, `keyword_masking.md`). Each
   top-5 feature is re-scored with the concept's matches hidden, on the
   raw code including comments. There are two masks: the keyword tokens
   alone, and the whole lines containing a match. The line mask is the
   real test, since tokens next to a keyword can carry it forward. A
   random mask of the same size changes nothing (≤ 0.003). The reference
   is what the other concepts' labels plus code length predict without
   the SAE (`other_labels`). Best line-masked validation AUROC over
   layers vs that reference (n = 4,199):
   - *Beyond the keyword and its co-occurrences:*
     - covariance 0.965 vs 0.634;
     - gradient 0.939 vs 0.700;
     - particle swarm 0.922 vs 0.596 (L27 f38837; the L9/L15 features
       fall to 0.55–0.63);
     - adaptive parameters 0.884 vs 0.775;
     - local search 0.876 vs 0.778;
     - population 0.926 vs 0.857;
     - surrogate ≈ 1.000 vs 0.886;
     - step-size adaptation 0.727 vs 0.650.
   - *Explained by co-occurring concepts:* crossover (0.965 vs 0.964) and
     differential evolution (0.962 vs 0.951). The surviving features fire
     on the DE/crossover machinery as a whole, so they read one concept
     from the other's code. They are a DE-family feature, not a
     crossover feature.
   - *Keyword detectors:*
     - elitism: 0.999 falls to 0.57–0.60 at L9–L21; L27 f46445 keeps
       0.76, at the 0.737 reference;
     - restart: falls to 0.78–0.81, barely above 0.737.

   Rank-0 features (median over concepts) fall from 0.97–0.99 to
   0.65–0.80 under line masking. Feature selection by raw AUROC favours
   keyword detectors; the more conceptual features are often at rank 1–4.
   Masking tests where the evidence is read, not what is represented:
   later lines can still attend back to a masked keyword.

### Next steps

- Concept features vs outcome: do they predict failure or fitness beyond
  the labels (intent vs implementation)?
- Behaviour metrics (`iohblade/behaviour_metrics.py`): the BLADE logs keep
  per-instance AUCs but no search trajectories, so a subset of algorithms
  has to be re-run with trajectory logging (CPU).
- Steering candidates for experiment 5: features that survive line
  masking: covariance (L15 f5776), gradient (L15 f6720), particle swarm
  (L27 f38837), adaptive parameters (L15 f14313). Steering a keyword
  detector (elitism, restart) would only test whether the model writes
  the word.
