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
uv run python $S/08_concept_outcomes.py    # features vs failure/fitness (~15 min CPU)
uv run python $S/09_failure_features.py    # failure-predictive SAE features
uv run python $S/10_sparse_failure_probe.py # strata + L1 probes (~10 min CPU)
uv run python $S/11_failure_types.py       # per error type (~5 min CPU)
CUDA_VISIBLE_DEVICES=2 uv run python $S/12_undefined_names.py  # ~3 min GPU
```

Data: the Qwen3-8B campaigns of experiment 1 (`main`, `prescreen`,
`prescreen2`; 84 runs, 8400 algorithms). Layers 9, 15, 21, 27. SAE:
Qwen-Scope `W64K-L0_100` (TopK, k = 100, 65,536 features); `common/sae.py`
loads it without SAELens.

## Status

Steps 1–4 of the design, the keyword-masking test (2026-11-07) and the
link to outcomes (2026-11-08) done; the three-way map with behaviour
metrics is still open.

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

6. **Concept features do not predict outcomes beyond the code; the rest
   of the SAE code predicts failure** (`results/k100/concept_outcomes.*`).
   This uses 10-fold cross-validation over runs. The blocks are:
   - M: the metadata of experiment 1;
   - L: the 18 labels + log length;
   - C: the 9 line-mask-robust concept features;
   - A: all 240 selected concept features;
   - S: the whole SAE code, as 256 label-free SVD components per layer;
   - T: code TF-IDF.

   | target | M+L | +C | +A | +S | M+L+T | +C | +S |
   |---|---|---|---|---|---|---|---|
   | failure (AUROC, n = 8,381) | 0.757 | +0.013 | +0.070 | **+0.120** | 0.804 | +0.001 | **+0.074** [0.058, 0.089] |
   | over budget (AUROC, n = 6,968) | 0.921 | +0.004 | +0.015 | +0.016 | 0.935 | +0.002 | +0.004 [−0.002, 0.012] |
   | fitness (Spearman, n = 6,968) | 0.746 | +0.001 | +0.006 | +0.016 | 0.749 | +0.001 | +0.012 [0.004, 0.018] |

   - The concept features add nothing once the code text is known.
   - The whole SAE code reaches failure AUROC 0.877. That is the same
     gain over text that experiment 1's residual-stream probe showed
     (+0.079). Its gain for fitness is small.
   - Within the algorithms that use a concept, the concept feature's
     strength is mostly unrelated to the outcome. One exception: among
     PSO algorithms, a stronger velocity feature (L27 f38837) goes with
     lower fitness (Spearman −0.22 [−0.31, −0.13], n = 1,183).

   The concept representation records *what* is implemented, not
   whether it will work. The SVD is fitted on all solutions, including
   the test folds; it uses no labels.
7. **Failure is spread over many weak features**
   (`results/k100/failure_features.*`). No single feature exceeds
   validation AUROC 0.76. The readable ones:
   - *Predict failure:*
     - scikit-learn Gaussian-process code (`Regressor`, `Kernel`,
       `Gaussian`, `.fit`, `.predict`); surrogate algorithms fail 93% of
       the time;
     - finite-difference gradients (`.gradient`, `(func`);
     - covariance setup (`.eye`);
     - random sampling calls (`.uniform`, `.normal`, `.clip`).
   - *Predict working code:* features that fire everywhere but more
     strongly in tidy code (`class`, `def`, imports, comments, loop
     headers).

   By label, failure rates are: surrogate 0.93, restart 0.50, elitism
   0.22, covariance 0.21, against 0.10–0.16 for DE, PSO, local search
   and adaptive parameters. That signal is already in L. What S adds
   beyond L and T seems to be a diffuse code-quality / complexity
   signal, not a concept.

8. **The failure signal is robust but needs hundreds of features**
   (`results/k100/failure_strata.csv`, `failure_sparse.csv`,
   `failure_probe.md`).
   - *Strata:* the gain of the SAE code over M+L+T holds wherever an
     obvious cause is fixed:
     - without surrogates: +0.086 [0.068, 0.103];
     - without surrogates or restarts: +0.084;
     - per operator: +0.060 (init) to +0.119 (simplify);
     - after a working parent: +0.080; after a failing parent: +0.146.
   - *Sparse probes* (L1, out of fold, stacked with M+L+T):
     - fewer than about 10 features add nothing beyond the text
       (≤ +0.006);
     - 25–40 features recover about half of the gain (L21, 40 features:
       +0.042);
     - the full gain needs about 260 features at one layer (L21: +0.089)
       or about 370 over all layers (+0.091; M+L+T+S 0.894).
     - L21 is the best single layer, as in experiment 1's probe.
   - *Named features* (L21, 37 features at C = 0.01):
     - *Toward failure:* scikit-learn Gaussian-process code (`.fit`,
       `_std`, `alpha`, `Process`, `Kernel`); matrix products (`.dot`,
       `.outer`); finite-difference gradients (`.gradient`);
       `random.choice` calls.
     - *Toward working code:* features that fire more often in working
       code:
       - an "only numpy imported, class follows" feature (f6057, on the
         newline before `class`) and `import`/`from` lines;
       - budget/parameter boilerplate (`self.budget = budget`,
         `self.F = F`);
       - `func.bounds` and indexing in DE updates;
       - `.copy`, `perturbation`, `for`, `init`.
   - *Ground truth:* every algorithm importing scikit-learn fails (2.4%
     of all, n = 186); with scipy, 75% fail (vs 16%). The SAE picks
     these up, but so can the text. What the SAE adds on top is a
     distributed sense of "standard, budget-aware numpy code" vs
     "unusual dependencies and linear algebra". It is not a few
     interpretable failure detectors.

9. **The SAE code beats the text most on reference errors**
   (`results/k100/failure_types.*`). Each error type is scored against
   working code (out-of-fold AUROC). The most common error messages:
   - ValueError: `np.random.choice` on a 2-D population, ambiguous
     array truth values;
   - TypeError: `np.gradient(func, x)` misuse;
   - ModuleNotFoundError: scikit-learn is not installed.

   | type | n | text | SAE (L21, L1) | text + SAE | delta |
   |---|---|---|---|---|---|
   | ValueError | 421 | 0.749 | 0.848 | 0.850 | +0.100 [0.072, 0.129] |
   | TypeError | 265 | 0.816 | 0.893 | 0.906 | +0.090 [0.053, 0.128] |
   | ModuleNotFoundError | 205 | 1.000 | 1.000 | 1.000 | 0 |
   | IndexError | 162 | 0.786 | 0.817 | 0.827 | +0.042 [0.021, 0.063] |
   | AttributeError | 121 | 0.607 | 0.783 | 0.786 | +0.179 [0.108, 0.242] |
   | NameError | 92 | 0.723 | 0.921 | 0.917 | +0.194 [0.139, 0.248] |

   - The gain is largest for errors about whether a name or attribute
     exists. A bag of words cannot see that: it needs to relate a use
     to its definition.
   - The TypeError probe is led by an `np.gradient` feature (f64822,
     AUROC 0.84).
   - The NameError probe is led by a single feature, L21 f7944 (AUROC
     0.85).
   - Timeouts are rare: only 9 of the failures.
10. **An "out-of-scope reference" feature** (`results/k100/undefined_names.*`).
    Most NameErrors (39 of 92) are `name 'func' is not defined`: the
    objective, a parameter of `__call__`, is used in a helper method
    that does not receive it.
    - *Labels:* every load of `func` was labelled from the AST with
      Python's scoping rules. 52 solutions have an out-of-scope
      reference; 40 of them fail with NameError, and only 2 work.
    - *Tokens:* 88 out-of-scope and 7,128 in-scope references, from 752
      solutions. Of the in-scope ones, 754 lie outside `__call__`, in
      helpers that take `func` as a parameter or in closures.

    | group | n | L21 f7944 active | L21 f9410 active |
    |---|---|---|---|
    | out of scope | 88 | 0.55 (mean 3.2) | 0.72 (mean 7.5) |
    | in scope, inside `__call__` | 6,374 | 0.000 | 0.006 |
    | in scope, elsewhere | 754 | 0.000 | 0.005 |
    | all other code tokens | – | 0.000 | – |

    - f7944 never fires on an in-scope `func`, or on any other code
      token. It fires on about half the out-of-scope references.
    - f9410 is the best separator among all 65,536 features (token
      AUROC 0.857); two more, f57487 and f16349, behave the same way.
    - These features read scope, not position: legitimate uses outside
      `__call__` stay silent.
    - The model represents the bug at the token where it writes it, and
      writes it anyway.

### Next steps

- Out-of-scope features beyond `func`: do f7944 / f9410 also fire on
  other undefined names (`norm`, `minimize` used without import;
  `self.func` never assigned)?
- Experiment 5 candidates from the failure analysis: the out-of-scope
  features (L21 f7944, f9410). Ablating or amplifying them while the
  model writes a helper method tests whether the representation can be
  used, e.g. to make the model pass `func` along.
- Behaviour metrics (`iohblade/behaviour_metrics.py`): the BLADE logs keep
  per-instance AUCs but no search trajectories, so a subset of algorithms
  has to be re-run with trajectory logging (CPU).
- Steering candidates for experiment 5: features that survive line
  masking: covariance (L15 f5776), gradient (L15 f6720), particle swarm
  (L27 f38837), adaptive parameters (L15 f14313). Steering a keyword
  detector (elitism, restart) would only test whether the model writes
  the word.
