# Experiment 5: Activation steering as a variation operator

**Question.**
- Can adding a direction to the residual stream act as a controllable
  mutation operator, as an alternative to prompt engineering or
  temperature?
- Can an outer CMA-ES search the steering space for directions that make
  LLaMEA produce better algorithms?

**Builds on.**
- `HookedLLM.set_steering` (already implemented and tested).
- Candidate directions from experiment 3 (`../REPORT_exp03.md`): SAE
  features that detect a concept even with the keyword lines masked, and
  the L21 "undefined name" features. Experiment 2 already showed that a
  probe direction (parent fitness) changes the size of the rewrite.

## Design

1. **Directions.** SAE decoder columns (unit norm) of experiment 3's
   features (`exp_config.DIRECTIONS`):

   | name | layer | feature | what it detects |
   |---|---|---|---|
   | `covariance` | 15 | 5776 | covariance-matrix construction |
   | `gradient` | 15 | 6720 | finite-difference gradients |
   | `swarm` | 27 | 38837 | particle-swarm velocity updates |
   | `local_search` | 27 | 26058 | local search |
   | `undef_9410` | 21 | 9410 | a name that is not defined (any) |
   | `undef_7944` | 21 | 7944 | `func` used out of scope |

   Controls: a random unit direction per layer (`random<L>`), and
   `resample`, an unsteered sample with another seed (natural
   variability).
2. **Dose.** A dose `c` adds `c × a × d` at every generated token, where
   `a` is the feature's typical peak activation (median of its top
   examples in experiment 3). `c = 1` therefore adds what the feature
   contributes when it fires strongly, at every position. Random
   directions use the median `a` of the features at their layer.
   Mean residual norms of the generated code: L15 67, L21 127, L27 334.
3. **Prompts and sampling.** Recorded LLaMEA queries from experiment 1
   (Qwen3-8B, T = 0.8, top-p = 0.95, thinking off; max 2048 new tokens).
   The sampling seed depends only on the prompt and the replicate, not on
   the condition (common random numbers), so differences between
   conditions are paired.
4. **Measures.** Every child is evaluated on experiment 1's
   budget-enforced MA-BBOB (AOCC, 5-D, 10 instances). Per sample:
   failure, fitness, experiment 3's concept labels, undefined references
   (experiment 3, step 13) and similarity to the unsteered child.

Campaigns (`exp_config.CAMPAIGNS`):

| tag | prompts | directions | doses | samples |
|---|---|---|---|---|
| `pilot` | 4 init + 4 mutation (`main`, parent worked) | gradient, covariance, swarm, random15/27 | 0.25, 0.5, 1, 2 | 168 |
| `dose` | 8 init + 32 mutation | covariance, gradient, swarm, local search, random15/27 | 0.5, 1, 1.5 | 800 |
| `scope` | the 99 recorded queries whose child used an undefined name | undef_9410, undef_7944, random21 | −4, +4 | 1,584 (k = 2) |

Later steps (not started):

5. **Steering as an operator.** Add the steered generation as an extra
   LLaMEA operator, and compare it with prompt-only operators at an equal
   evaluation budget.
6. **CMA-ES in steering space.** Parameterise the steering vector as a
   combination of k directions with a coefficient vector z. CMA-ES
   optimises z; the fitness of z is the mean AOCC of algorithms generated
   under steering z.

## How to run

```bash
source experiments/mi/env.sh
S=experiments/mi/exp05_steering/scripts
# 1. directions, scales, residual norms, prompts (~2 min, one GPU)
CUDA_VISIBLE_DEVICES=2 uv run python $S/01_prepare.py --tag dose --device cuda:0
# 2. generate + evaluate (2 processes per GPU on GPUs 2/3; ~15 samples/min)
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((2 + k / 2)) \
  uv run python $S/02_generate.py --tag dose --shard $k --n-shards 4 & done
# 3. summary per condition, paired deltas vs unsteered (CPU)
uv run python $S/03_analyze.py --tag dose
```

## Status

Pilot run (2026-11-08); `dose` and `scope` running.

### Pilot (`results/pilot/summary.md`)

8 prompts, 1 sample per condition; too small for CIs and only used to
pick the doses.

- **Swarm steering writes PSO.** At L27, `swarm` at dose 1 put a particle
  swarm into 8 of 8 children (unsteered 0 of 8) with no failures; at
  dose 0.5, 5 of 8.
- **Gradient steering writes gradients but breaks code.** At dose 1, 7 of
  8 children have finite-difference gradients (unsteered 4 of 8), and 4
  of 8 fail (unsteered 1).
- **Covariance steering does little:** 1–2 of 8 at any dose.
- **Dose 2 breaks generation at L15** for every direction, random
  included: the output degenerates and runs to the token limit. At L27
  (residual norm 334) random dose 2 is still harmless.
- Even the smallest steer changes the code a lot (similarity to the
  unsteered child 0.61–0.70 at dose 0.25): at T = 0.8, any change in the
  logits sends the sample down another path. The `resample` control in
  the later campaigns measures how much of that is ordinary sampling
  variability.
