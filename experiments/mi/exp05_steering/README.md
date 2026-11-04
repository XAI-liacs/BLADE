# Experiment 5: Activation steering as a variation operator

**Question.**
- Can adding a direction to the residual stream act as a controllable
  mutation operator, as an alternative to prompt engineering or
  temperature?
- Can an outer CMA-ES search the steering space for directions that make
  LLaMEA produce better algorithms?

**Builds on.**
- `HookedLLM.set_steering` (already implemented and tested).
- Candidate directions from experiment 1 (probe weights, e.g. the "will
  improve" direction) and experiment 3 (concept features).

## Design (draft)

1. **Directions.**
   - Contrastive mean differences (e.g. exploratory vs. exploitative
     algorithm descriptions, or `new` vs. `refine` prompts).
   - Probe weight vectors from experiment 1.
   - SAE decoder directions for the concept features from experiment 3.
2. **Dose–response.** For each direction, sweep α and the layer. Measure:
   - validity (failure rate);
   - concept presence in the code (AST);
   - behaviour metrics;
   - fitness.
3. **Steering as an operator.** Add the steered generation as an extra
   LLaMEA operator, and compare it with prompt-only operators at an equal
   evaluation budget.
4. **CMA-ES in steering space.** Parameterise the steering vector as a
   low-dimensional combination of k directions (concept features or PCA
   basis) with a coefficient vector z. CMA-ES optimises z. Fitness of z =
   the mean AOCC of algorithms generated under steering z (a two-level
   search: an optimiser in latent space on top of LLaMEA in code space).
   Start with k ≈ 8–16 and a small inner budget.

Records already log the steering configuration (`record["steering"]`), and
`02_build_dataset.py` marks steered rows.

## Libraries to add

`cma` (outer loop); `steering-vectors` / `repeng` for baselines (optional).

## Status

Planned.
