# Experiment 2: Is the feedback actually used?

**Question.** When LLaMEA asks the model to mutate an algorithm, how much of
the edit is driven by the **feedback** (score, error trace), by the **parent
code**, by the **population summary** and by the **operator** instruction?
Does richer feedback get used, or ignored?

**Builds on.**
- Experiment 1's recorded runs: prompts and responses are in
  `mi_activations/index.jsonl`, and `HookedLLM.replay` gives per-token states.
- Experiment 1's probes: the layers where parent fitness is decodable are
  the natural place to look.

## Design (draft)

1. **Segment** every mutation prompt into spans: task, population summary,
   parent description, parent code, feedback, error and operator. The
   markers come from `llamea.LLaMEA.construct_prompt`.
2. **Attribution** from the generated code (or from chosen output tokens,
   e.g. the first changed line) back to the spans:
   - gradient × input / integrated gradients (`inseq` or `captum`) as a
     baseline;
   - attribution graphs with `circuit-tracer` and the
     `mwhanna/qwen3-8b-transcoders` (per-layer transcoders), on selected
     examples;
   - causal checks: ablate or patch a span (e.g. swap the feedback
     between two parents) and measure how the edit and the child fitness
     change.
3. **Compare operators** (`refine` / `new` / `simplify`): `new` should rely
   less on the parent code.
4. **Feedback variants** (new runs): plain score; score + error; score +
   BLADE behaviour metrics; and the explicit over-budget message that
   MA-BBOB currently omits (see the top-level README). Does attribution to
   the feedback grow with its information content, and does fitness follow?

## Run

```bash
source experiments/mi/env.sh
S=experiments/mi/exp02_feedback_attribution/scripts
# span deletion + score counterfactuals + gradient x input (one GPU, ~25 min)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python $S/01_span_attribution.py --tag main --n-per-operator 120
uv run python $S/02_summarise.py --tag main
uv run jupyter lab experiments/mi/exp02_feedback_attribution/notebooks/01_span_attribution.ipynb
```

`spans.py` splits a mutation prompt into seven spans (task, population,
parent description, parent code, feedback, operator, format). All 1152
mutation prompts of experiment 1 `main` segment cleanly.

## Results

### Step 1 (2026-11-04): span deletion on 360 recorded queries

The model re-reads each prompt and its own response (teacher forcing). The
measure is how much less likely the response becomes, in nats per response
token, when one span is deleted, or when the parent's score in the
feedback is replaced by 0.010 or 0.900. 120 queries per operator, from
experiment 1 `main`. 95% bootstrap CIs over runs. Figure:
`results/main/span_attribution.png`.

| Span (deleted) | refine | new | simplify | gradient × input share |
|---|---|---|---|---|
| parent code | 0.331 | 0.330 | 0.361 | 52% |
| output format | 0.062 | 0.061 | 0.070 | 8% |
| population summary | 0.015 | 0.015 | 0.016 | 16% |
| task description | 0.011 | 0.017 | 0.012 | 11% |
| operator instruction | 0.003 | 0.004 | 0.009 | 1.4% |
| parent description | 0.006 | 0.007 | 0.006 | 5% |
| feedback | 0.002 | 0.002 | 0.002 | 2% |

1. **The response is built from the parent code.** Deleting it costs
   0.33–0.36 nats per token, 5× more than any other span. This holds for
   every operator.
2. **The `new` operator does not make the model start over.** Its
   responses depend on the parent code exactly as much as `refine`'s do.
   The children are rewrites of the parent: median code similarity 0.80
   (refine 0.85, simplify 0.94). Only 8% keep the class name, so they
   *look* new.
3. **The operator instruction barely matters** (0.003–0.009 nats per
   token).
4. **The score in the feedback is ignored.** Replacing 0.196 by 0.010 or
   0.900 changes the response's log-probability by < 0.001 nats per token
   (score_high: −0.0000 [−0.0001, 0.0001]; score_low: 0.0004 [0.0003,
   0.0005]). The model does not condition its edit on how good the parent
   is, although experiment 1 shows it *represents* parent fitness (ρ ≈ 0.7).
   So the representation is there but unused in the edit.
5. **Feedback matters when it is an error message.** When the parent
   failed, deleting the feedback costs 0.027 vs 0.001 nats per token,
   20× more.

Caveat: deletion measures how much the *realised* response depends on a
span. Copying from the parent code makes that span dominant almost by
construction. The score counterfactual is the cleaner test of feedback
use, because it changes only the information.

Caveat (found in step 3): the score counterfactual of step 1 changed the
score only in the feedback. The population summary still listed the true
score at full precision, so "ignored" could mean "read elsewhere". Step 3
changes both; the conclusion holds (below).

### Steps 3–5 (2026-11-05): generation counterfactuals

192 recorded mutation queries, each resampled 4 times per prompt variant
with the `main` settings (T = 0.8, top-p = 0.95), and every child evaluated
on the same budget-enforced MA-BBOB problem: 4180 children, 0
infrastructure errors, ~4 GPU-hours on 4 GPUs.

- **Scored parents** (96 queries from `main`, 32 per operator, 12 runs):
  score set to 0.010 or 0.900 (written change −0.44 / +0.45 AOCC on
  average) in **both** the feedback and the population list; feedback
  deleted; explicit over-budget message (85 queries whose parent
  overspent); steering ±2·v at layer 15 (all positions), where v is the
  difference of mean activations between top- and bottom-tercile parents
  (|v| = 4.8, 8% of the activation norm); a random direction of the same
  norm as a control.
- **Crashed parents** (96 queries, pooled from `main` and the shadow
  runs of `prescreen`/`prescreen2`, 32 runs): error deleted; error
  replaced by a generic "the algorithm crashed" message.

The sampling seed depends only on the query and replicate (common random
numbers), so effects are paired. Effect = variant − full, averaged within a
query; 95% bootstrap CI over LLaMEA runs. `pf_L15` is experiment 1's
parent-fitness probe read out during the resampled query. Figures:
`results/cf/counterfactuals_scored.png`, `counterfactuals_error.png`.

**Scored parents**

| Variant | `pf_L15` readout | child fails | working child overspends | child fitness | similarity to parent |
|---|---|---|---|---|---|
| score → 0.010 | −0.039 [−0.043, −0.036] | +0.013 (ns) | −0.011 (ns) | −0.003 (ns) | −0.016 [−0.026, −0.007] |
| score → 0.900 | +0.038 [+0.034, +0.044] | +0.023 (ns) | −0.010 (ns) | −0.012 (ns) | −0.001 (ns) |
| feedback deleted | −0.017 | +0.016 (ns) | −0.004 (ns) | +0.003 (ns) | +0.004 (ns) |
| over-budget message | +0.002 (ns) | +0.018 (ns) | **−0.062 [−0.090, −0.037]** | +0.013 (ns) | +0.002 (ns) |
| steer fitness − | −0.83 | −0.021 (ns) | −0.008 (ns) | +0.021 (ns) | **+0.025 [+0.007, +0.042]** |
| steer fitness + | +0.83 | +0.039 (ns) | +0.010 (ns) | −0.008 (ns) | **−0.032 [−0.044, −0.019]** |
| steer random | −0.065 | **+0.055 [+0.012, +0.098]** | −0.015 [−0.030, −0.002] | −0.014 (ns) | +0.005 (ns) |

Base rates (`full`): child fails 12.8%, working child overspends 93%,
child fitness 0.330, similarity 0.85.

1. **The written score barely reaches the fitness representation.** A
   ±0.44 AOCC change in the written score, in both places, moves the
   internal readout by ±0.04: under a tenth. The child's failure rate,
   overspending and fitness do not change. Step 1's conclusion survives
   the confound: the model forms its fitness estimate from the code, not
   from the number.
2. **The over-budget message works, modestly.** Telling the model that
   the parent exceeded the budget cuts overspending children of those
   parents from 99.6% to 93.5%, with no effect on failures or fitness. Most children still
   overspend: the message does not make the model write a budget check.
3. **The fitness direction is causal for edit size, in a surprising
   direction.** Steering towards "good parent" makes the child *less*
   similar to the parent (−0.032), steering towards "bad parent" makes it
   *more* similar (+0.025); the random direction does neither (+0.005),
   so the effect is specific to v. The written score acts the other way
   round (low score → slightly larger rewrite). Steering does not change
   child fitness detectably.
4. **Steering has a generic cost.** The random direction raises failures
   by 5.5 points, so any push of this size (8% of the norm) degrades
   code. The fitness-direction effects on failure are within that noise.

**Crashed parents**

| Variant | child fails | fails with the parent's error type | error line kept | similarity | child fitness |
|---|---|---|---|---|---|
| full error message | 0.432 | 0.247 | 0.351 | 0.778 | 0.119 |
| error deleted | **+0.224 [+0.137, +0.328]** | **+0.214 [+0.140, +0.301]** | **+0.197 [+0.043, +0.359]** | −0.062 [−0.080, −0.043] | −0.044 [−0.080, −0.019] |
| generic "crashed" message | **+0.164 [+0.089, +0.250]** | **+0.154 [+0.092, +0.222]** | **+0.144 [+0.011, +0.285]** | −0.031 [−0.048, −0.011] | −0.025 [−0.047, −0.004] |

5. **The error message is used to fix the code.** Without it the child
   fails 22 points more often, almost entirely with the parent's own
   exception type (+21), and keeps the offending line 20 points more
   often. The model rewrites more but blindly (similarity −0.06).
6. **The content of the error matters, not only the fact of a crash.** A
   generic "it crashed" message recovers 27% of the effect on failure;
   the exception type, message and line carry the rest.

So the feedback that LLaMEA sends is used when it is an error trace and
ignored when it is a score. This matches step 1's span deletion (error
feedback 20× more influential than score feedback), now with behavioural
consequences.

### Step 6: attribution graphs

`06_attribution_graphs.py` (circuit-tracer 0.5 with the
`mwhanna/qwen3-8b-transcoders`, separate venv because circuit-tracer
needs transformers 4.x). For recorded children that work, two targets per
query: the **edit token** (first token where the child departs from the
parent's code, preferring an edit on the line named in the error) and a
**copy token** (first token of the nearest earlier line copied unchanged)
as a control. Influence of each node (feature, error node, token
embedding) on the target logit, summed by the prompt span of its
position. 24 crashed-parent and 12 scored-parent queries, shortest
contexts first (1000–1600 tokens).

Results: running (2026-11-05). The first 10 graphs (crashed parents)
are in `data/processed/cf/attribution_graphs-*.jsonl`.

### Next steps
- Look at the top error-span features (`top_feedback_features`) on
  Neuronpedia-style dashboards: do they encode the exception type, the
  line, or "something went wrong"?
- Per-operator split of the counterfactual effects (32 queries per
  operator is thin; pool more queries if needed).
- Experiment 3: richer feedback (BLADE behaviour metrics, budget usage
  per instance) as new LLaMEA runs, since errors are used but the score
  is not.

## Run (steps 3–6)

```bash
source experiments/mi/env.sh
S=experiments/mi/exp02_feedback_attribution/scripts
# 192 queries x variants, steering vector (CPU, ~1 min)
uv run python $S/03_prepare_counterfactuals.py --tag cf
# 4180 samples: 12 shards on 4 GPUs, ~4 h
bash $S/launch.sh cf 12 4
uv run python $S/05_summarise_counterfactuals.py --tag cf
# attribution graphs: separate venv (circuit-tracer + transformers 4.x);
# one graph needs a whole 96 GB GPU (batch size 4), ~9 min each
SCRIPT=06_attribution_graphs.py PYTHON=<circuit venv>/bin/python \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash $S/launch.sh cf 4 4
uv run jupyter nbconvert --execute --inplace \
  experiments/mi/exp02_feedback_attribution/notebooks/02_counterfactuals.ipynb
```

Memory note for circuit-tracer: the forward pass expands the prompt to
`batch_size` copies and keeps the attention patterns of all 36 layers for
the backward pass (~5 GB per copy at 1500 tokens), on top of ~40 GB for
the model and the active transcoder rows. Batch size 4 peaks at 91 GB.

## Status

Steps 1–5 done. Step 6 (attribution graphs) running.
