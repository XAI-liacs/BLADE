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

## Libraries to add

`circuit-tracer` (git), `inseq` or `captum`.

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

### Next steps
- **Generation-level counterfactuals.** Sample new responses with the
  score changed, or with the error removed, and compare the edits (AST
  diff) and the child fitness. Does the over-budget message that MA-BBOB
  omits change behaviour (design item 4)?
- **Attribution graphs** (`circuit-tracer` + `mwhanna/qwen3-8b-transcoders`)
  on selected examples: how does the model route from the error text to the
  changed line?
- **Link to experiment 1:** the fitness representation at layer 15 is not
  used for the edit. Patch the parent-fitness direction and check whether
  the edit changes.

## Status

Step 1 done (span deletion, score counterfactual, gradient × input).
