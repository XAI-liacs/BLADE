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

## Status

Planned; starts after the experiment 1 pilot.
