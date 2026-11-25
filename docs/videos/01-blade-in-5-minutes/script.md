# Episode 1: BLADE in 5 minutes

**Goal:** a viewer who has never seen BLADE installs it, creates a first
experiment, checks it and starts a run, and knows where the results are.

**Length:** about 5 minutes (about 650 words of narration).

**Setup before recording:**

- An empty folder, `uv` installed, Ollama running with
  `qwen2.5-coder:14b` pulled (`ollama pull qwen2.5-coder:14b`).
- `export MLFLOW_DISABLE_AGENT_HINT=1`.
- The evaluation environment already cached (run `iohblade check` once
  before recording), unless you want to show the first-time build.

Outputs below were captured on `main` (1.2.0 release candidate) on
25 November 2026. Timings are from a cloud container and will differ.

---

## Scene 1 — Hook (0:00–0:25)

**Screen:** the BLADE banner, then fade to the platform figure.

**Narration:**

> Large language models can now write optimisation algorithms, and then
> improve them, generation after generation. Methods like LLaMEA,
> FunSearch and EoH do exactly that. But which of these methods works
> best, and on which problems? That is the question BLADE answers. In the
> next five minutes we set up and run a first benchmark.

## Scene 2 — What BLADE does (0:25–1:05)

**Screen:** the platform figure, highlighting the columns from left to
right as they are named.

**Narration:**

> BLADE is a benchmark suite for LLM-driven algorithm design. You choose
> three things: an LLM, one or more design methods, and the problems they
> should solve. BLADE runs every method with the same LLM and the same
> evaluation budget, evaluates every generated algorithm in an isolated
> process, and logs the code, the score and where each algorithm came
> from. At the end you compare the methods, and you can look at what was
> actually designed.

## Scene 3 — Install (1:05–1:30)

**Screen:** terminal.

```console
$ uv init blade-demo && cd blade-demo
$ uv add iohblade
```

**Narration:**

> BLADE is a Python package, `iohblade` on PyPI. We use `uv`, but
> `pip install iohblade` works too. After installing, you have an
> `iohblade` command.

## Scene 4 — A first experiment (1:30–2:35)

**Screen:** terminal, then the YAML file in an editor.

```console
$ uv run iohblade init my-first-benchmark
Created a 'builtin' experiment:
  my-first-benchmark/experiment.yml
  my-first-benchmark/reference_solution.py
  my-first-benchmark/experiment.schema.json

Next: edit the TODOs, then
  uv run iohblade check my-first-benchmark/experiment.yml
  uv run iohblade run my-first-benchmark/experiment.yml
```

Open `experiment.yml`; scroll slowly through the four sections.

**Narration:**

> `iohblade init` creates a starting experiment. Everything is in one
> YAML file.
>
> At the top, the experiment itself: where results go, three runs per
> method, and a budget of twenty generated algorithms per run.
>
> Then the LLM. Here a local Qwen model through Ollama, so this runs on
> your own machine. Hosted models such as OpenAI, Claude or Gemini are one
> line away; their API keys come from the environment, never from this
> file.
>
> Then the methods: LLaMEA, an evolutionary method where the LLM refines
> and combines algorithms, and random search as a baseline. Every method
> should beat random search.
>
> And finally the problem: circle packing. Fill a circular container with
> circles of given sizes and cover as much area as possible. The LLM has
> to write the packing algorithm.

*(If editors show the schema: point out the completion and hover hints
from `experiment.schema.json`.)*

## Scene 5 — Validate (2:35–2:55)

```console
$ uv run iohblade validate my-first-benchmark/experiment.yml
Output directory: .../my-first-benchmark/results
Methods (2):
  - LLaMEA-local (LLM: local, budget: 20)
  - RS-local (LLM: local, budget: 20)
Problems (1):
  - circle_packing_n40
Seeds: [0, 1, 2]
Runs: 2 methods x 1 problems x 3 seeds = 6

Configuration is valid.
```

**Narration:**

> `validate` reads the file and tells us what will run: two methods, one
> problem, three seeds, six runs. Nothing is executed yet. A typo here
> gives an error with the line number, before you spend a single token.

## Scene 6 — Check (2:55–3:30)

```console
$ uv run iohblade check my-first-benchmark/experiment.yml
Checking my-first-benchmark/experiment.yml ...

[OK  ] problem circle_packing_n40: evaluated smoke_test_solution reference_solution.py in 3.9s, fitness 579.05 (4.3s)
[OK  ] method LLaMEA-local: ran on 'circle_packing_n40' with budget 2 (2.8s)
[OK  ] method RS-local: ran on 'circle_packing_n40' with budget 2 (2.6s)

3 ok - ready to run.
(LLMs were not contacted; add --llm to test them.)
```

**Narration:**

> `check` is a smoke test. It evaluates a known reference solution on the
> problem, so we know the evaluation works, and runs each method for two
> steps with a stand-in for the LLM. No LLM calls, so it is free and
> fast. Add `--llm` to also test the connection to your model.

## Scene 7 — Run (3:30–4:25)

```console
$ uv run iohblade run my-first-benchmark/experiment.yml --set experiment.runs=1 --set experiment.budget=10
```

Record the full run; show it as a time-lapse with an on-screen note
"time-lapse, about N minutes" (fill in the real duration).

Expected end of output (shape; numbers will differ):

```text
Run overview:
Method        LLM                Problem             Seed  Evals  Best  Errors
------------------------------------------------------------------------------
LLaMEA-local  qwen2.5-coder-14b  circle_packing_n40  0     10/10  ...   ...     ✅
RS-local      qwen2.5-coder-14b  circle_packing_n40  0     10/10  ...   ...     ✅
Errors: evaluations that failed in the generated code (c), in the harness (h) or by timeout (t).

Summary: 2 of 2 runs finished.
Evaluations: 20, of which failed: ...
Logs: .../results/experiment.log and run.log in each run's folder.

Results: .../my-first-benchmark/results
```

**Narration:**

> `run` starts the experiment. For this video we override two settings
> from the command line: one run and a budget of ten, so it finishes
> quickly. The overview shows each run, how many algorithms were
> evaluated, the best score so far, and how many evaluations failed.
> Failures are normal: generated code does not always work, and that is
> part of what we measure.
>
> If you stop a run, run the same command again and BLADE resumes where
> it left off.

## Scene 8 — First result (4:25–5:05)

```console
$ uv run iohblade-webapp
```

**Screen:** the browser: the experiment, the convergence plot, then the
code of the best algorithm.

**Narration:**

> Everything is in the results folder: every generated algorithm, its
> score and its parent. The web app gives a quick look: convergence per
> method, and with one click the code of the best algorithm the LLM wrote.
> We go into the analysis in a later episode.

## Scene 9 — Outro (5:05–5:25)

**Screen:** the banner with the repository URL and `pip install iohblade`.

**Narration:**

> That is BLADE in five minutes: init, validate, check, run. In the next
> episode we look at the YAML file in detail: more methods, the built-in
> benchmark suites, and how to scale up. Links to the code and the
> documentation are below.

---

## Notes for the recording

- The LLM column shows the model name, not the provider key.
- `check` builds the evaluation environment on first use, which can take a
  minute; cache it before recording or cut the wait.
- Scene 7 with a 14B model on one GPU: plan for the real run to take
  several minutes; record once, then time-lapse.
