# Episode 2: Experiments in YAML

**Goal:** the viewer can write a comparison experiment from scratch: several
LLMs and methods, built-in suites, seeds and budgets; they can fix errors
from `validate`, try variants with `--set`, and resume a stopped run.

**Length:** about 6 minutes including the standard opening.

**Files:** `experiment.yml` (the experiment built in this episode) and
`typos.yml` (the same file with two mistakes) in this folder.

**Setup before recording:**

- A folder with `experiment.yml` and `typos.yml`, plus the
  `my-first-benchmark/` folder from episode 1 with its (short) results.
- `GEMINI_API_KEY` set in the shell (any value works for `validate`).
- An editor with YAML support (VS Code with the Red Hat YAML extension shows
  the schema completion).

Outputs below were captured on `main` on 25 November 2026; the suite
names (`bbob_groups`) need BLADE 1.2.1.

---

## Scene 0 — Standard opening (0:00–0:15)

The shared opening from `docs/videos/README.md`, then the title card
"Episode 2 · Experiments in YAML".

## Scene 1 — Where we left off (0:15–0:35)

**Screen:** episode 1's `experiment.yml`, then an empty editor tab.

**Narration:**

> In the first episode we ran the starter experiment. Now we write our own:
> a real comparison between methods and LLMs, on a set of benchmarks. One
> YAML file, four sections: the experiment, the LLMs, the methods and the
> problems.

## Scene 2 — The experiment section (0:35–1:15)

**Screen:** type the first section; let the editor's completion pop up.

```yaml
# yaml-language-server: $schema=./experiment.schema.json
version: 1

experiment:
  output_dir: results/compare
  runs: 5
  budget: 100
  n_jobs: 4
```

**Narration:**

> The first line connects the editor to BLADE's schema, so we get
> completion and documentation while typing.
>
> `output_dir` is where everything goes. `runs: 5` means five independent
> runs per method and problem, with seeds zero to four; you can also list
> the seeds yourself. `budget` is the number of algorithms each run may
> generate and evaluate: the same for every method, which is what makes
> the comparison fair. And `n_jobs` runs four of them in parallel.

## Scene 3 — LLMs (1:15–1:45)

```yaml
llms:
  qwen:
    provider: ollama
    model: "qwen2.5-coder:14b"
  gemini:
    provider: gemini
    model: gemini-2.0-flash
    api_key: "${env:GEMINI_API_KEY}"
```

**Narration:**

> Under `llms` we give each model a name of our choice. A local Qwen model
> through Ollama, and Gemini through Google's API. The key is not in the
> file: `${env:...}` reads it from the environment or from a `.env` file
> next to the experiment. That way you can share the experiment without
> sharing your keys. The next episode is all about LLMs.

## Scene 4 — Methods (1:45–2:30)

```yaml
methods:
  - type: llamea
    name: "LLaMEA-{llm}"
    llm: [qwen, gemini]
  - type: shinka_evolve
    name: "Shinka-{llm}"
    llm: qwen
  - type: random_search
    name: "RS-{llm}"
    llm: qwen
```

Cut to the terminal for a moment:

```console
$ uv run iohblade list methods
```

**Narration:**

> Methods refer to the LLMs by name. Giving LLaMEA a list of two LLMs
> creates two methods, one per model, and `{llm}` in the name is filled in,
> so we can compare the same method with two different models.
>
> We add ShinkaEvolve and, as always, random search as a baseline.
> `iohblade list` shows every built-in method, problem, suite and LLM
> provider. Method settings go under `params`; without them, each method
> uses its defaults.

## Scene 5 — Problems and suites (2:30–3:15)

```yaml
problems:
  - suite: bbob_groups
    params: {dim: 5, budget_factor: 1000}
  - type: circle_packing
```

Cut to the terminal:

```console
$ uv run iohblade list suites
```

Show the docs page "Benchmarks" (the suites table) for a few seconds.

**Narration:**

> A problem is either a single benchmark, with `type`, or a whole suite.
> The `bbob_groups` suite gives the five BBOB function groups as five
> problems, with fixed training and test instances, so results are
> comparable between papers; the `bbob` suite has the 24 functions one by
> one. Here we make the groups five-dimensional with a smaller budget for
> the generated algorithms.
>
> Other suites cover the AlphaEvolve maths problems and combinatorial
> problems; `default` combines the BBOB groups with both, `quick` is a
> small mix for trying out a setup, and `full` runs everything. The documentation lists them with estimated running
> times.

## Scene 6 — Validate, and fixing mistakes (3:15–4:10)

First the version with mistakes:

```console
$ uv run iohblade validate typos.yml
Invalid experiment configuration (2 issues):
  - typos.yml:23:5: methods[1].type: Unknown method 'shinka_evolv'. Did you mean 'shinka_evolve'?
  - typos.yml:32:14: problems[0].params.dims: Unknown parameter 'dims' for suite 'bbob_groups'. Did you mean 'dim'?
```

Then the correct file:

```console
$ uv run iohblade validate experiment.yml
Output directory: .../results/compare
Methods (4):
  - LLaMEA-qwen (LLM: qwen, budget: 100)
  - LLaMEA-gemini (LLM: gemini, budget: 100)
  - Shinka-qwen (LLM: qwen, budget: 100)
  - RS-qwen (LLM: qwen, budget: 100)
Problems (6):
  - BBOB-g1-5d
  - BBOB-g2-5d
  - BBOB-g3-5d
  - BBOB-g4-5d
  - BBOB-g5-5d
  - circle_packing_n40
Seeds: [0, 1, 2, 3, 4]
Runs: 4 methods x 6 problems x 5 seeds = 120

Configuration is valid.
```

**Narration:**

> Before running anything, we validate. Mistakes are reported with the
> file, line and column, and a suggestion. Here a misspelled method and a
> parameter the suite does not have. Parameters are checked against the
> actual code, so a typo never silently falls back to a default.
>
> Fixed, `validate` shows the plan: four methods, six problems, five
> seeds, 120 runs. That number is worth a look before you start: it tells
> you how much compute and how many LLM calls you are about to spend.

## Scene 7 — Variants with --set (4:10–4:45)

```console
$ uv run iohblade run experiment.yml --dry-run --set experiment.runs=1
...
Seeds: [0]
Runs: 4 methods x 6 problems x 1 seeds = 24
Output directory is new: all runs will start.
```

**Narration:**

> Any value can be changed from the command line with `--set`, without
> editing the file. Handy for a quick test: one run instead of five.
> `--dry-run` shows what would happen without running anything. For a
> real quick test, also give it its own `output_dir`, so the test results
> do not end up in the main experiment.

## Scene 8 — Resuming (4:45–5:35)

Back in episode 1's folder; its quick run (one run per method, budget 10)
is in `results/quick`:

```console
$ uv run iohblade run my-first-benchmark/experiment.yml --dry-run --set experiment.output_dir=results/quick --set experiment.budget=10
...
Runs: 2 methods x 1 problems x 3 seeds = 6
Resuming: 2 of 6 runs already finished and will be skipped; 4 will run.
```

Then the same folder with the file's own budget of 20:

```console
$ uv run iohblade run my-first-benchmark/experiment.yml --dry-run --set experiment.output_dir=results/quick
...
Cannot resume: The output directory .../results/quick holds runs with another budget than this configuration. Resuming would mix budgets in one comparison.
    Budget changed for 2 started run(s): LLaMEA-local (qwen2.5-coder-14b) on circle_packing_n40, seed 0: 10 recorded, 20 planned; RS-local (qwen2.5-coder-14b) on circle_packing_n40, seed 0: 10 recorded, 20 planned.
    Either use a new output directory (change experiment.output_dir, e.g. --set experiment.output_dir=results/trial), or restore the previous budget to resume.
```

**Narration:**

> Long experiments get interrupted: a reboot, a cluster time limit, an API
> outage. Running the same command again resumes: finished runs are
> skipped, the rest starts. You can also add runs, methods or problems
> later, and only the new ones run. Here, two runs are done and four more
> would be added.
>
> What BLADE will not do is mix two different setups in one folder. Our
> quick test used a budget of ten; the file says twenty. Resuming would
> compare runs with different budgets as if they were the same, so BLADE
> stops before writing anything and says what differs. The same happens
> if you remove runs, methods or problems. Every run also records the
> exact configuration it used, so you can always see how a result was
> produced.

## Scene 9 — Outro (5:35–5:55)

**Screen:** the full `experiment.yml`, then the banner.

**Narration:**

> That is the whole format: experiment, LLMs, methods, problems; validate,
> dry-run, run. Next episode: choosing LLMs, from hosted APIs to open
> models on your own GPUs.

---

## Notes for the recording

- The `--dry-run` outputs above were captured with a test LLM; the method
  names and run counts are the same with real models.
- In scene 8, `my-first-benchmark/results/quick` must hold episode 1's
  quick run (one run per method, budget 10). The budget check needs the
  fix on branch `fix-plots-registry-resume` (in the 1.2.0 release).
- `validate` exits with code 2 for an invalid file, 0 for a valid one.
