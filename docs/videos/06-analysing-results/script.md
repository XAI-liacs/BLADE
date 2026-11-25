# Episode 6: Analysing results

**Goal:** the viewer can open a finished (or running) experiment in the web
app, compare methods, follow how algorithms evolved, read the LLM
conversations and download the best algorithms; and make the same plots and
tables from Python for a paper.

**Length:** about 6.5 minutes including the standard opening.

**Setup before recording:**

- A finished experiment with real results: at least two methods, two or
  more problems and three or more seeds (the episode 2 experiment, or a
  smaller version of it). Results where every run failed make the plots
  empty.
- A Jupyter notebook or Python file with the code from scene 6, tested on
  those results.

Commands and code below were checked on `main` on 25 November 2026, on a
small experiment with a test LLM.

---

## Scene 0 — Standard opening (0:00–0:15)

Shared opening, then the title card "Episode 6 · Analysing results".

## Scene 1 — More than a score (0:15–0:45)

**Narration:**

> At the end of an experiment, you want to know which method won. But in
> algorithm design, the more interesting question is often *what* was
> designed, and how the search got there. BLADE keeps everything: every
> generated algorithm, its code, its score, its feedback and its parent.
> In this episode we look at that, first in the web app, then in Python.

## Scene 2 — The web app (0:45–1:40)

```console
$ cd my-experiment
$ uv run iohblade-webapp
```

**Screen:** the browser. Sidebar with the experiments; select one; the run
table and the progress per run.

**Narration:**

> The web app looks for experiments in the `results` folder of the
> directory you start it in. It also works while an experiment is running,
> so you can watch progress live.
>
> For each experiment we see every run: method, problem, seed, how far it
> is, and the best score. Under "Configuration" is the exact setup the
> experiment ran with, including the code of custom problems and methods,
> and the experiment log is one click away.

## Scene 3 — Comparing methods (1:40–2:40)

**Screen:** the convergence section: select methods and problems, toggle
"Aggregate runs".

**Narration:**

> The convergence plot shows the best score so far against the number of
> evaluations, per method. With "Aggregate runs" we see the mean over
> seeds with a band around it, so we can tell a real difference from luck.
> Always look at several seeds: LLM-driven search is noisy, and one lucky
> run says little.
>
> The x-axis is evaluations, not time, because the budget is the same for
> every method. That is the fair comparison.

## Scene 4 — Code evolution (2:40–3:50)

**Screen:** "Code Evolution Graph": choose a method, a problem and a run;
feature "Total Token Count"; click "Plot Evolution Graph". Hover over a
node.

**Narration:**

> This is my favourite view. Every dot is an algorithm the LLM wrote; the
> lines show which algorithm it came from. Left to right is the order in
> which they were written, the colour is the score, and the height a
> property of the code, here its length in tokens. Bigger dots had more
> children.
>
> You can see how the search works: does it refine one line of algorithms,
> or jump between ideas? Do the algorithms keep getting longer without
> getting better? Other features show the complexity of the code or its
> number of parameters.
>
> This is where explainability comes in: we do not only want a good
> algorithm, we want to understand where it came from.

## Scene 5 — The algorithms and the conversations (3:50–4:40)

**Screen:** "Top Solutions": expand a method, download the best algorithm,
open it in the editor. Then "Run Logs": open a "Conversation Log".

**Narration:**

> Under "Top Solutions", the best algorithm per method and problem, ready
> to download as a Python file. Read it: often you recognise known ideas,
> sometimes something new.
>
> And the run logs hold the full conversation with the LLM: every prompt,
> every answer. When a method behaves strangely, this is where you find
> out why.

## Scene 6 — The same in Python (4:40–5:50)

**Screen:** a notebook.

```python
from iohblade.loggers import ExperimentLogger
from iohblade.plots import fitness_table, plot_boxplot_fitness, plot_convergence

logger = ExperimentLogger("results/compare", read=True)

plot_convergence(logger)      # best-so-far per method, mean over seeds
plot_boxplot_fitness(logger)  # final scores per method and problem
fitness_table(logger)         # mean ± std per cell, best in bold (with p-values)
```

**Narration:**

> For a paper you want the plots as files and the numbers in a table. The
> same results open in Python with `ExperimentLogger` in read mode. The
> plots module has the convergence plot, box plots of the final scores,
> and a table with mean and standard deviation per method and problem,
> where a method is printed in bold when it is significantly better than
> all others.
>
> For continuous problems, the behaviour metrics module goes one level
> deeper: it describes how a generated optimiser searches, for example how
> much it explores versus exploits.

## Scene 7 — Outro (5:50–6:10)

**Narration:**

> In the last episode we let a coding agent do the setup work: from a
> one-sentence request to a checked experiment.

---

## Notes for the recording

- The web app reads `./results`: start it in the folder that holds the
  experiment's `results/` directory (for episode 1, `my-first-benchmark/`).
- `plot_boxplot_fitness` raised a `ValueError` on an experiment where every
  run had failed (episode 1 with the test LLM); it works on normal results.
- `fitness_table` returns a pandas table; its docstring describes the
  significance test.
- A public leaderboard is planned (#236); mention it only once it exists.
