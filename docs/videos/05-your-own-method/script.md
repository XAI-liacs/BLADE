# Episode 5: Your own method

**Goal:** the viewer implements their own LLM-driven search method as a
`Method` subclass, follows the rules that keep comparisons fair (budget,
optimisation direction, returning the best solution), and benchmarks it
against a built-in baseline.

**Length:** about 6 minutes including the standard opening.

**Setup before recording:**

- An empty folder; BLADE installed.
- Read `docs/tutorial_custom_method.rst` once; this episode follows it.

Outputs below were captured on `main` on 25 November 2026.

---

## Scene 0 — Standard opening (0:00–0:15)

Shared opening, then the title card "Episode 5 · Your own method".

## Scene 1 — Why your own method (0:15–0:40)

**Narration:**

> If you are developing a new method for LLM-driven algorithm design, the
> hardest part of the paper is often the comparison: running the other
> methods fairly, on the same problems, with the same budget. With BLADE
> you write your method as one class, and the comparison comes for free.

## Scene 2 — The template (0:40–1:10)

```console
$ uv run iohblade init my-method --template custom-method
Created a 'custom-method' experiment:
  my-method/experiment.yml
  my-method/my_method.py
  my-method/reference_solution.py
  my-method/experiment.schema.json
```

```yaml
methods:
  - type: custom
    source: my_method.py
    class: MyMethod
    name: "MyMethod-{llm}"
    llm: local
  - type: random_search            # a baseline to compare with
    name: "RS-{llm}"
    llm: local
```

**Narration:**

> The `custom-method` template sets up our method next to random search,
> on circle packing. Our method is `type: custom` with its file and class,
> and it gets an LLM like any built-in method.

## Scene 3 — A (1+1) strategy (1:10–2:30)

**Screen:** `my_method.py`, `__call__`.

```python
class MyMethod(Method):
    def __init__(self, llm, budget, name="MyMethod", improve_prompt=None):
        super().__init__(llm, budget, name)
        self.improve_prompt = improve_prompt or (
            "Improve the following solution. Feedback from its evaluation: "
        )

    def __call__(self, problem: Problem):
        minimise = getattr(problem, "minimisation", False)
        prompt = [{"role": "user", "content": problem.get_prompt()}]
        best = problem(self.llm.sample_solution(prompt))
        for _ in range(self.budget - 1):
            message = (
                f"{problem.get_prompt()}\n{self.improve_prompt}{best.feedback}\n"
                f"```python\n{best.code}\n```"
            )
            child = problem(
                self.llm.sample_solution([{"role": "user", "content": message}])
            )
            if better(child.fitness, best.fitness, minimise):
                best = child
        return best
```

**Narration:**

> The template is a simple (1+1) strategy. Ask the LLM for a first
> solution using the problem's prompt. Then, again and again, show it the
> best solution so far with its feedback, and ask for an improvement. Keep
> the child if it is better.
>
> Two calls do all the work. `self.llm.sample_solution` asks the LLM and
> turns its answer into a solution: code, name and description.
> `problem(...)` evaluates it: safely, in a separate process, and logged.

## Scene 4 — The rules (2:30–3:40)

**Screen:** highlight the lines as they are mentioned.

**Narration:**

> Three rules keep the comparison fair.
>
> One: every call to `problem(...)` is one evaluation, and a run has
> `budget` of them. Here the first solution plus `budget - 1`
> improvements. When the budget is used up, further evaluations return a
> failure score, so going over gains nothing.
>
> Two: respect the direction. Some problems are minimised, most are
> maximised; `problem.minimisation` tells you which. Always compare
> fitness with that in mind, like the `better` helper does here.
>
> Three: return the best solution you found. BLADE records every
> evaluation anyway, but the returned solution is what the run reports.
>
> Settings of your method become constructor arguments, like
> `improve_prompt`, and can be set from the YAML file under `params`.
> `get_config` records them in the results, so every run documents how it
> was configured.

## Scene 5 — Check and run (3:40–4:50)

```console
$ uv run iohblade check my-method/experiment.yml
Checking my-method/experiment.yml ...

[OK  ] problem circle_packing_n40: evaluated smoke_test_solution reference_solution.py in 1.4s, fitness 579.05 (1.4s)
[OK  ] method MyMethod-local: ran on 'circle_packing_n40' with budget 2 (2.9s)
[OK  ] method RS-local: ran on 'circle_packing_n40' with budget 2 (2.9s)

3 ok - ready to run.
(LLMs were not contacted; add --llm to test them.)
```

```console
$ uv run iohblade run my-method/experiment.yml
```

Time-lapse of the run overview to the end.

**Narration:**

> `check` runs our method for two steps with a stand-in LLM. That catches
> crashes and interface mistakes in a few seconds. Then we run it for real
> against random search. If your method does not beat random search with
> the same budget, that is worth knowing early.

## Scene 6 — Beyond the template (4:50–5:30)

**Screen:** the methods list on the docs page, then the platform figure.

**Narration:**

> From here, the method is yours: a population, crossover between
> solutions, a different prompt strategy. And because it is a BLADE method,
> you can run it against LLaMEA, EoH, FunSearch, ShinkaEvolve and the
> others, on all built-in suites and with any LLM, by adding it to the
> methods list.

## Scene 7 — Outro (5:30–5:50)

**Narration:**

> Next episode: the results. How to look into an experiment, compare
> methods, and see what the LLM actually designed.

---

## Notes for the recording

- The budget behaviour in scene 4 (failure score after the budget) is
  implemented in `Problem.__call__`; the template never reaches it.
