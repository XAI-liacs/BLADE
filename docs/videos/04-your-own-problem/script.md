# Episode 4: Your own problem

**Goal:** the viewer turns their own optimisation problem into a BLADE
benchmark: the prompt the LLM sees, an example solution, the `evaluate()`
method that scores generated code, and `iohblade check` to catch mistakes
before spending LLM calls.

**Length:** about 7 minutes including the standard opening.

**Setup before recording:**

- An empty folder; BLADE installed.
- Read `docs/tutorial_custom_problem.rst` once; this episode follows it.

Outputs below were captured on `main` on 25 November 2026.

---

## Scene 0 — Standard opening (0:00–0:15)

Shared opening, then the title card "Episode 4 · Your own problem".

## Scene 1 — Why your own problem (0:15–0:40)

**Narration:**

> The built-in benchmarks are good for comparing methods. But most people
> come to BLADE with a problem of their own: a scheduling task, a
> simulator, a design problem. In this episode we turn such a problem into
> a BLADE benchmark. It is one Python class.

## Scene 2 — The template (0:40–1:15)

```console
$ uv run iohblade init my-problem --template custom-problem
Created a 'custom-problem' experiment:
  my-problem/experiment.yml
  my-problem/my_problem.py
  my-problem/reference_solution.py
  my-problem/experiment.schema.json
```

Show the `problems` section of `experiment.yml`:

```yaml
problems:
  - type: custom
    source: my_problem.py              # or a package folder
    class: MyProblem
    params: {dim: 5, budget: 200}
    smoke_test_solution: reference_solution.py   # used by `iohblade check`
    dependencies: []                   # extra pip packages for evaluate()
```

**Narration:**

> The `custom-problem` template gives us three files: the experiment, the
> problem class, and a reference solution. In the experiment, our problem
> is `type: custom`, with the file and the class name. `params` go to the
> constructor and are checked against it, just like for built-in problems.

## Scene 3 — What the LLM sees (1:15–2:15)

**Screen:** `my_problem.py`, the constructor.

```python
class MyProblem(Problem):
    def __init__(self, dim: int = 5, budget: int = 200, name="MyProblem"):
        super().__init__(name=name, eval_timeout=60)
        self.dim = dim
        self.budget = budget
        self.minimisation = False
        self.task_prompt = (
            f"Write a Python class that minimises a black-box function of {dim} "
            f"variables in [-5, 5] using at most {budget} function evaluations.\n"
        )
        self.example_prompt = "An example:\n```python\n" + EXAMPLE + "```\n"
        self.format_prompt = "Give the full class in one ```python code block.\n"
```

**Narration:**

> A problem has three prompts. The task prompt describes what the LLM
> should write: here, an optimiser for a black-box function. The example
> prompt shows a working solution, so the model knows the exact interface.
> And the format prompt says how to answer: one code block.
>
> Two more settings. `eval_timeout`: how long one evaluation may take
> before it is stopped. And `minimisation`: whether a lower score is
> better. BLADE maximises by default.
>
> Spend time on the task prompt. It is the only thing the LLM knows about
> your problem: the interface, the constraints, what counts as good.

## Scene 4 — evaluate(): scoring generated code (2:15–3:30)

```python
    def evaluate(self, solution: Solution) -> Solution:
        namespace: dict[str, Any] = {}
        exec(solution.code, namespace)
        algorithm = namespace[first_class_name(solution.code)](
            budget=self.budget, dim=self.dim
        )
        shift = np.random.default_rng(0).uniform(-3, 3, self.dim)

        def func(x):  # TODO: your objective
            return float(np.sum((np.asarray(x) - shift) ** 2))

        best_f, _ = algorithm(func)
        solution.set_scores(-float(best_f), f"Best value found: {best_f:.4g}.")
        return solution
```

**Narration:**

> `evaluate` receives a solution holding the code the LLM wrote. We run the
> code, create the algorithm, let it optimise our function, and score it
> with `set_scores`: a fitness, and feedback in words.
>
> That feedback goes back to the LLM in the next prompt, so make it useful:
> not just "score 3.2", but what went wrong or what could be better.
>
> You do not need to protect yourself against bad code here. BLADE runs
> every evaluation in a separate process with its own virtual environment
> and a time limit. If the generated code crashes or hangs, the error
> becomes feedback for the LLM, and your experiment keeps going.
>
> Packages that only `evaluate` needs go under `dependencies` in the
> experiment file, and are installed in that environment.

## Scene 5 — The reference solution and check (3:30–4:20)

```console
$ uv run iohblade check my-problem/experiment.yml
Checking my-problem/experiment.yml ...

[OK  ] problem MyProblem: evaluated smoke_test_solution reference_solution.py in 1.8s, fitness -8.38202 (1.8s)
[OK  ] method LLaMEA-local: ran on 'MyProblem' with budget 2 (2.8s)

2 ok - ready to run.
(LLMs were not contacted; add --llm to test them.)
```

**Narration:**

> The reference solution is a known-good answer, here random search. Use
> the same code as in your example prompt. `check` builds the evaluation
> environment, scores the reference solution, and runs the methods for two
> steps without an LLM. If this passes, your problem works.

## Scene 6 — When something is wrong (4:20–5:15)

Introduce a bug in `func` (`shift` → `shft`), then:

```console
$ uv run iohblade check my-problem/experiment.yml
Checking my-problem/experiment.yml ...

[FAIL] problem MyProblem: evaluating smoke_test_solution reference_solution.py failed: In the code, line 12, in __call__, the following error occurred: | NameError: name 'shft' is not defined | On line: f = func(x) (1.5s)
[SKIP] method LLaMEA-local: no problem with a working reference solution to run on (0.0s)

1 fail, 1 skip - fix the failures before running.
```

Fix the typo and run `check` again (OK).

**Narration:**

> Here is why `check` matters. A small typo in our objective function.
> Without the check, every algorithm the LLM writes would fail, and we
> might think the LLM is bad. `check` points at the problem before a single
> LLM call. Fix it, check again, and we are ready.
>
> And if a run does start with a broken problem, BLADE notices when the
> failures come from the harness rather than from the generated code, and
> stops the run instead of burning the whole budget.

## Scene 7 — Training and test instances (5:15–6:05)

**Screen:** the `test()` method in the template, then a BBOB problem's
training and test instances in the docs.

```python
    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)
```

**Narration:**

> One more method: `test`. During the search, methods only ever see the
> score from `evaluate`. `test` is for afterwards: you call it yourself on
> the best algorithms, on instances the search never saw. That tells you
> whether an algorithm generalises or just fits its training instances.
> The built-in BBOB suites have separate training and test instances for
> exactly this.

## Scene 8 — Outro (6:05–6:25)

**Narration:**

> That is a BLADE problem: prompts, an example, `evaluate`, and a reference
> solution for `check`. The documentation has a longer tutorial. Next
> episode: the other side, writing your own search method.

---

## Notes for the recording

- The template's `test()` simply calls `evaluate()`; scene 7 explains why a
  real problem would use other instances. BLADE does not call `test()`
  during a run; it is meant for the analysis afterwards.
- The harness-error stop (scene 6, last paragraph) happens after
  `Problem.max_harness_errors` failures in a row.
