# BLADE experiment reference

Full guide: https://xai-liacs.github.io/BLADE/yaml_experiments.html.
`iohblade schema` prints the JSON schema; `iohblade list` the built-in keys.

## Contents
- [Fields](#fields)
- [Built-in components](#built-in-components)
- [Custom problem](#custom-problem)
- [Custom method](#custom-method)
- [Custom LLM](#custom-llm)
- [LLaMEA operators](#llamea-operators)
- [LLaMEA variants](#llamea-variants)
- [LLaMEA HPO](#llamea-hpo)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)

## Fields

```yaml
version: 1                       # optional
experiment:
  output_dir: results/my-exp     # required; under results/ for the web app; resumed if it exists
  name: my-experiment
  runs: 5                        # independent runs per method x problem
  seeds: [1, 2, 3]               # explicit unique seeds (instead of runs)
  budget: 100                    # solutions per run; default budget of every method
  n_jobs: 1                      # runs in parallel
  show_stdout: false
  log_stdout: false
  log_level: INFO                # DEBUG | INFO | WARNING | ERROR
  logger: {type: file, params: {}}   # file | mlflow | trackio
  env_cache: true                # share evaluation virtualenvs between problems
  isolated_pythonpath: false     # evaluation sees only iohblade + dependencies
llms:
  <name>:
    provider: ollama             # see `iohblade list llms`; custom for your own
    model: "qwen2.5-coder:14b"
    api_key: "${env:OPENAI_API_KEY}"
    llms: [a, b]                 # only for provider: multi
    params: {}                   # extra constructor arguments
methods:
  - type: llamea                 # see `iohblade list methods`; custom for your own
    name: "LLaMEA-{llm}"
    llm: qwen                    # or a list of LLM names
    budget: 50                   # overrides experiment.budget
    params: {}                   # constructor arguments
problems:
  - type: circle_packing         # one problem; see `iohblade list problems`
    name: CirclePacking
    params: {}
    smoke_test_solution: reference_solution.py   # used by `iohblade check`
    dependencies: []             # pip requirements for the evaluation env
  - suite: unit_square_packing   # or a suite: several problems, no `name`
    params: {use_best: false}    # AlphaEvolve suites: start from the best-known solution
  - suite: continuous_bbo        # default suites: bbob_groups (5), bbob (f1-f24), ma_bbob,
    params: {dim: 10, budget_factor: 2000}  # continuous_bbo, alphaevolve_math, combinatorial, default, quick, full
```

- `params` are checked against the component's constructor; unknown or
  missing parameters are reported by `validate`. Strings are converted to
  enum values where a parameter expects an enum.
- `${env:NAME}` and `${env:NAME:-default}` read the environment or a
  `.env` file next to the YAML file. Quote them inside `[...]`/`{...}`.
  An unquoted lone placeholder gets a type (`n_parents: ${env:NP:-4}` is
  the integer 4); a quoted one (`"${env:KEY}"`) and credential keys
  (`api_key`, `*_token`, `*_secret`, `*_password`) stay strings.
- `--set dotted.path=value` overrides any value on the command line,
  e.g. `--set methods.0.budget=20` (`${env:...}` works there too).

## Built-in components

Run `iohblade list` for the current list. To see the parameters of a
component, print its constructor signature and docstring:

```bash
python -c "import inspect; from iohblade.registry import registry as r; c = r.resolve('problem', 'circle_packing'); print(inspect.signature(c)); help(c)"
```

(`'method'`, `'suite'`, `'llm'` or `'logger'` for other kinds.) LLaMEA
forwards extra `params` to `llamea.LLaMEA`, so its options (`n_parents`,
`n_offspring`, `elitism`, `operators`, ...) are documented there. Some
components need extra packages; `iohblade list` shows how to install them.

## Custom problem

```yaml
problems:
  - type: custom
    source: my_problem.py          # or a package folder; "submodule:Class" for class
    class: MyProblem
    params: {dim: 5}
    dependencies: [scipy>=1.11]    # also read from requirements.txt / pyproject.toml next to it
    smoke_test_solution: reference_solution.py
```

Start from `iohblade init DIR -t custom-problem`. The class:

```python
from iohblade.problem import Problem
from iohblade.solution import Solution

class MyProblem(Problem):
    def __init__(self, dim=5, name="MyProblem"):
        super().__init__(name=name, eval_timeout=60)   # seconds per evaluation
        self.dim = dim
        self.minimisation = False                       # BLADE maximises by default
        self.task_prompt = "Write a Python class ..."   # what the LLM must write
        self.example_prompt = "An example:\n```python\n...\n```\n"  # working code!
        self.format_prompt = "Give the full class in one ```python code block.\n"

    def evaluate(self, solution: Solution) -> Solution:
        # exec solution.code, run it, then:
        solution.set_scores(fitness, "feedback text shown to the LLM")
        return solution

    def test(self, solution): return self.evaluate(solution)
    def to_dict(self): return {"dim": self.dim}
    def get_config(self):
        return {"tags": [], "name": self.name, "prompt": self.get_prompt(),
                "minimisation": self.minimisation, "config": self.to_dict()}
```

Rules:
- `evaluate()` runs in a separate virtualenv with `dependencies`; the
  problem object is created in BLADE's own process. Import
  evaluation-only packages inside `evaluate()`.
- Execute the candidate's code only inside `evaluate()`. An exception
  before `solution.code` is used counts as a harness error.
- The code block in `example_prompt` (or `smoke_test_solution`) must be a
  working solution: `iohblade check` evaluates it.
- The source folder is on the evaluation `PYTHONPATH`, so data files and
  helper modules next to it work. Don't name the file like an installed
  module: `validate` reports that as an error (rename the file or folder).

## Custom method

```yaml
methods:
  - type: custom
    source: my_method.py
    class: MyMethod
    name: "MyMethod-{llm}"
    llm: qwen
    params: {}
```

```python
from iohblade.method import Method

class MyMethod(Method):
    def __init__(self, llm, budget, name="MyMethod"):
        super().__init__(llm, budget, name)

    def __call__(self, problem):
        prompt = [{"role": "user", "content": problem.get_prompt()}]
        best = problem(self.llm.sample_solution(prompt))   # problem(...) evaluates
        for _ in range(self.budget - 1):
            ...  # build a prompt from best.code / best.feedback, sample, evaluate
        return best

    def to_dict(self): return {"name": self.name, "budget": self.budget}
    def get_config(self): return {"name": self.name, "config": {}}
```

Compare fitness respecting `problem.minimisation`. Methods run in BLADE's
own environment: install what they import there. Start from
`iohblade init DIR -t custom-method`.

## Custom LLM

`provider: custom` with `source` and `class`: a subclass of
`iohblade.llm.LLM` that implements `_query(self, session, **kwargs)`
(`session` is a list of `{"role": ..., "content": ...}` messages; return
the reply text) and `get_config()` (a list with one
`{"model": ..., "config": {...}}` dict per model).

## LLaMEA operators

```yaml
- type: llamea
  llm: qwen
  params:
    n_parents: 2
    n_offspring: 2
    elitism: true
    operators:
      - {name: refine, prompt: "Refine the strategy of the selected solution.", weight: 0.7, number_of_parents: 1}
      - {name: crossover, prompt: "Combine the two selected solutions into a better one.", weight: 0.3, number_of_parents: 2}
```

`prompt` is required; `number_of_parents` must not exceed `n_parents`.

## LLaMEA variants

`params: {variant: hpo | bo | sage}` uses a ready-made LLaMEA configuration
(llamea>=1.6.0): `hpo` = `HPO: true`, (5+5); `bo` = (4+16), crossover 0.6;
`sage` = `feature_guided_mutation: true` (needs the `feature-guidance`
group). Other `params` override the preset; BLADE still sets the prompts
(from the problem), the budget and the direction. Without `name` the method
is `LLaMEA-<VARIANT>`. `variant: mo` and `multi_objective: true` are
rejected: LLaMEA's multi-objective mode does not accept BLADE's `Fitness`.

```yaml
- {type: llamea, llm: qwen, params: {variant: bo}}
```

## LLaMEA HPO

`params: {HPO: true}` makes LLaMEA also generate a configuration space;
problems that support tuning (`automl`, `bbob_sbox`, `ma_bbob`,
`kerneltuner`) then tune it with SMAC (cost `1 - score`, e.g. `1 - AOCC`)
and score the incumbent, and their evaluation environment adds
`smac>=2.4.1` and `ConfigSpace`. `validate` notes which problems tune and
warns for the others (evaluated with default hyperparameters). Tuning
budget: problem params `hpo_trials` (`bbob_sbox` 100, `ma_bbob` and
`kerneltuner` 50) and `hpo_instances` (12); tuning makes evaluations
slower, so raise `eval_timeout`.

## Python API

`from iohblade.experiment import Experiment; Experiment.from_yaml("experiment.yml")()`
runs a YAML file from Python. Scripts that start BLADE themselves must set
`multiprocessing.set_start_method("spawn", force=True)` first.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Unknown field 'x'. Did you mean 'y'?` | Typo; follow the suggestion |
| `Unknown parameter 'x' for problem 'p'` | Not a constructor argument; read the class signature |
| `Unknown problem type` | Use a key from `iohblade list problems` |
| Missing package for a method/LLM | Install it in BLADE's environment (`validate` names it) |
| Environment build fails (exit 3) | See `<output_dir>/env-errors/`; fix `dependencies`; add `swig`/`cmake` for source builds; `iohblade cache clear` for a corrupt cache |
| `[FAIL] problem ...` in `check` | Harness or reference solution broken; fix the problem code |
| `[FAIL] method ...` in `check` | Method crashes with a valid solution; fix the method |
| `check --llm` fails | Wrong key, model name or server not running (`ollama serve`) |
| Local model on vLLM/SGLang | `provider: openai`, `params: {base_url: "http://localhost:8000/v1"}` (no `api_key` needed); several GPUs: one server each, combined with `provider: multi` |
| Run stops with `HarnessError` | 3 harness errors in a row; see the run's `error.txt` and `run.log` |
| `HPOUnavailableError` (harness error) | smac cannot be imported in the evaluation env; let HPO be enabled on the problem (LLaMEA `HPO: true`, `problem.enable_hpo()`) so its env installs `smac>=2.4.1`; `iohblade cache clear` if that env is stale |
| Mostly `c`/`t` errors | Generated code fails or is slow: clarify prompts, give a working example, raise `eval_timeout` |
| Resumed run warns the configuration changed | Previous record kept as `blade_config-<timestamp>/`; use a new `output_dir` for a different experiment |
| `run` (exit 2): output directory already holds runs of a different setup | Fewer runs/seeds, a renamed/removed method or problem, or another budget for runs that already started. Use a new `output_dir` (`--set experiment.output_dir=results/trial`) or restore the settings. Adding runs, seeds, methods or problems is fine: finished runs are kept, new ones start |
