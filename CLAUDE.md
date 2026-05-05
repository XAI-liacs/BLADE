# CLAUDE.md

Guidance for Claude Code and other AI assistants working with the BLADE repository.

## Project Overview

**BLADE** (Benchmarking LLM-driven Automated Design and Evolution) is a Python benchmark suite for evaluating automatic algorithm design methods, particularly those that use LLMs to generate and evolve optimization heuristics. The package is published as `iohblade` on PyPI.

Core idea: LLM-based methods are given a problem prompt and iteratively generate/improve Python code implementing optimization heuristics. BLADE provides a standardized harness to run these methods, evaluate the resulting heuristics, and compare against human-designed baselines.

## Setup

- Requires **Python 3.11+**.
- Uses **uv** (v0.7.19+) as the package manager.
- Install dependencies:
  ```bash
  uv sync
  ```
- For development tools (linting, testing) and documentation:
  ```bash
  uv sync --group dev --group docs
  ```
- Optional groups: `methods` (EoH, ReEvo), `trackio`, `kerneltuner`, `apple-silicon`, `extraproblems`.

## Development Commands

Always prefix commands with `uv run` to use the project's virtual environment.

```bash
# Run all tests
uv run pytest tests/

# Run only unit tests
uv run pytest tests/unit/

# Run with coverage
uv run pytest --cov=iohblade --cov-report=xml tests/

# Format code (required before committing)
uv run isort iohblade/
uv run black iohblade/

# Launch the results web app
uv run iohblade-webapp

# Search the codebase (prefer rg over grep -R or ls -R)
rg "search_term" iohblade/
```

## Repository Layout

```
iohblade/               # Main package
├── __init__.py         # Public API + enforces multiprocessing spawn method
├── experiment.py       # Experiment orchestration (abstract base)
├── problem.py          # Problem abstraction + subprocess evaluation isolation
├── solution.py         # Solution/Individual: code, fitness, feedback, metadata
├── fitness.py          # Multi-objective fitness with Pareto dominance
├── method.py           # Abstract Method base class
├── llm.py              # LLM provider abstraction (OpenAI, Claude, Gemini, Ollama, ...)
├── mcts_node.py        # MCTS tree node
├── plots.py            # Visualization (convergence, behaviour metrics)
├── behaviour_metrics.py# Behavioural profiling of optimization runs
├── webapp.py           # Streamlit web app for browsing results
├── assets.py           # Static assets and data
├── utils.py            # Shared utilities, exceptions, code extraction
├── benchmarks/         # 15+ benchmark domains (BBOB, SBOX, MA-BBOB, AutoML, ...)
├── baselines/          # Human-designed reference algorithms (e.g., ModCMA-ES)
├── loggers/            # Logging backends: file, MLflow, Trackio
├── methods/            # LLM search methods: LLaMEA, LHNS, MCTS-AHD, RandomSearch, FunSearch
└── misc/               # AST utilities, namespace preparation for code execution
tests/
├── unit/               # 19 unit test files (target: 80% coverage)
└── e2e/                # End-to-end integration tests
examples/               # Runnable example scripts and Jupyter notebooks
run_benchmarks/         # Pre-configured benchmark runner scripts
docs/                   # Sphinx documentation source
```

## Core Abstractions

Understanding these classes is essential for working with the codebase:

| Class | File | Role |
|-------|------|------|
| `Problem` | `problem.py` | Abstract base; defines the LLM prompt, evaluates submitted code in an isolated subprocess |
| `Solution` | `solution.py` | A candidate algorithm: holds code string, `Fitness` value, feedback, genealogy |
| `Fitness` | `fitness.py` | Scalar or multi-objective fitness; supports Pareto dominance comparisons |
| `Method` | `method.py` | Abstract base for search algorithms; called iteratively to generate `Solution` objects |
| `LLM` | `llm.py` | Abstract LLM provider; concrete subclasses for each vendor |
| `Experiment` | `experiment.py` | Orchestrates runs of (method, problem) pairs with seeding and parallel execution |
| `ExperimentLogger` | `loggers/` | Records results; file-based, MLflow, or Trackio backends |

### Typical Experiment Workflow

```python
from iohblade.llm import Ollama_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.benchmarks import BBOB_SBOX
from iohblade.loggers import ExperimentLogger
from iohblade.experiment import Experiment

llm = Ollama_LLM("llama3")
methods = [LLaMEA(llm, budget=50), RandomSearch(llm, budget=50)]
problems = [BBOB_SBOX(training_instances=[...], test_instances=[...])]
logger = ExperimentLogger("results/my_experiment")

experiment = Experiment(methods=methods, problems=problems, runs=5, exp_logger=logger)
experiment()
```

## Code Style Conventions

- **Indentation**: 4 spaces (no tabs).
- **Line length**: ~80 characters (soft limit).
- **Naming**: `CamelCase` for classes, `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for module-level constants.
- **Type hints**: Use PEP 484 style (`dict[str, float]`, `list[str]`), not `Dict`/`List` from `typing`.
- **Docstrings**: Google-style with `Args:`, `Returns:`, `Raises:` sections.
- **Abstract interfaces**: Use `ABC` + `@abstractmethod` for extensible base classes.
- **Optional imports**: Wrap optional/heavy dependencies in `try/except ImportError` to keep startup fast.
- **Import order** (enforced by isort): stdlib → third-party → local (relative).

## Testing Requirements

- Run `uv run pytest tests/` for any code change. Doc-only changes do not require tests.
- Target test coverage: **80%** (`--cov=iohblade`).
- Use `pytest-mock` for mocking LLM calls — never make real API calls in tests.
- Fixtures should use `tmp_path` for temporary directories and clean up after themselves.
- Roundtrip tests (dict serialization and pickle) are expected for data classes like `Solution` and `Fitness`.

## Key Patterns & Gotchas

### Multiprocessing Start Method
`iohblade/__init__.py` forces the `spawn` multiprocessing start method at import time. If you write scripts, set this at the top before any other imports:
```python
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```

### Solution Evaluation is Subprocess-Isolated
`Problem.evaluate()` runs submitted code inside an isolated virtualenv subprocess. This prevents malicious or buggy generated code from crashing the main process. Do not bypass this when adding new benchmarks.

### Custom Exceptions
Use these from `iohblade.utils` instead of generic exceptions:
- `NoCodeException` — LLM response contained no valid Python code.
- `OverBudgetException` — Method exceeded its evaluation budget.
- `TimeoutException` — Evaluation timed out.
- `ThresholdReachedException` — Target fitness threshold was met.

### Multi-Objective Fitness
`Fitness` supports scalar and multi-objective modes. Use Pareto dominance operators (`<`, `>`, `<=`, `>=`) rather than direct float comparison when fitness may be multi-dimensional.

### LLM Provider Abstraction
All LLM providers share the same interface. Add new providers by subclassing `LLM` in `llm.py` and implementing the `query()` method.

### Benchmarks are Self-Contained
Each benchmark in `iohblade/benchmarks/` defines its own `Problem` subclass, system prompt, and evaluation logic. Adding a new benchmark means creating a new subclass — do not modify existing benchmark logic.

## CI/CD

GitHub Actions workflows (`.github/workflows/`):
- `test-ubuntu.yml` / `test-windows.yml` — Unit tests on push/PR.
- `test-e2e.yml` — End-to-end integration tests.
- `publish.yml` — Publishes to PyPI on release via `uv build` + `uv publish`.
- `docs2pages.yml` — Builds Sphinx docs and deploys to GitHub Pages on release.

## Documentation

Sphinx docs live in `docs/`. To build locally:
```bash
uv run sphinx-apidoc -o docs/ iohblade/   # regenerate API docs
cd docs && uv run sphinx-build -b html . _build
```

Docs are auto-deployed to GitHub Pages when a release is published.

## Contribution Workflow

1. File an issue before starting significant work.
2. Develop on a feature branch.
3. Format with `isort` + `black` before committing.
4. Ensure `pytest tests/` passes with ≥80% coverage.
5. Update `docs/` for any public API changes.
6. Open a pull request referencing the issue.
