# ModularCMAES Usage Guidelines for Agents

Use these instructions whenever a task requires configuring, running, tuning, or extending ModularCMAES.

Source: [IOHprofiler/ModularCMAES usage documentation](https://github.com/IOHprofiler/ModularCMAES#usage-)

Module reference: [IOHprofiler/ModularCMAES module documentation](https://github.com/IOHprofiler/ModularCMAES#modules-)

These examples are verified against the C++ interface in `modcma==1.2.0`. This
project exclusively uses `modcma.c_maes`; do not import optimiser classes,
parameter classes, or `fmin` directly from top-level `modcma`.

## 1. Required optimisation inputs

- `func`: the objective function, accepting a candidate vector and returning a scalar value
- `x0`: the initial search point; required by `c_maes.fmin`, optional in low-level `Settings`
- `sigma0`: the initial step size; required by `c_maes.fmin`, optional in low-level `Settings` (defaults to `2.0`).
  If no informed value is available and finite bounds exist, use `0.3 * float(np.mean(ub - lb))` (a scalar).
- `budget`: the maximum number of objective-function evaluations

Treat the objective as a minimisation objective unless the surrounding task explicitly defines a transformation for maximisation.

## 2. High-level interface

Use `c_maes.fmin` for ordinary single-objective optimisation when no custom control loop is needed:

```python
from modcma import c_maes

xopt, fopt, evals, cma = c_maes.fmin(
    func,
    x0,
    sigma0,
    budget,
    active=True,
    target=10.0,
    matrix_adaptation="NONE",
)
```

Pass `func`, `x0`, `sigma0`, and `budget`; they are required. Module/setting values
are supplied as keyword arguments using their `Modules`/`Settings` names (string
names are accepted for module options here).

## 3. Low-level interface

Use the low-level API for explicit module construction, access to optimiser state, or control over individual CMA-ES phases.

```python
import numpy as np
from modcma import c_maes

modules = c_maes.parameters.Modules()
modules.active = True
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.MATRIX

settings = c_maes.parameters.Settings(
    dim=10,
    modules=modules,
    sigma0=2.5,
    budget=10000,
    lb=lower,
    ub=upper,
)
parameters = c_maes.Parameters(settings)

cma = c_maes.ModularCMAES(parameters)
cma.run(func)

xopt, fopt, evals = cma.p.stats.global_best.x, cma.p.stats.global_best.y, cma.p.stats.evaluations
```

Instead of `cma.run(func)`, you may step manually:

```python
while not cma.break_conditions():
    cma.step(func)
```

Population size (`lambda0`) and parent count (`mu0`) belong to `Settings`, not `Modules`.
The supported `Settings(...)` constructor arguments in this version are:

```text
dim, modules, target, max_generations, budget, sigma0, lambda0, mu0,
x0, lb, ub, integer_variables, cs, cc, cmu, c1, damps, acov, verbose,
always_compute_eigv
```

## 4. Module-selection guidelines

ModularCMAES groups its algorithmic choices into module categories on
`c_maes.parameters.Modules()`. Options can generally be combined, but not every
combination is meaningful. Boolean modules default to disabled.

### 4.1 Matrix adaptation

Selects how the search distribution learns dependencies between variables.

```python
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE
```

Choices: `COVARIANCE` (standard full covariance adaptation, the normal baseline),
`MATRIX` (MA-ES, similar behaviour with lower overhead), `SEPARABLE` (diagonal
adaptation, useful when full pairwise dependencies are unnecessary), `NONE`
(step-size adaptation only), or `CHOLESKY`/`CMSA`/`COVARIANCE_NO_EIGV`/`NATURAL_GRADIENT`
for deliberate experimentation.

### 4.2 Active update

```python
modules.active = True
```

Incorporates poorly-performing individuals with negative weights when adapting the covariance matrix. Can improve adaptation but changes covariance updates materially.

### 4.3 Elitism

```python
modules.elitist = True
```

Switches from comma to plus selection, letting strong solutions survive across generations. Can accelerate convergence on unimodal problems but may reduce diversity and cause premature convergence on multimodal ones.

### 4.4 Orthogonal sampling

```python
modules.orthogonal = True
```

Applies Gram-Schmidt so sampled mutation directions are orthogonal, for more evenly distributed sampling.

### 4.5 Sequential selection

```python
modules.sequential_selection = True
```

Stops evaluating the remainder of a generation once enough improvements are found, reducing objective-function evaluations.

### 4.6 Threshold convergence

```python
modules.threshold_convergence = True
```

Requires mutation vectors to satisfy a length threshold that decreases over time, prolonging exploration before local search.

### 4.7 Per-candidate sigma sampling

```python
modules.sample_sigma = True
```

Draws an individual step size per candidate from a log-normal distribution based on the global `sigma`, adding self-adaptation.

### 4.8 Base sampler and sample transformation

These are two **independent** settings. The base sampler generates points in the
uniform unit hypercube; the sample transformer then maps those points to the
target search distribution shape.

```python
modules.sampler = c_maes.options.BaseSampler.SOBOL              # HALTON, SOBOL, or UNIFORM
modules.sample_transformation = c_maes.options.SampleTranformerType.GAUSSIAN  # the target distribution shape
```

Never treat `SOBOL`/`HALTON` alone as the final search distribution — always set both fields. There is no `GAUSSIAN` base sampler; `GAUSSIAN` only exists as a sample transformation.

### 4.9 Recombination weights

```python
modules.weights = c_maes.options.RecombinationWeights.DEFAULT
```

Determines how selected individuals contribute to strategy updates. Use `DEFAULT` as the baseline unless alternative selection pressure is being tested.

### 4.10 Mirrored and pairwise sampling

```python
modules.mirrored = c_maes.options.Mirror.PAIRWISE
```

Mirrored sampling generates opposite mutation vectors to improve coverage. `PAIRWISE` additionally keeps only the better candidate from each mirrored pair, preventing paired vectors from cancelling each other.

### 4.11 Step-size adaptation

```python
modules.ssa = c_maes.options.StepSizeAdaptation.CSA
```

Controls how the global search scale changes. Treat a change here as a major algorithmic change, not a minor numerical setting.

### 4.12 Restart strategy and criteria

```python
modules.restart_strategy = c_maes.options.RestartStrategy.IPOP
```

`RESTART` restarts without a population-size schedule; `IPOP` increases population size after each restart; `BIPOP` alternates between larger and smaller population regimes.

Restart triggers can be controlled through `cma.p.criteria.items` (a `CriteriaVector`, not a plain Python list). Clearing it disables criterion-triggered restarts:

```python
cma.p.criteria.items.clear()
```

Do not remove or alter restart criteria silently — if a criterion uses a class-level tolerance, changing it can affect every instance in the process.

### 4.13 Bound correction

```python
modules.bound_correction = c_maes.options.CorrectionMethod.SATURATE
```

Determines how infeasible candidates are handled. Choices: `NONE`, `SATURATE` (clipping), `MIRROR`, `TOROIDAL` (wrapping), `COTN` (truncated-normal-based correction), `UNIFORM_RESAMPLE`, `RESAMPLE`. Choose a method compatible with the physical meaning of the variables.

### 4.14 Centre placement on restart

```python
modules.center_placement = c_maes.options.CenterPlacement.UNIFORM
```

Controls the sampling distribution's centre after a restart: `X0` (original initial point), `ZERO` (all-zero vector), `UNIFORM` (random point inside the search space), `CENTER` (geometric centre of the bounded search space). `ZERO`/`CENTER` require finite bounds.
