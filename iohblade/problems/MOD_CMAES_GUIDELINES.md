# ModularCMAES Usage Guidelines for Agents

Use these instructions whenever a task requires configuring, running, tuning, or extending ModularCMAES.

Source: [IOHprofiler/ModularCMAES usage documentation](https://github.com/IOHprofiler/ModularCMAES#usage-)

Module reference: [IOHprofiler/ModularCMAES module documentation](https://github.com/IOHprofiler/ModularCMAES#modules-)

These examples are verified against the C++ interface in `modcma==1.2.0`.
Re-check the installed API when upgrading the package. This project exclusively
uses `modcma.c_maes`; do not import optimiser classes, parameter classes, or
`fmin` directly from top-level `modcma`.

## 1. Required optimisation inputs

Before running an optimisation, identify and validate:

- `func`: the objective function, accepting a candidate vector and returning a scalar value
- `x0`: the initial search point; required by `c_maes.fmin`, optional in low-level `Settings`
- `sigma0`: the initial step size; required by `c_maes.fmin`, optional in low-level `Settings`, where `modcma==1.2.0` defaults to `2.0`.
- `budget`: the maximum number of objective-function evaluations
- bounds, target value, variable types, and dimensionality, where applicable

If no informed value for `sigma0` is available and finite bounds exist, consider
approximately `0.3` times a representative bound width. The C++ interface in
`modcma==1.2.0` requires a scalar, so for vector bounds a typical choice is
`0.3 * float(np.mean(ub - lb))`.

Treat the objective as a minimisation objective unless the surrounding task explicitly defines a transformation for maximisation.

## 2. High-level interface

Use `c_maes.fmin` for ordinary single-objective optimisation when no custom control loop is needed:

```python
from modcma import c_maes

x0 = [0.0, 1.0, 2.0, 3.0]
sigma0 = 0.234
budget = 100

xopt, fopt, evals, cma = c_maes.fmin(
    func,
    x0,
    sigma0,
    budget,
    active=True,
    target=10.0,
    cc=0.8,
    matrix_adaptation="NONE",
)
```

Rules:

- Pass `func`, `x0`, `sigma0`, and `budget`; they are required.
- Supply module and setting values as keyword arguments using their names in the API's `Modules` and `Settings` objects.
- String names may be used for module options in the high-level interface.
- Record `xopt`, `fopt`, and the number of evaluations in results.

## 3. Low-level interface

Use the low-level API when the task requires explicit module construction, access to optimiser state, custom stopping checks, or control over individual CMA-ES phases.

```python
import numpy as np
from modcma import c_maes

# Define objective function
def func(x: np.ndarray) -> float:
    return float(np.sum(x**2))

modules = c_maes.parameters.Modules()
modules.active = True
modules.matrix_adaptation = (
    c_maes.options.MatrixAdaptationType.MATRIX
)

settings = c_maes.parameters.Settings(
    dim=10,
    modules=modules,
    sigma0=2.5,
    budget=10000
)
parameters = c_maes.Parameters(settings)

cma = c_maes.ModularCMAES(parameters)
cma.run(func)

xopt, fopt, evals, cma = cma.p.stats.global_best.x, cma.p.stats.global_best.y, cma.p.stats.evaluations, cma
```

Available execution levels include:

```python
# One complete optimiser step at a time
while not cma.break_conditions():
    cma.step(func)
```

```python
# Individual internal phases
while not cma.break_conditions():
    cma.mutate(func)
    cma.select()
    cma.recombine()
    cma.adapt()
```

Preserve the phase order shown above unless the research task explicitly investigates a modified algorithm.

For `modcma==1.2.0`, the writable C++ `Modules` fields are exactly:

```text
active, bound_correction, center_placement, elitist, matrix_adaptation,
mirrored, orthogonal, repelling_restart, restart_strategy, sample_sigma,
sample_transformation, sampler, sequential_selection, ssa,
threshold_convergence, weights
```

Population sizes belong to `Settings` as `lambda0` and `mu0`.

The supported `Settings(...)` constructor arguments in this version are:

```text
dim, modules, target, max_generations, budget, sigma0, lambda0, mu0,
x0, lb, ub, integer_variables, cs, cc, cmu, c1, damps, acov, verbose,
always_compute_eigv
```

Here `sigma0` is a scalar.

## 4. Automated configuration and tuning

Use the generated configuration space when another optimiser or AutoML system must select ModularCMAES modules or numerical settings.

```python
from modcma.c_maes import get_configspace

cs = get_configspace(dim=10)
```

The generated `ConfigSpace.ConfigurationSpace` can include:

- categorical choices for algorithmic modules  (e.g., mirrored sampling, restart strategy, bound correction)
- population settings such as `lambda0` and `mu0`
- `sigma0`
- learning-rate and damping settings such as `cs`, `cc`, `cmu`, `c1`, and `damps`
- the constraint `mu0 <= lambda0`

To tune only module choices:

```python
cs_modules = get_configspace(
    dim=10,
    add_popsize=False,
    add_sigma=False,
    add_learning_rates=False,
)
```

Convert a sampled, default, or externally supplied configuration into a settings object before constructing the optimiser:

```python
import numpy as np
from modcma import c_maes
from modcma.c_maes import get_configspace, settings_from_config

dim = 10
cs = get_configspace(dim=dim)

# Sample or load a configuration
config = cs.sample_configuration()

# Or for defaults
default = cs.get_default_configuration()

# Configuration values may be edited before conversion.
config["sampler"] = "HALTON"

# Extra Settings keyword arguments can be supplied during conversion.
settings = settings_from_config(
    dim=dim,
    config=config,
    lb=-5.0 * np.ones(dim),
    ub=5.0 * np.ones(dim),
)

parameters = c_maes.Parameters(settings)
cma = c_maes.ModularCMAES(parameters)
cma.run(func)
```

When evaluating configurations, use comparable budgets, objective definitions, initialisation rules, and random-seed policies. Report both the selected module combination and numerical hyperparameters.

## 5. Integer variables

Declare integer variables by their zero-based coordinate indices:

```python
settings = c_maes.parameters.Settings(
    dim=5,
    integer_variables=[1, 2],
)
```

The current integer-variable mechanism is rudimentary. It applies a lower bound to `sigma` for integer coordinates and rounds those coordinates before objective evaluation. Do not describe it as a general mixed-integer optimisation guarantee.

For an all-integer problem, use all coordinate indices:

```python
integer_variables = list(range(dim))
```

## 6. Module-selection guidelines

ModularCMAES groups its algorithmic choices into module categories. Options can
generally be combined, but not every combination is necessarily meaningful or
compatible. Boolean modules are disabled by default. Verify support in the
installed C++ interface and avoid enabling modules without explaining the
intended effect.

### 6.1 Matrix adaptation

This module selects how the search distribution learns dependencies between variables.

```python
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE
```

Relevant C++ choices include:

- `COVARIANCE`: standard full covariance-matrix adaptation
- `MATRIX`: MA-ES, which has similar empirical behaviour on many problems and lower computational overhead
- `SEPARABLE`: diagonal adaptation, useful when full pairwise dependencies are unnecessary or too expensive
- `NONE`: step-size adaptation without matrix adaptation
- `CHOLESKY`, `CMSA`, `COVARIANCE_NO_EIGV`, and `NATURAL_GRADIENT`: alternative adaptation mechanisms for deliberate experimentation

Use `COVARIANCE` as the normal baseline. Select alternatives only when dimensionality, runtime, separability assumptions, or the experiment design justify them.

### 6.2 Active update

Active update incorporates poorly performing individuals with negative weights when adapting the covariance matrix.

```python
modules.active = True
```

It may improve adaptation, but it changes covariance updates materially. Treat `active=False` and `active=True` as distinct algorithm configurations and report the choice.

### 6.3 Elitism

Elitism switches from comma selection to plus selection, allowing strong solutions from the previous generation to survive.

```python
modules.elitist = True
```

It can accelerate convergence on unimodal problems, but may reduce population diversity and encourage premature convergence. Do not enable it automatically for multimodal exploration.

### 6.4 Orthogonal sampling

Orthogonal sampling applies a Gram-Schmidt procedure so newly sampled mutation directions are orthogonal.

```python
modules.orthogonal = True
```

Use it when more evenly distributed sampling directions are relevant. Since it modifies the sampling design, include it explicitly in experiment metadata.

### 6.5 Sequential selection

Sequential selection compares evaluated candidates against the current best and can stop evaluating the remainder of a generation once sufficiently many improvements have been found.

```python
modules.sequential_selection = True
```

This can reduce objective-function evaluations, especially when evaluations are expensive. Evaluation counts may then differ between generations and configurations, so comparisons must use actual evaluation counts rather than iteration counts.

### 6.6 Threshold convergence

Threshold convergence requires mutation vectors to satisfy a length threshold that decreases over time. It is intended to prolong exploration before transitioning towards local search.

```python
modules.threshold_convergence = True
```

Use it only when the exploration-to-exploitation schedule is part of the intended configuration.

### 6.7 Per-candidate sigma sampling

The sample-sigma module draws an individual step size for each candidate from a log-normal distribution based on the global `sigma`.

```python
modules.sample_sigma = True
```

This introduces additional self-adaptation. Report it separately from the global step-size adaptation method.

### 6.8 Base sampler and sample transformation

In the C++ backend, the base sampler generates points in the uniform unit hypercube. The sample transformer then maps those points to the target search distribution. These are separate decisions.

```python
modules.sampler = c_maes.options.BaseSampler.SOBOL
modules.sample_transformation = c_maes.options.SampleTranformerType.GAUSSIAN
```

Base-sampler choices include ordinary uniform sampling, Sobol sequences, and Halton sequences. Sobol and Halton are low-discrepancy alternatives intended to cover the unit hypercube more evenly.

Sample transformations include:

- `NONE`, no transformation
- `GAUSSIAN`, the default target distribution
- `SCALED_UNIFORM`
- `LAPLACE`
- `LOGISTIC`
- `CAUCHY`
- `DOUBLE_WEIBULL`

Do not describe `SOBOL` or `HALTON` alone as the final search distribution in the C++ backend. Always consider both `sampler` and `sample_transformation`. Preserve the API spelling `SampleTranformerType`, including its spelling, when that is the identifier exposed by the installed package.

### 6.9 Recombination weights

Recombination weights determine how selected individuals contribute to strategy updates.

```python
modules.weights = c_maes.options.RecombinationWeights.DEFAULT
```

Choices include `DEFAULT`, `EQUAL`, and `EXPONENTIAL`. Use `DEFAULT` as the baseline unless alternative selection pressure is being tested.
e
### 6.10 Mirrored and pairwise sampling

Mirrored sampling generates opposite mutation vectors to improve coverage. Pairwise mode retains only the better candidate from each mirrored pair for recombination, preventing paired vectors from cancelling each other.

```python
modules.mirrored = c_maes.options.Mirror.MIRRORED
# or
modules.mirrored = c_maes.options.Mirror.PAIRWISE
```

Use `PAIRWISE` only when mirrored sampling is intended. Report whether mirroring is disabled, ordinary, or pairwise.

### 6.11 Step-size adaptation

The step-size adaptation module controls how the global search scale changes.

```python
modules.ssa = c_maes.options.StepSizeAdaptation.CSA
```

The choices exposed by `modcma==1.2.0` are `CSA`, `TPA`, `MSR`, `XNES`,
`MXNES`, `LPXNES`, `PSR`, `SR`, and `SA`.
Treat a change in step-size adaptation as a major algorithmic change, not a
minor numerical setting.

### 6.12 Restart strategy and criteria

Restart strategies respond to stagnation or other restart conditions:

- `NONE`: no configured restart strategy
- `RESTART`: restart without a population-size schedule
- `IPOP`: increase population size after each restart
- `BIPOP`: alternate between larger and smaller population regimes
- `STOP`: C++-only option that stops when a restart criterion is met

```python
modules.restart_strategy = c_maes.options.RestartStrategy.IPOP
```

The C++ backend allows restart triggers to be controlled through `cma.p.criteria.items`. This property is a `CriteriaVector`, not a normal Python list. Clearing it disables criterion-triggered restarts:

```python
cma.p.criteria.items.clear()
```

Must not remove or alter restart criteria silently. If a criterion uses a class-level or static tolerance, changing it can affect every instance in the process.

Custom criteria must retain a Python reference rather than being constructed inline:

```python
from modcma import c_maes


class MyCriterion(c_maes.restart.Criterion):
    def __init__(self):
        super().__init__("MyCriterionName")

    def on_reset(self, par: c_maes.Parameters):
        """Called when a restart happens (also at the start)"""
        pass

    def update(self, par: c_maes.Parameters):
        """Called after each iteration, needs to modify self.met"""
        self.met = True


criterion = MyCriterion()
criteria = c_maes.restart.CriteriaVector([criterion])
cma.p.criteria.items = criteria
```

Keep Python references to both `criterion` and `criteria` for at least as long as
the optimiser uses them. Ensure `update()` sets `self.met` according to the
actual condition. The unconditional `True` above is only a structural example.

### 6.13 Bound correction

Bound correction determines how infeasible candidates are handled.

```python
modules.bound_correction = c_maes.options.CorrectionMethod.SATURATE
```

Documented approaches include no correction, saturation or clipping, mirroring, toroidal wrapping, correction based on a truncated normal distribution (`COTN`), uniform resampling, and resampling. Choose a method that is compatible with the physical meaning of the variables.

### 6.14 Centre placement on restart

The C++-only centre-placement module controls the sampling distribution's centre after a restart:

- `X0`: return to the original initial point
- `ZERO`: use the all-zero vector
- `UNIFORM`: sample a point uniformly inside the search space
- `CENTER`: use the geometric centre of the bounded search space

Centre placement is initiated by a restart strategy. Validate that `ZERO` and `CENTER` are meaningful and feasible for the problem, and require finite bounds for choices that depend on the search-space extent.
