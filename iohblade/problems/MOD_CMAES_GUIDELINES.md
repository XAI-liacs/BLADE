# ModularCMAES Usage Guidelines for Agents

Use these instructions whenever a task requires configuring, running, tuning, or extending ModularCMAES.

Source: [IOHprofiler/ModularCMAES usage documentation](https://github.com/IOHprofiler/ModularCMAES#usage-)

Module reference: [IOHprofiler/ModularCMAES module documentation](https://github.com/IOHprofiler/ModularCMAES#modules-)

## 1. Default implementation choice

- Use the C++ backend exposed through Python (`modcma.c_maes`) by default.
- Prefer it for performance, complete module support, and fine-grained control.
- Use the pure-Python implementation only when the task explicitly requires the legacy API, educational inspection, or the ask-tell interface.
- State clearly when a requested feature is available only in one implementation.

## 2. Installation

Install the package with:

```bash
pip install modcma
```

Do not install from source unless development of ModularCMAES itself is required.

## 3. Required optimisation inputs

Before running an optimisation, identify and validate:

- `func`: the objective function, accepting a candidate vector and returning a scalar value
- `x0`: the initial search point
- `sigma0`: the initial step size
- `budget`: the maximum number of objective-function evaluations
- bounds, target value, variable types, and dimensionality, where applicable

If no informed value for `sigma0` is available and finite bounds exist, consider approximately `0.3 * (ub - lb)`. Confirm that the resulting value has the shape expected by the selected interface.

Treat the objective as a minimisation objective unless the surrounding task explicitly defines a transformation for maximisation.

## 4. Preferred high-level interface

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

## 5. Low-level interface

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
    c_maes.parameters.MatrixAdaptationType.MATRIX
)

settings = c_maes.parameters.Settings(
    dim=10,
    modules=modules,
    sigma0=2.5,
)
parameters = c_maes.Parameters(settings)

cma = c_maes.ModularCMAES(parameters)
cma.run(func)
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

## 6. Automated configuration and tuning

Use the generated configuration space when another optimiser or AutoML system must select ModularCMAES modules or numerical settings.

```python
from modcma.cmaescpp import get_configspace

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
cs_modules = get_configspace(dim=10,add_popsize=False,add_sigma=False,add_learning_rates=False)
```

Convert a sampled, default, or externally supplied configuration into a settings object before constructing the optimiser:

```python
import numpy as np
from modcma import c_maes
from modcma.cmaescpp import get_configspace, settings_from_config

dim = 10
cs = get_configspace(dim=dim)

# Sample or load a configuration
config = cs.sample_configuration()

# Or for defaults
default = cs.default_configuration()

# Configuration values may be edited before conversion.
config["sampler"] = "HALTON"

# Note that keyword arguments like lb in the next example, can be passed to settings like so
settings = settings_from_config(dim=dim,config=config,lb=np.ones(dim))

parameters = c_maes.Parameters(settings)
cma = c_maes.ModularCMAES(parameters)
cma.run(func)
```

When evaluating configurations, use comparable budgets, objective definitions, initialisation rules, and random-seed policies. Report both the selected module combination and numerical hyperparameters.

## 7. Integer variables

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

## 8. Legacy pure-Python implementation

The pure-Python implementation is not actively developed and may lack newer modules and performance improvements.

Use it only when justified:

```python
from modcma import fmin

xopt, fopt, used_budget = fmin(
    func=sum,
    x0=[1, 2, 3, 4],
    budget=1000,
    active=True,
    sigma0=2.5,
)
```

Or:

```python
import numpy as np
from modcma import ModularCMAES


def func(x: np.ndarray) -> float:
    return float(np.sum(x**2))


cma = ModularCMAES(func, dim=10, budget=10_000)
cma = cma.run()
```

## 9. Ask-tell interface

The ask-tell interface is available only in the legacy pure-Python implementation. Use it when objective evaluations must be managed externally, for example through simulations, APIs, or distributed execution.

```python
import numpy as np
from modcma import AskTellCMAES


def func(x: np.ndarray) -> float:
    return float(np.sum(x**2))


cma = AskTellCMAES(dim=10, budget=10_000, active=True)

while not cma.break_conditions():
    xi = cma.ask()
    fi = func(xi)
    cma.tell(xi, fi)
```

Do not attempt to use this interface through the C++ backend. For parallel or asynchronous evaluations, verify the optimiser's expected ask-tell semantics before submitting multiple outstanding candidates.

## 10. Module-selection guidelines

ModularCMAES groups its algorithmic choices into module categories. Options can generally be combined, but not every option is supported by both backends. Boolean modules are disabled by default. Must verify backend support and avoid enabling modules without explaining the intended effect.

### 10.1 Matrix adaptation

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

Use `COVARIANCE` as the normal baseline. Select alternatives only when dimensionality, runtime, separability assumptions, or the experiment design justify them. The extended alternatives are C++-backend features.

### 10.2 Active update

Active update incorporates poorly performing individuals with negative weights when adapting the covariance matrix.

```python
modules.active = True
```

It may improve adaptation, but it changes covariance updates materially. Treat `active=False` and `active=True` as distinct algorithm configurations and report the choice.

### 10.3 Elitism

Elitism switches from comma selection to plus selection, allowing strong solutions from the previous generation to survive.

```python
modules.elitist = True
```

It can accelerate convergence on unimodal problems, but may reduce population diversity and encourage premature convergence. Do not enable it automatically for multimodal exploration.

### 10.4 Orthogonal sampling

Orthogonal sampling applies a Gram-Schmidt procedure so newly sampled mutation directions are orthogonal.

```python
modules.orthogonal = True
```

Use it when more evenly distributed sampling directions are relevant. Since it modifies the sampling design, include it explicitly in experiment metadata.

### 10.5 Sequential selection

Sequential selection compares evaluated candidates against the current best and can stop evaluating the remainder of a generation once sufficiently many improvements have been found.

```python
modules.sequential_selection = True
```

This can reduce objective-function evaluations, especially when evaluations are expensive. Evaluation counts may then differ between generations and configurations, so comparisons must use actual evaluation counts rather than iteration counts.

### 10.6 Threshold convergence

Threshold convergence requires mutation vectors to satisfy a length threshold that decreases over time. It is intended to prolong exploration before transitioning towards local search.

```python
modules.threshold_convergence = True
```

Use it only when the exploration-to-exploitation schedule is part of the intended configuration.

### 10.7 Per-candidate sigma sampling

The sample-sigma module draws an individual step size for each candidate from a log-normal distribution based on the global `sigma`.

```python
modules.sample_sigma = True
```

This introduces additional self-adaptation. Report it separately from the global step-size adaptation method.

### 10.8 Base sampler and sample transformation

In the C++ backend, the base sampler generates points in the uniform unit hypercube. The sample transformer then maps those points to the target search distribution. These are separate decisions.

```python
modules.sampler = c_maes.options.BaseSampler.SOBOL
modules.sample_transformation = c_maes.options.SampleTranformerType.GAUSSIAN
```

Base-sampler choices include ordinary uniform sampling, Sobol sequences, and Halton sequences. Sobol and Halton are low-discrepancy alternatives intended to cover the unit hypercube more evenly.

Sample transformations include:

- `GAUSSIAN`, the default target distribution
- `SCALED_UNIFORM`
- `LAPLACE`
- `LOGISTIC`
- `CAUCHY`
- `DOUBLE_WEIBULL`

Do not describe `SOBOL` or `HALTON` alone as the final search distribution in the C++ backend. Always consider both `sampler` and `sample_transformation`. Preserve the API spelling `SampleTranformerType`, including its spelling, when that is the identifier exposed by the installed package.

### 10.9 Recombination weights

Recombination weights determine how selected individuals contribute to strategy updates.

```python
modules.weights = c_maes.options.RecombinationWeights.DEFAULT
```

Choices include `DEFAULT`, `EQUAL`, and `EXPONENTIAL`. The C++ `EXPONENTIAL` option corresponds to the pure-Python `"1/2^lambda"` choice. Use `DEFAULT` as the baseline unless alternative selection pressure is being tested.

### 10.10 Mirrored and pairwise sampling

Mirrored sampling generates opposite mutation vectors to improve coverage. Pairwise mode retains only the better candidate from each mirrored pair for recombination, preventing paired vectors from cancelling each other.

```python
modules.mirrored = c_maes.options.Mirror.MIRRORED
# or
modules.mirrored = c_maes.options.Mirror.PAIRWISE
```

Use `PAIRWISE` only when mirrored sampling is intended. Report whether mirroring is disabled, ordinary, or pairwise.

### 10.11 Step-size adaptation

The step-size adaptation module controls how the global search scale changes.

```python
modules.ssa = c_maes.options.StepSizeAdaptation.CSA
```

Documented choices include `CSA`, `TPA`, `MSR`, `XNES`, `MXNES`, `LPXNES`, and `PSR`, with additional C++ options potentially exposed by the installed release. Use `CSA` as the standard baseline. Treat a change in step-size adaptation as a major algorithmic change, not a minor numerical setting.

### 10.12 Restart strategy and criteria

Restart strategies respond to stagnation or other restart conditions:

- `NONE`: no configured restart strategy
- `RESTART`: restart without a population-size schedule
- `IPOP`: increase population size after each restart
- `BIPOP`: alternate between larger and smaller population regimes
- `STOP`: C++-only option that stops when a restart criterion is met

```python
modules.restart_strategy = c_maes.options.RestartStrategy.IPOP
```

The C++ backend allows restart triggers to be controlled through `cma.p.criteria.items`. Clearing this list disables criterion-triggered restarts:

```python
cma.p.criteria.items = []
```

Must not remove or alter restart criteria silently. If a criterion uses a class-level or static tolerance, changing it can affect every instance in the process.

Custom criteria must retain a Python reference rather than being constructed inline:

```python
import modcma


class MyCriterion(modcma.restart.Criterion):
    def __init__(self):
        super().__init__("MyCriterionName")

    def on_reset(self, par: modcma.Parameters):
        """Called when a restart happens (also at the start)"""
        pass

    def update(self, par: modcma.Parameters):
        
        """Called after each iteration, needs to modify self.met"""
        self.met = True


criterion = MyCriterion()
cma.p.criteria.items = [criterion]
```

Ensure `update()` sets `self.met` according to the actual condition. The unconditional `True` above is only a structural example.

### 10.13 Bound correction

Bound correction determines how infeasible candidates are handled.

```python
modules.bound_correction = c_maes.options.CorrectionMethod.SATURATE
```

Documented approaches include no correction, saturation or clipping, mirroring, toroidal wrapping, correction based on a truncated normal distribution (`COTN`), uniform resampling, and resampling. Choose a method that is compatible with the physical meaning of the variables.

### 10.15 Centre placement on restart

The C++-only centre-placement module controls the sampling distribution's centre after a restart:

- `X0`: return to the original initial point
- `ZERO`: use the all-zero vector
- `UNIFORM`: sample a point uniformly inside the search space
- `CENTER`: use the geometric centre of the bounded search space

Centre placement is initiated by a restart strategy. Validate that `ZERO` and `CENTER` are meaningful and feasible for the problem, and require finite bounds for choices that depend on the search-space extent.