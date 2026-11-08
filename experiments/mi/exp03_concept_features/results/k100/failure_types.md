# Failure types

| type | n_type | n_working | auroc_T | auroc_S | auroc_T+S | delta | ci_lo | ci_hi | n_features_S |
|---|---|---|---|---|---|---|---|---|---|
| ValueError | 421 | 6968 | 0.749 | 0.848 | 0.850 | 0.100 | 0.072 | 0.129 | 115.800 |
| TypeError | 265 | 6968 | 0.816 | 0.893 | 0.906 | 0.090 | 0.053 | 0.128 | 81.200 |
| ModuleNotFoundError | 205 | 6968 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 27.800 |
| IndexError | 162 | 6968 | 0.786 | 0.817 | 0.827 | 0.042 | 0.021 | 0.063 | 63.200 |
| AttributeError | 121 | 6968 | 0.607 | 0.783 | 0.786 | 0.179 | 0.108 | 0.242 | 21.500 |
| NameError | 92 | 6968 | 0.723 | 0.921 | 0.917 | 0.194 | 0.139 | 0.248 | 17.400 |

## ValueError (n = 421): T 0.749, S 0.848, T+S 0.850

Example error: `In the code, line 104, in __call__, the following error occurred: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all() On line: if new_fitness < self.f_opt:`

- **f3428** coef +1.15, AUROC 0.684, active 0.53 / 0.19; max tokens `','` ×55, `'.choice'` ×20, `'from'` ×9
  - 8.5: `perturbations = np.random.normal(0, self.step_size«,» (self.dim,`
  - 8.2: `perturbations = np.random.normal(0, self.step_size«,» (self.dim,`
- **f18553** coef +0.67, AUROC 0.724, active 0.88 / 0.50; max tokens `'.dot'` ×13, `'.outer'` ×6, `'np'` ×5
  - 11.9: `Apply gradient-based mutation to the new population new_population += np«.outer»(grad, np.ones`
  - 11.9: `# Track elites if len(self.elites) <« int»(self.elite_ratio`
- **f60118** coef +0.36, AUROC 0.601, active 0.26 / 0.07; max tokens `'from'` ×41, `'.choice'` ×27, `'between'` ×20
  - 13.6: `from the best or a random elite x = np.random.choice«([»p[0]`
  - 11.6: `urb = func(x_perturb) # Choose better of the« two» moves if`
- **f12826** coef +0.36, AUROC 0.646, active 0.51 / 0.24; max tokens `','` ×19, `'-'` ×16, `'1'` ×15
  - 22.0: `_pop) self.pop = np.where(better_indices[:, np.newaxis«,» :], trial, pop`
  - 22.0: `.f_pop) self.pop = np.where(better_indices[:, np«.newaxis», :], trial,`
- **f64822** coef +0.31, AUROC 0.557, active 0.22 / 0.11; max tokens `'.gradient'` ×61, `'grad'` ×24, `':'` ×2
  - 19.0: `mutation based on dimension importance dim_importance = np.abs(np«.gradient»(self.x_opt))`
  - 17.5: `adjust covariance based on dimensional importance dim_weights = np.abs(np«.gradient»(self.x_opt))`
- **f44116** coef +0.31, AUROC 0.605, active 0.51 / 0.34; max tokens `'int'` ×86, `'.randint'` ×5, `'[int'` ×2
  - 30.8: `_size) + k] = remaining_pop[idx1] new_fitness«[int»(self.elite_ratio`
  - 29.3: `.argsort(self.f_pop) elites = self.pop[sorted_indices[:«int»(self.pop_size *`
- **f10233** coef +0.25, AUROC 0.654, active 0.68 / 0.44; max tokens `''` ×10, `'='` ×10, `'np'` ×7
  - 11.5: `armony)) self.f_pop = np.append(self.f_pop,« new»_f) self`
  - 11.4: `self.pop.append((new_x, new_f)) self.pop«.sort»(key=lambda p:`
- **f47298** coef +0.25, AUROC 0.643, active 0.50 / 0.25; max tokens `'Sample'` ×7, `'select'` ×7, `'ly'` ×7
  - 16.3: `.budget - self.init_samples): # Bayesian sampling: generate candidate« points» from the kernel`
  - 14.4: `if np.random.rand() < self.hmcr: # Select« value» from harmony memory`

## TypeError (n = 265): T 0.816, S 0.893, T+S 0.906

Example error: `In the code, line 1273, in gradient, the following error occurred: TypeError: invalid number of arguments`

- **f64822** coef +2.48, AUROC 0.843, active 0.72 / 0.11; max tokens `'.gradient'` ×61, `'grad'` ×24, `':'` ×2
  - 19.0: `mutation based on dimension importance dim_importance = np.abs(np«.gradient»(self.x_opt))`
  - 17.5: `adjust covariance based on dimensional importance dim_weights = np.abs(np«.gradient»(self.x_opt))`
- **f5501** coef +0.89, AUROC 0.821, active 0.69 / 0.13; max tokens `','` ×78, `'self'` ×4, `'('` ×4
  - 8.4: `00: # Compute gradient grad = np.gradient(func«,» x, 1`
  - 7.6: `self.f_values[i]: grad = np.gradient(self.f_values[i«],» self.population[i])`
- **f7639** coef +0.83, AUROC 0.858, active 0.80 / 0.24; max tokens `'.fit'` ×17, `'_std'` ×12, `'alpha'` ×12
  - 14.5: `return func(x) def grad_func(x): return np«.gradient»(f(x))`
  - 13.8: `e-2, 1e2)) + WhiteKernel(noise«_level»=1e-`
- **f54067** coef +0.72, AUROC 0.806, active 0.67 / 0.13; max tokens `'.gradient'` ×65, `'crossover'` ×16, `'for'` ×2
  - 9.3: `.initial_samples = initial_samples self.alpha = alpha # mean« parameter» for Gaussian process`
  - 8.8: `paper c = 0.101 # Constant« from» SPSA paper`
- **f12306** coef +0.27, AUROC 0.786, active 0.65 / 0.15; max tokens `','` ×43, `'/'` ×27, `'))'` ×8
  - 8.3: `_factor self.last_grad = (f - self.f_opt)« /» self.step_size`
  - 7.8: `# Gradient-based perturbation grad = np.gradient(func(x«),» x, method='`
- **f31280** coef +0.20, AUROC 0.828, active 0.78 / 0.32; max tokens `'.gradient'` ×35, `','` ×23, `'1'` ×8
  - 8.9: `# Simple gradient descent step grad = np.gradient(func, x«,» axis=0)`
  - 8.9: `init__(self, budget=10000, dim=«1»0, pop_size`
- **f14888** coef +0.15, AUROC 0.797, active 0.62 / 0.07; max tokens `'.gradient'` ×68, `','` ×12, `'.bounds'` ×7
  - 7.0: `func): # Initialize population bounds = np.array([func«.bounds».lb, func.bounds`
  - 6.7: `Adaptive kernel lengthscale self.kernel_lengthscale = self.gp«.kernel»_.get_params()['`
- **f48121** coef +0.02, AUROC 0.800, active 0.70 / 0.19; max tokens `'.dim'` ×13, `'def'` ×12, `'acquisition'` ×10
  - 18.0: `] # Use gradient descent to refine the best acquisition point « def» objective(x):`
  - 17.2: `Select elite based on a combination of function value and diversity def« elite»_score(individual):`

## ModuleNotFoundError (n = 205): T 1.000, S 1.000, T+S 1.000

Example error: `In the code, line 2, in <module>, the following error occurred: ModuleNotFoundError: No module named 'sklearn' On line: from sklearn.gaussian_process import GaussianProcessRegressor`

- **f31417** coef +0.89, AUROC 1.000, active 1.00 / 0.00; max tokens `'Regressor'` ×17, `','` ×15, `'Process'` ×15
  - 25.7: `self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha«,» n_restarts_optimizer`
  - 25.5: `C if self.kernel == 'rbf': kernel =« R»BF(length_scale=`
- **f18520** coef +0.50, AUROC 1.000, active 1.00 / 0.00; max tokens `'sklearn'` ×96
  - 41.7: `import numpy as np from scipy.stats import gaussian_kde from« sklearn».gaussian_process import`
  - 41.5: `optimization. # Code: '''python import numpy as np from« sklearn».gaussian_process import`
- **f2211** coef +0.50, AUROC 1.000, active 1.00 / 0.00; max tokens `'sklearn'` ×96
  - 11.6: `exploitation. # Code: '''python import numpy as np from« sklearn».gaussian_process import`
  - 11.5: `as np from sklearn.gaussian_process import GaussianProcessRegressor from« sklearn».gaussian_process.k`
- **f58669** coef +0.36, AUROC 1.000, active 1.00 / 0.01; max tokens `'Regressor'` ×38, `'BF'` ×37, `'Kernel'` ×21
  - 18.8: `import numpy as np from sklearn.gaussian_process import GaussianProcess«Regressor» from sklearn.g`
  - 18.5: `Regressor from sklearn.gaussian_process.kernels import RBF, Constant«Kernel» class BayesianRandom`
- **f15766** coef +0.35, AUROC 1.000, active 1.00 / 0.00; max tokens `'_process'` ×88, `'aussian'` ×6, `'.g'` ×2
  - 15.6: `numpy as np from scipy.stats import norm from sklearn.gaussian«_process» import GaussianProcessRegressor`
  - 15.5: `numpy as np from scipy.stats import beta from sklearn.gaussian«_process» import GaussianProcessRegressor`
- **f33909** coef +0.28, AUROC 1.000, active 1.00 / 0.00; max tokens `'.k'` ×92, `'import'` ×4
  - 5.9: `sklearn.gaussian_process import GaussianProcessRegressor from sklearn.gaussian_process«.k»ernels import RBF`
  - 5.9: `.values) # Initialize Gaussian process from sklearn.gaussian_process« import» GaussianProcessRegressor`
- **f60689** coef +0.24, AUROC 1.000, active 1.00 / 0.00; max tokens `'.k'` ×50, `'ernels'` ×46
  - 9.4: `.gaussian_process import GaussianProcessRegressor from sklearn.gaussian_process«.k»ernels import RBF`
  - 9.4: `sklearn.gaussian_process import GaussianProcessRegressor from sklearn.gaussian_process«.k»ernels import RBF`
- **f33709** coef +0.24, AUROC 1.000, active 1.00 / 0.00; max tokens `'aussian'` ×59, `'.g'` ×21, `'.k'` ×16
  - 19.8: `sklearn.gaussian_process import GaussianProcessRegressor from sklearn.gaussian_process«.k»ernels import RBF`
  - 19.5: `numpy as np from scipy.stats import gaussian_kde from sklearn«.g»aussian_process import Gaussian`

## IndexError (n = 162): T 0.786, S 0.817, T+S 0.827

Example error: `In the code, line 42, in __call__, the following error occurred: IndexError: list index out of range On line: a, b, c = self.population[rand_idx[0]], self.population[rand_idx[1]], self.population[rand_idx[2]]`

- **f43344** coef +1.19, AUROC 0.681, active 0.41 / 0.07; max tokens `''` ×34, `'='` ×17, `'('` ×12
  - 6.7: `0.5, 1.5) # Random selection of« »3 distinct indices`
  - 6.4: `if self.pop_size > 5: self.pop_size = max«(»5, self.pop`
- **f10090** coef +0.47, AUROC 0.690, active 0.51 / 0.16; max tokens `''` ×11, `'rink'` ×9, `'_size'` ×8
  - 10.7: `= np.sort(self.f_pop)[:self.pop_size] # Optionally« grow» population if not at`
  - 10.1: `_new # Dynamic population control if evals < self«.b»udget: #`
- **f44116** coef +0.34, AUROC 0.694, active 0.63 / 0.34; max tokens `'int'` ×86, `'.randint'` ×5, `'[int'` ×2
  - 30.8: `_size) + k] = remaining_pop[idx1] new_fitness«[int»(self.elite_ratio`
  - 29.3: `.argsort(self.f_pop) elites = self.pop[sorted_indices[:«int»(self.pop_size *`
- **f42549** coef +0.16, AUROC 0.716, active 0.70 / 0.32; max tokens `'[:'` ×65, `'):'` ×7, `'[-'` ×7
  - 36.3: `# Multi-directional exploration directions = self.directions«[:»self.max_directions`
  - 36.1: `_size * self.elitism_ratio) new_population[:elite_size«]» = self.population[:`
- **f43215** coef +0.12, AUROC 0.756, active 0.83 / 0.49; max tokens `'int'` ×51, `'.array'` ×25, `'(int'` ×9
  - 26.4: `(2 * 1e-6) self.grad_cache[t«uple»(trial.flatten())`
  - 26.3: `np.linspace(learning_rate, learning_rate * 0.1,« int»(budget /`
- **f56625** coef +0.12, AUROC 0.669, active 0.47 / 0.16; max tokens `''` ×28, `'needed'` ×7, `'duction'` ×6
  - 6.5: `.evaluation_count # Occasionally replace population with new random points « » if np.random.rand`
  - 5.9: `< self.f_opt: self.f_opt = best_f « self».x_opt = best`
- **f13558** coef +0.04, AUROC 0.691, active 0.57 / 0.22; max tokens `')[:'` ×48, `'][:'` ×25, `'[:'` ×8
  - 11.4: `[:-1] = np.exp(-0.5 * np.dot(samples«[:»self.init_samples]`
  - 9.8: `self.directions = np.vstack([new_dir, self.directions«[:»self.max_directions`
- **f54856** coef +0.04, AUROC 0.692, active 0.69 / 0.36; max tokens `'_ratio'` ×53, `'_indices'` ×25, `'indices'` ×12
  - 8.5: `5.0, (int(self.pop_size * self.subpop_size«_ratio»), self.dim))`
  - 8.3: `for j in range(self.pop_size): # Select three distinct« indices» (excluding elites)`

## AttributeError (n = 121): T 0.607, S 0.783, T+S 0.786

Example error: `In the code, line 51, in _evaluate_fitness, the following error occurred: AttributeError: 'HybridAdaptiveMomentumSearch' object has no attribute 'func' On line: return self.func(x)`

- **f22411** coef +0.66, AUROC 0.690, active 0.45 / 0.10; max tokens `''` ×21, `'0'` ×4, `')'` ×4
  - 17.1: `_i = np.clip(pop_i, func.bounds.lb, func.bounds.«ub») self.pop`
  - 15.8: `self.Y = np.array([func(x) for x in self.ac«quisition»_function(self.X)])`
- **f9410** coef +0.42, AUROC 0.824, active 0.90 / 0.75; max tokens `''` ×13, `'))'` ×6, `')'` ×6
  - 30.6: `.lb, func.bounds.ub, (self.pop_size, self.dim«)) » fitness = np`
  - 28.1: `(np.mean(fitness)) return self.f_opt, self.x_opt« »'''`

## NameError (n = 92): T 0.723, S 0.921, T+S 0.917

Example error: `In the code, line 83, in _learn_from_best, the following error occurred: NameError: name 'func' is not defined On line: pop_i = np.clip(pop_i, func.bounds.lb, func.bounds.ub)`

- **f7944** coef +2.71, AUROC 0.851, active 0.74 / 0.07; max tokens `'func'` ×13, `'.gradient'` ×6, `'pop'` ×5
  - 15.4: `_scale * np.random.randn(self.dim) candidate = current_x +« step» * direction + noise`
  - 13.4: `(1, -1), return_std=True) return self.ac«quisition»_weight * (-mean`
- **f3197** coef +0.15, AUROC 0.650, active 0.30 / 0.01; max tokens `'import'` ×28, `'.stats'` ×26, `'scipy'` ×25
  - 43.4: `.kernels import RBF, ConstantKernel as C from scipy.stats« import» norm class Bayesian`
  - 43.2: `_process.kernels import RBF, ConstantKernel as C from scipy«.optimize» import minimize class`
- **f61717** coef +0.01, AUROC 0.807, active 0.72 / 0.17; max tokens `'norm'` ×16, `'BF'` ×9, `'Process'` ×8
  - 18.9: `and covariance def acquisition(x): # Prior distribution (G«aussian») prior =`
  - 15.9: `.x_opt = self.pop[self.best_idx] self._update_ac«quisition»_model(self.x_opt`
