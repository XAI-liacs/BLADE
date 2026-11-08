# Failure probe on the SAE code

## Gain of the SAE code over metadata + labels + text, by stratum

| stratum | n | fail_rate | auroc_MLT | auroc_MLTS | delta | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| all | 8381 | 0.169 | 0.804 | 0.877 | 0.074 | 0.058 | 0.089 |
| no surrogate | 8150 | 0.147 | 0.770 | 0.856 | 0.086 | 0.068 | 0.103 |
| no surrogate/restart | 7853 | 0.146 | 0.774 | 0.858 | 0.084 | 0.065 | 0.102 |
| parent worked | 7712 | 0.152 | 0.794 | 0.874 | 0.080 | 0.062 | 0.097 |
| parent failed | 333 | 0.495 | 0.604 | 0.751 | 0.146 | 0.082 | 0.221 |
| operator=init | 336 | 0.235 | 0.833 | 0.893 | 0.060 | 0.016 | 0.105 |
| operator=new | 2715 | 0.270 | 0.803 | 0.882 | 0.079 | 0.060 | 0.099 |
| operator=refine | 2656 | 0.154 | 0.725 | 0.804 | 0.079 | 0.048 | 0.109 |
| operator=simplify | 2674 | 0.071 | 0.775 | 0.894 | 0.119 | 0.078 | 0.158 |

## Sparse (L1) probes: out-of-fold AUROC vs number of features

| layer | C | n_features | auroc_S | auroc_MLT+S | delta_vs_MLT | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| 9 | 0.002 | 5.800 | 0.695 | 0.807 | 0.004 | 0.000 | 0.007 |
| 9 | 0.005 | 5.800 | 0.712 | 0.809 | 0.005 | 0.002 | 0.009 |
| 9 | 0.010 | 24.800 | 0.801 | 0.837 | 0.034 | 0.022 | 0.046 |
| 9 | 0.020 | 57.900 | 0.822 | 0.845 | 0.041 | 0.028 | 0.056 |
| 9 | 0.050 | 164.600 | 0.848 | 0.859 | 0.056 | 0.041 | 0.070 |
| 9 | 0.100 | 390.500 | 0.861 | 0.867 | 0.064 | 0.048 | 0.078 |
| 15 | 0.002 | 3.200 | 0.609 | 0.802 | -0.002 | -0.003 | -0.000 |
| 15 | 0.005 | 10.200 | 0.730 | 0.809 | 0.006 | 0.001 | 0.010 |
| 15 | 0.010 | 28.000 | 0.788 | 0.832 | 0.028 | 0.017 | 0.041 |
| 15 | 0.020 | 68.700 | 0.832 | 0.849 | 0.046 | 0.032 | 0.060 |
| 15 | 0.050 | 205.000 | 0.871 | 0.874 | 0.070 | 0.055 | 0.085 |
| 15 | 0.100 | 508.500 | 0.884 | 0.886 | 0.082 | 0.067 | 0.097 |
| 21 | 0.002 | 5.700 | 0.669 | 0.804 | 0.001 | -0.002 | 0.004 |
| 21 | 0.005 | 21.100 | 0.789 | 0.827 | 0.023 | 0.015 | 0.032 |
| 21 | 0.010 | 40.300 | 0.822 | 0.845 | 0.042 | 0.029 | 0.055 |
| 21 | 0.020 | 80.000 | 0.860 | 0.868 | 0.065 | 0.051 | 0.078 |
| 21 | 0.050 | 261.000 | 0.891 | 0.892 | 0.089 | 0.075 | 0.103 |
| 21 | 0.100 | 702.700 | 0.896 | 0.897 | 0.094 | 0.079 | 0.108 |
| 27 | 0.002 | 5.500 | 0.703 | 0.809 | 0.005 | 0.001 | 0.009 |
| 27 | 0.005 | 13.300 | 0.768 | 0.817 | 0.013 | 0.008 | 0.019 |
| 27 | 0.010 | 28.900 | 0.820 | 0.843 | 0.039 | 0.027 | 0.051 |
| 27 | 0.020 | 73.500 | 0.848 | 0.858 | 0.054 | 0.041 | 0.069 |
| 27 | 0.050 | 232.300 | 0.876 | 0.878 | 0.074 | 0.061 | 0.089 |
| 27 | 0.100 | 642.400 | 0.885 | 0.886 | 0.083 | 0.069 | 0.096 |
| all | 0.002 | 10.400 | 0.705 | 0.809 | 0.005 | 0.002 | 0.009 |
| all | 0.005 | 25.400 | 0.798 | 0.830 | 0.026 | 0.017 | 0.035 |
| all | 0.010 | 58.900 | 0.831 | 0.849 | 0.045 | 0.033 | 0.058 |
| all | 0.020 | 118.000 | 0.866 | 0.871 | 0.068 | 0.054 | 0.083 |
| all | 0.050 | 373.800 | 0.894 | 0.894 | 0.091 | 0.077 | 0.105 |
| all | 0.100 | 1004.900 | 0.899 | 0.899 | 0.096 | 0.082 | 0.110 |

## Features of the L1 probe at C = 0.01, layer 21 (all data)

- **f7639** coef +0.96, AUROC 0.732, active 0.61 / 0.24 (failing / working); top examples fail 1.00; max tokens `'.fit'` ×17, `'_std'` ×12, `'alpha'` ×12
  - 14.5: `return func(x) def grad_func(x): return np«.gradient»(f(x))`
  - 13.8: `e-2, 1e2)) + WhiteKernel(noise«_level»=1e-`
  - 13.4: `guide search # Predict on current population self.gp«.fit»(self.X, self`
- **f6057** coef -0.64, AUROC 0.472, active 0.90 / 1.00 (failing / working); top examples fail 0.33; max tokens `''` ×96
  - 8.5: `in continuous optimization. # Code: '''python import numpy as np« »class AdaptiveDifferential`
  - 8.5: `in continuous optimization. # Code: '''python import numpy as np« »class AdaptiveCov`
  - 8.3: `exploration and exploitation. # Code: '''python import numpy as np« »class GradientAdaptive`
- **f18553** coef +0.39, AUROC 0.726, active 0.87 / 0.50 (failing / working); top examples fail 0.44; max tokens `'.dot'` ×13, `'.outer'` ×6, `'np'` ×5
  - 11.9: `Apply gradient-based mutation to the new population new_population += np«.outer»(grad, np.ones`
  - 11.9: `# Track elites if len(self.elites) <« int»(self.elite_ratio`
  - 11.9: `, axis=0) else: self.grad_history = np«.array»([grad])`
- **f29748** coef +0.38, AUROC 0.745, active 0.70 / 0.31 (failing / working); top examples fail 1.00; max tokens `'Process'` ×11, `'='` ×8, `','` ×6
  - 7.8: `.gpr.predict(x.reshape(1, -1))[0] « sigma» = std[np.arg`
  - 7.6: `self.X = pop self.y = fitness self.g«p» = GaussianProcessRegressor`
  - 7.5: `) # Predictive mean and variance mean, std =« self».gpr.predict(self`
- **f50086** coef -0.36, AUROC 0.329, active 0.52 / 0.80 (failing / working); top examples fail 0.08; max tokens `'budget'` ×48, `'F'` ×33, `'_decay'` ×2
  - 6.0: `diversity_threshold=0.3): self.budget =« budget» self.dim`
  - 5.9: `udget = budget self.dim = dim self.F =« F» # Mutation factor`
  - 5.9: `udget = budget self.dim = dim self.F =« F» # Mutation factor`
- **f64822** coef +0.34, AUROC 0.608, active 0.30 / 0.11 (failing / working); top examples fail 0.90; max tokens `'.gradient'` ×61, `'grad'` ×24, `':'` ×2
  - 19.0: `mutation based on dimension importance dim_importance = np.abs(np«.gradient»(self.x_opt))`
  - 17.5: `adjust covariance based on dimensional importance dim_weights = np.abs(np«.gradient»(self.x_opt))`
  - 16.6: `=0)) pop_curvature = np.abs(np.gradient(np.array«([»func(x) for`
- **f5501** coef +0.33, AUROC 0.639, active 0.39 / 0.13 (failing / working); top examples fail 0.90; max tokens `','` ×78, `'self'` ×4, `'('` ×4
  - 8.4: `00: # Compute gradient grad = np.gradient(func«,» x, 1`
  - 7.6: `self.f_values[i]: grad = np.gradient(self.f_values[i«],» self.population[i])`
  - 7.4: `descent step on the best solution gradient = np.gradient(best_fitness«,» best_solution)`
- **f15097** coef -0.32, AUROC 0.323, active 0.56 / 0.79 (failing / working); top examples fail 0.01; max tokens `'0'` ×14, `']'` ×13, `'ers'` ×4
  - 7.6: `x_new = np.clip(x_new, func.bounds.lb, func«.bounds».ub)`
  - 7.6: `self.F_decay = F_decay self.CR_initial =« CR»_initial self`
  - 7.6: `self.F * (self.pop[idxs[1]] - self.pop«[idx»s[2]])`
- **f46455** coef -0.31, AUROC 0.324, active 0.87 / 0.98 (failing / working); top examples fail 0.18; max tokens `'import'` ×85, `'from'` ×11
  - 3.4: `'''python import numpy as np from scipy.stats import norm «from» sklearn.gaussian_process`
  - 3.4: `adaptive mutation. # Code: '''python import numpy as np «from» scipy.stats import norm`
  - 3.4: `urbation and Dynamic Trust Region Refinement # Code: '''python «import» numpy as np`
- **f30141** coef -0.29, AUROC 0.369, active 0.66 / 0.85 (failing / working); top examples fail 0.15; max tokens `'.copy'` ×21, `''` ×19, `','` ×7
  - 6.0: `x_plus[i] += eps x_minus = x.copy() « » x_minus[i]`
  - 6.0: `# Generate new population new_population = np.zeros((pop_size«,» self.dim))`
  - 5.8: `_size = step_size self.refine_factor = refine_factor « » self.step_decay =`
- **f50563** coef -0.23, AUROC 0.352, active 0.69 / 0.88 (failing / working); top examples fail 0.07; max tokens `''` ×57, `'.'` ×15, `'.b'` ×6
  - 4.7: `.vstack((self.pop, np.random.uniform(func.bounds.lb, func.bounds«.»ub, (1`
  - 4.7: `self.budget_decay) return self.f_opt, self.x_opt« »'''`
  - 4.6: `0.4) return self.f_opt, self.x_opt« »'''`
- **f59943** coef -0.19, AUROC 0.435, active 0.49 / 0.67 (failing / working); top examples fail 0.16; max tokens `'pert'` ×36, `'urbation'` ×31, `'ert'` ×9
  - 19.6: `) / (2 * self.learning_rate) # Apply gradient« pert»urbation self`
  - 19.4: `python import numpy as np class DE_Chaos_Pert«urbation»_Diversity:`
  - 19.2: `.zeros(self.dim) for j in range(self.dim): pert«urb» = np.zeros(self`
- **f13690** coef -0.16, AUROC 0.390, active 0.42 / 0.64 (failing / working); top examples fail 0.16; max tokens `'-'` ×86, `'+'` ×9, `''` ×1
  - 16.0: `1e-5 curvature = (func(x_plus)« +» func(x_minus)`
  - 15.9: `derivative hessian[i, j] = (f_plus_i« +» f_minus_i -`
  - 14.8: `= (func(self.x_opt + self.trust_radius * e)« -» func(self.x_opt`
- **f42754** coef -0.14, AUROC 0.356, active 0.74 / 0.91 (failing / working); top examples fail 0.05; max tokens `'.'` ×45, `'9'` ×13, `'='` ×11
  - 6.1: `=1.0, diversity_factor=0.5, decay_rate«=»0.99`
  - 6.0: `.8, pop_size=5, grad_smoothing=0.«9»5, adaptive_m`
  - 6.0: `=0.3, beta=0.9, gamma=0«.»99, sigma`
- **f3847** coef -0.11, AUROC 0.366, active 0.67 / 0.84 (failing / working); top examples fail 0.02; max tokens `'.f'` ×25, `'0'` ×15, `'f'` ×7
  - 6.4: `exploration rate over time self.exploration = max(self.ex«pl»oration * 0`
  - 6.2: `+= step f = func(x) if f < self«.f»_opt: self`
  - 6.0: `() # Decay learning rate self.learning_rate *= self.dec«ay»_rate return`
- **f21020** coef -0.11, AUROC 0.439, active 0.95 / 0.99 (failing / working); top examples fail 0.08; max tokens `''` ×91, `'.x'` ×2, `'keep'` ×1
  - 7.5: `stage += 1 return self.f_opt, self.x_opt« »'''`
  - 7.3: `steps += self.mu return self.f_opt, self.x_opt« »'''`
  - 7.3: `.best_fitness_history.append(best_f) return best_f, best_x« »'''`
- **f13579** coef -0.10, AUROC 0.480, active 0.75 / 0.88 (failing / working); top examples fail 0.21; max tokens `'9'` ×90, `'1'` ×2, `'0'` ×2
  - 43.6: `1, grad_freq=50, momentum=0.«9», lr_decay=`
  - 43.6: `scaling = 1.96 * std # «9»5% confidence interval`
  - 43.4: `) else: self.inertia_weight = min(0.«9», self.inertia`
- **f15688** coef +0.08, AUROC 0.652, active 0.34 / 0.05 (failing / working); top examples fail 0.91; max tokens `'Kernel'` ×11, `','` ×6, `'as'` ×5
  - 11.0: `+ std * norm.pdf(Z) imp[imp < 0«]» = 0`
  - 10.5: `std, best_f): with np.errstate(divide='ignore«',» invalid='ignore'):`
  - 10.1: `, mean, std, best_f): with np.errstate(div«ide»='ignore'):`
- **f46296** coef -0.07, AUROC 0.400, active 0.40 / 0.58 (failing / working); top examples fail 0.08; max tokens `'grad'` ×87, `'pert'` ×4, `'pop'` ×1
  - 6.6: `(self, x, func, eps=1e-6): « grad» = np.zeros_like`
  - 6.4: `(self.local_search_steps): # Generate a random perturbation « pert»urbation = np.random`
  - 5.9: `current_x = population[best_idx] # Estimate gradient « grad» = self._estimate`
- **f21345** coef -0.06, AUROC 0.319, active 0.49 / 0.76 (failing / working); top examples fail 0.02; max tokens `''` ×11, `']'` ×10, `'.b'` ×4
  - 7.7: `< self.CR or trial[j] == mutant[j]: trial[j«]» = mutant[j]`
  - 7.6: `== 0 and i > 0: # Refine every« »10 iterations`
  - 7.4: `ity_threshold: self.dim = int(np.ceil(self.dim * self«.re»duction_factor))`
- **f8969** coef -0.06, AUROC 0.345, active 0.22 / 0.49 (failing / working); top examples fail 0.01; max tokens `']'` ×7, `'.b'` ×6, `'.x'` ×6
  - 9.0: `) f_local = np.array([func(x) for x in« local»_points]) #`
  - 8.7: `) f_local = np.array([func(x) for x in« local»_points]) #`
  - 8.7: `k in range(self.dim): if np.random.rand() < self«.c»rossover_rate:`
- **f59012** coef -0.06, AUROC 0.374, active 0.21 / 0.45 (failing / working); top examples fail 0.08; max tokens `'lo'` ×20, `'.x'` ×14, `'pl'` ×8
  - 4.7: `_sample, gpr, xi=0.01): """«Expected» Improvement acquisition function."""`
  - 4.4: `budget = self.budget exploration_steps = int(self.expl«oration»_ratio * budget)`
  - 4.3: `solution perturb = np.random.normal(0, self.p«ert»urb_scale, self`
- **f3428** coef +0.06, AUROC 0.642, active 0.47 / 0.19 (failing / working); top examples fail 0.30; max tokens `','` ×55, `'.choice'` ×20, `'from'` ×9
  - 8.5: `perturbations = np.random.normal(0, self.step_size«,» (self.dim,`
  - 8.2: `perturbations = np.random.normal(0, self.step_size«,» (self.dim,`
  - 7.9: `offspring grad_perturb = np.zeros(self.pop_size,« dtype»=bool) for`
- **f54912** coef -0.05, AUROC 0.387, active 0.50 / 0.71 (failing / working); top examples fail 0.14; max tokens `'for'` ×8, `'pert'` ×6, `'random'` ×6
  - 6.7: `# Code: '''python import numpy as np class Adaptive«M»omentumDrivenRandomSearch`
  - 6.4: `] = amplitude * np.sin(frequency * i) # Apply« pert»urbations to current`
  - 6.0: `else: self.sigma *= 0.95 #« Random» perturbation to avoid`
- **f15550** coef +0.04, AUROC 0.636, active 0.36 / 0.09 (failing / working); top examples fail 0.73; max tokens `'0'` ×10, `'self'` ×7, `'(func'` ×5
  - 10.4: `1, size=self.dim) pop_i = np.clip(pop_i«,» func.bounds.lb,`
  - 8.6: `Sort by diversity self.population.sort(key=lambda x: diversity[self«.population».index(x)])`
  - 8.3: `.gradient_history[j] * self.gradient_decay + np.gradient(self.fitness«[j»], self.pop[j`
- **f23504** coef -0.04, AUROC 0.371, active 0.74 / 0.92 (failing / working); top examples fail 0.11; max tokens `''` ×29, `'np'` ×29, `'func'` ×12
  - 4.3: `for Continuous Optimization # Code: '''python import numpy as np« »class AdaptiveHybrid`
  - 3.5: `ADRMDE) # Code: '''python import numpy as np« »class ADRM`
  - 3.4: `and Elite Diversity Preservation # Code: '''python import numpy as« np» class AdaptiveD`
- **f5392** coef -0.03, AUROC 0.397, active 0.46 / 0.70 (failing / working); top examples fail 0.12; max tokens `''` ×22, `'('` ×16, `'-'` ×7
  - 11.3: `.gradient_estimate = (x2 - x1) * (f2« -» f1) /`
  - 11.1: `[-2] val_diff = next_value - self.best_value « » self.gradient_estimate +=`
  - 11.1: `_history[-1] grad = (f_curr - f_prev)« /» (np.linalg.norm`
- **f55055** coef -0.03, AUROC 0.354, active 0.58 / 0.80 (failing / working); top examples fail 0.03; max tokens `'init'` ×96
  - 7.5: `numpy as np class HybridGradientMomentumSearch: def __«init»__(self, budget`
  - 4.6: `class AdaptiveHybridDifferentialEvolutionGradient: def __«init»__(self, budget`
  - 4.6: `class EnhancedAdaptiveGradientPerturbationSearch: def __«init»__(self, budget`
- **f8399** coef -0.03, AUROC 0.392, active 0.81 / 0.95 (failing / working); top examples fail 0.02; max tokens `'dim'` ×35, `'='` ×25, `'budget'` ×16
  - 5.6: `__(self, budget=10000, dim=1«0», step_size=`
  - 5.2: `def __init__(self, budget=10000,« dim»=10,`
  - 5.1: `class AdaptiveGradientMomentumES: def __init__(self,« budget»=100`
- **f60549** coef -0.02, AUROC 0.332, active 0.74 / 0.91 (failing / working); top examples fail 0.05; max tokens `'for'` ×96
  - 8.1: `.x_opt = x # Local gradient-based refinement phase « for» _ in range(self`
  - 8.0: `random_budget = int(self.budget * self.alpha) « for» _ in range(random`
  - 7.9: `random search for exploration if rand_budget > 0: « for» _ in range(min`
- **f58070** coef -0.02, AUROC 0.355, active 0.90 / 0.96 (failing / working); top examples fail 0.21; max tokens `'('` ×93, `'(('` ×2, `'1'` ×1
  - 17.2: `self.step_size) self.hessian_estimate[i, i] =« (»grad_plus - grad`
  - 16.0: `self.step_size ** 2) self.curvature_matrix = («1» - self.curvature`
  - 16.0: `.pop() # Diversity and restart mechanism if evaluations_used %« (»self.update_frequency *`
- **f18169** coef -0.02, AUROC 0.403, active 0.82 / 0.90 (failing / working); top examples fail 0.09; max tokens `'0'` ×62, `'Enhanced'` ×5, `'Ad'` ×4
  - 5.4: `Code: '''python import numpy as np class HADSD«M»_DM: def`
  - 5.2: `= initial_step self.grad_step = grad_step self«.ex»ploration_ratio =`
  - 5.2: `. # Code: '''python import numpy as np class Enhanced«Ad»aptiveDirectionalEv`
- **f55948** coef -0.01, AUROC 0.380, active 0.69 / 0.84 (failing / working); top examples fail 0.09; max tokens `''` ×52, `']'` ×24, `'.'` ×9
  - 5.3: `]: self.pop[j] = trial self.fitness[j«]» = f_trial`
  - 5.3: `]: self.pop[j] = trial self.fitness[j«]» = f_trial`
  - 5.3: `]: self.pop[j] = trial self.fitness[j«]» = f_trial`
- **f14082** coef -0.01, AUROC 0.347, active 0.58 / 0.80 (failing / working); top examples fail 0.05; max tokens `'Reduce'` ×9, `'focus'` ×8, `'close'` ×7
  - 12.6: `) / (2 * 1e-4) # Move« in» the direction opposite to`
  - 12.3: `current_step *= self.shrink_factor # Add small decay to« prevent» step size from growing`
  - 12.2: `self.mutation_factor *= self.exploration_decay # Reduce mutation« to» focus on exploitation`
- **f59809** coef -0.01, AUROC 0.348, active 0.49 / 0.74 (failing / working); top examples fail 0.07; max tokens `'='` ×11, `''` ×9, `','` ×8
  - 7.3: `= np.mean(np.sum(np.abs(self.population[np.newaxis, :, :]« -» self.population[:, np`
  - 6.9: `~improved] # Update global best current_best_idx« =» np.argmin(f`
  - 6.8: `(np.sum(np.abs(self.population[np.newaxis, :, :] - self.population«[:,» np.newaxis, :`
- **f582** coef -0.01, AUROC 0.401, active 0.77 / 0.89 (failing / working); top examples fail 0.20; max tokens `'='` ×76, `'_trial'` ×7, `']'` ×4
  - 8.8: `< self.fitness[j]: self.fitness[j] = f«_trial» self.pop`
  - 8.7: `) if f_trial < self.f_opt: self.f_opt« =» f_trial`
  - 8.5: `f_trial < self.f_values[j]: self.f_values[j]« =» f_trial`
- **f44060** coef -0.00, AUROC 0.370, active 0.52 / 0.74 (failing / working); top examples fail 0.09; max tokens `'pert'` ×10, `'__('` ×7, `'.x'` ×5
  - 4.6: `_factor * np.random.randn(self.dim) x_rand = np.clip«(x»_rand, func.bounds`
  - 4.5: `x = np.clip(x, func.bounds.lb, func.bounds«.»ub) f`
  - 4.5: `self.f_opt: self.f_opt = f_rand self«.x»_opt = x_rand`
