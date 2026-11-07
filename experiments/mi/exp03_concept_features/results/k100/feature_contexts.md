# Top features per concept

## L9 · population

- **f25223** AUROC 0.987 (val), active 0.99 / 0.02; max tokens `'.pop'` ×33, `'.population'` ×23, `'population'` ×11
  - 9.4: `= np.random.uniform(func.bounds.lb, func.bounds.ub, («pop»_size, self.dim`
  - 9.4: `_opt = x # Maintain population diversity if len(self«.population») > self.init`
  - 9.3: `=0.3, decay_rate=0.995,« pop»_size=None, elite`
  - 9.2: `budget self.dim = dim self.population_size = population«_size» self.learning`
  - 8.9: `np.eye(self.dim) * self.init_std**2 self«.population» = np.random.normal`
- **f18382** AUROC 0.977 (val), active 0.96 / 0.01; max tokens `'.pop'` ×48, `'pop'` ×19, `'_pop'` ×12
  - 14.3: `.pop = [] # Initial population for _ in range(self«.pop»_size): x`
  - 14.2: `* self.sigma new_pop = mean + np.random.randn(self«.pop»size, self.dim`
  - 14.2: `.randn(self.popsize, self.dim) * self.sigma new«_pop» = mean + np`
  - 13.9: `self.sigma = sigma self.pop_size = pop_size« » self.refine`
  - 13.8: `self.pop = np.vstack((self.pop, samples)) self.f«_pop» = np.concatenate((`
- **f15233** AUROC 0.977 (val), active 0.98 / 0.10; max tokens `'population'` ×33, `'.pop'` ×17, `'.population'` ×10
  - 5.9: `population for diversity population.append(x.copy()) if len(pop«ulation») > 5`
  - 5.8: `- sigma: Standard deviation for random perturbations. -« pop»_size: Size of`
  - 5.3: `a small elite pool for gradient estimation if len(self.elite«_pool») > int(self`
  - 5.2: `_opt = None total_evaluations = 0 population« =» [] # Initialize`
  - 5.1: `0 while evaluations < self.budget: # Generate a« population» of candidate solutions`

## L9 · differential_evolution

- **f18778** AUROC 1.000 (val), active 1.00 / 0.18; max tokens `'mutant'` ×53, `'mutants'` ×40, `'adapt'` ×1
  - 3.1: `parent parent = np.random.choice(pop) # Generate a« mutant» by adding noise`
  - 3.1: `np.random.choice(self.pop_size, 3, replace=False) « mutant» = pop[a]`
  - 3.0: `.random.rand() < self.crossover_rate: trial[j] =« mutant»[j] samples`
  - 3.0: `itness) for i in range(self.budget): # Generate« mutant» vectors mutants`
  - 3.0: `, axis=0), (self.pop_size, self.dim)) « mutants» = np.clip(mut`
- **f29715** AUROC 0.999 (val), active 1.00 / 0.13; max tokens `'mutant'` ×84, `'ant'` ×6, `'ary'` ×5
  - 4.2: `.F * (b - c) # Boundary handling « mutant» = np.clip(mut`
  - 4.1: `ant, func.bounds.lb, func.bounds.ub) f_mut«ant» = func(mutant`
  - 4.1: `trial[k] == pop[j][k]: trial[k] =« mutant»[k] #`
  - 4.1: `if np.random.rand() < self.CR or trial[j] ==« mutant»[j]: trial`
  - 4.1: `.choice(self.pop_size, size=self.pop_size, replace=True) « mutant» = population[parents`
- **f45243** AUROC 0.999 (val), active 1.00 / 0.00; max tokens `'mutant'` ×85, `'mutants'` ×9, `'Vari'` ×1
  - 4.5: `optimization. # Code: '''python import numpy as np class« Vari»anceAdaptiveGradient`
  - 4.0: `j in range(self.pop_size): parent = swarm[j] « mutant» = mutants[j]`
  - 3.9: `itness)] for i in range(self.budget): # Generate« mutants» for j`
  - 3.8: `copy() for i in range(self.budget): # Generate« mutants» for j`
  - 3.6: `] best_level_idx = np.argmin(mutant_fitness) « mutant» = mutant_levels[`

## L9 · crossover

- **f56696** AUROC 0.986 (val), active 0.99 / 0.12; max tokens `'rossover'` ×51, `'crossover'` ×20, `'utation'` ×18
  - 2.2: `= dim self.pop_size = pop_size self.m«utation»_factor = mutation_factor`
  - 2.2: `0.99, mutation_rate=0.8,« crossover»_rate=0.`
  - 2.2: `_size self.mutation_rate = mutation_rate self.c«rossover»_rate = crossover_rate`
  - 2.2: `mutation_rate=0.1, mutation_std=0.5,« crossover»_rate=0.`
  - 2.1: `_strength self.elite_ratio = elite_ratio self.c«rossover»_rate = crossover_rate`
- **f40515** AUROC 0.987 (val), active 1.00 / 0.34; max tokens `'crossover'` ×21, `'_rate'` ×20, `''` ×17
  - 18.7: `20, mut_scale=0.5, cross_prob=0«.»7, grad_step`
  - 13.3: `ated) # Crossover (with current best) crossed =« x» + np.random.uniform`
  - 13.2: `=0.01, mutation_rate=0.1, crossover«_rate»=0.8`
  - 13.2: `* (pop[b] - pop[c]) # Crossover « » cross = np.random`
  - 13.1: `self.mutation_rate = mutation_rate self.crossover«_rate» = crossover_rate`
- **f8157** AUROC 0.978 (val), active 1.00 / 0.05; max tokens `'rossover'` ×47, `'crossover'` ×27, `''` ×9
  - 5.3: `if self.budget <= 0: break # Select« parents» parent1`
  - 5.1: `# Crossover if np.random.rand() < self.c«rossover»_rate: mutant`
  - 5.1: `rossover mask = np.random.rand(self.dim) < self.c«rossover»_rate offspring`
  - 4.9: `rossover_rate: parent1 = np.random.choice(self.population) « parent»2 = np.random`
  - 4.9: `in range(self.population_size - len(elite)): # Select two« parents» parent1`

## L9 · particle_swarm

- **f43690** AUROC 0.975 (val), active 0.99 / 0.89; max tokens `''` ×22, `'.velocity'` ×18, `'='` ×10
  - 8.0: `: momentum-based update self.velocity = self.momentum * self«.velocity» + (self.best`
  - 7.9: `# Add momentum self.velocity = self.momentum * self«.velocity» + perturbation`
  - 7.9: `self.dim = dim self.learning_rate = learning_rate« » self.momentum`
  - 7.8: `with momentum self.x_opt += self.grad_learning_rate * self«.velocity» self.x`
  - 7.8: `x = x + self.velocity # Exploration phase with probability exploration«_rate» if np`
- **f15090** AUROC 0.995 (val), active 1.00 / 0.21; max tokens `'='` ×38, `')'` ×21, `'.velocity'` ×19
  - 24.0: `= func(new_x) # Update velocity and position self«.v» = self.alpha *`
  - 23.9: `new_f self.x_opt = new_x self.velocity« =» np.zeros(self.dim`
  - 23.8: `self.x_opt = x # Initialize momentum velocity« =» np.zeros(n)`
  - 23.7: `self.best_x = x # Initialize momentum « velocity» = np.zeros(self`
  - 23.6: `init_step self.rand_step_size = sigma self.velocity« =» np.zeros(dim)`
- **f6424** AUROC 0.984 (val), active 0.97 / 0.00; max tokens `'.vel'` ×47, `'ocities'` ×25, `'velocity'` ×10
  - 8.9: `] + cognitive + social new_pos = pop[i] +« vel» # Boundary`
  - 8.6: `.pop[j]) self.pop[j] = self.pop[j] +« vel» self.pop`
  - 8.5: `np.eye(self.dim)) # Avoid singular matrix # Update« velocities» and positions`
  - 8.5: `1, r2 = np.random.rand(), np.random.rand() « vel» = self.inertia`
  - 8.4: `* (self.gbest - pop[j]) pop[j] +=« vel» # Apply`

## L9 · covariance

- **f19031** AUROC 0.984 (val), active 1.00 / 0.05; max tokens `'ov'` ×42, `'covariance'` ×27, `'ariance'` ×9
  - 12.7: `.penalty_coeff = penalty_coeff self.cov_update_freq =« cov»_update_freq`
  - 12.3: `a + self.F * (b - c) # Apply« covariance»-based mutation`
  - 12.1: `riors, axis=0) self.cov_matrix = np.c«ov»(self.priors,`
  - 12.0: `self.cov_matrix = np.eye(dim) # Initial« covariance» matrix def`
  - 12.0: `self.fitness[elite_indices] # Estimate covariance matrix (sample« covariance») self.c`
- **f10007** AUROC 0.970 (val), active 1.00 / 0.68; max tokens `'covariance'` ×14, `'ov'` ×11, `'ariance'` ×8
  - 24.1: `=0) deviations = pop - mean cov = np«.c»ov(deviations,`
  - 23.8: `self.x_opt = self.pop[best_idx] # Update« covariance» matrix using simple adaptive`
  - 23.5: `samples = np.random.multivariate_normal(self.x_opt, covariance,« self».pop_size)`
  - 23.5: `np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix«) » trial = pop`
  - 23.4: `Adaptive covariance matrix (simplified) self.C = np.eye(self«.dim») * self.init`
- **f50675** AUROC 0.982 (val), active 0.99 / 0.05; max tokens `'ov'` ×84, `'cov'` ×9, `'Cov'` ×2
  - 5.3: `05, penalty_coeff=1.0, « cov»_update_freq=5`
  - 4.4: `utation_strength, self.dim) mutation_vector = np.dot(self.c«ov»ariance_matrix, mutation`
  - 3.9: `np.sum((trial - self.mean) ** 2 / self.c«ov»_matrix, axis=`
  - 3.8: `rowvar=False) self.cov_matrix = np.maximum(self.c«ov»_matrix, 1`
  - 3.8: `_size self.decay_rate = decay_rate self.c«ov»_update_freq = cov`

## L9 · step_size_adaptation

- **f43396** AUROC 0.769 (val), active 0.94 / 0.39; max tokens `'*='` ×96
  - 2.0: `*= 0.95 else: self.step_size« *=» 1.0`
  - 2.0: `self.best_f - 1e-6: self.step_size« *=» self.beta`
  - 2.0: `# Decay gradient influence over time grad_decay_factors[j]« *=» self.grad_decay`
  - 2.0: `# Decay gradient influence over time grad_decay_factors[j]« *=» self.grad_decay`
  - 2.0: `# Decay gradient influence over time grad_decay_factors[j]« *=» self.grad_decay`
- **f14902** AUROC 0.767 (val), active 1.00 / 0.90; max tokens `'*='` ×30, `'self'` ×21, `''` ×16
  - 7.0: `5: self.radius *= self.expand_rate else: « self».radius *= self.sh`
  - 6.8: `(self.budget, 500) if remaining_budget <=« »0: return`
  - 6.5: `if best_f < f_center: current_radius *= self.expand_factor« » else:`
  - 6.5: `random search phase for _ in range(remaining_budget // «2»): x =`
  - 6.4: `, increase perturbation size to explore self.current_step_size *=« »1.1`
- **f34298** AUROC 0.769 (val), active 1.00 / 0.96; max tokens `'_decay'` ×15, `'('` ×12, `'9'` ×12
  - 13.7: `ise_scale perturbed_samples = np.tile(self.x_opt,« (»self.mu,`
  - 12.6: `=0.7, mutation_strength=0.1, budget«_decay»=0.9`
  - 12.1: `with momentum self.momentum_term = self.momentum_term *« »0.9 +`
  - 11.9: `# Reduce step size after exploration self.step_size *=« (»1 - lr)`
  - 11.9: `i) for i in range(num_scales)] self.learning_rate«_schedule» = [learning_rate`

## L9 · restart

- **f45657** AUROC 1.000 (val), active 1.00 / 0.01; max tokens `'starts'` ×95, `'restart'` ×1
  - 5.0: `__call__(self, func): for _ in range(self.re«starts»): # Initialize`
  - 4.8: `self.best_x = None for _ in range(self.re«starts»): # Initialize`
  - 4.8: `, 1e2)), alpha=self.alpha, n_re«starts»_optimizer=10`
  - 4.7: `e2)), alpha=self.sigma_noise**2, n_re«starts»_optimizer=10`
  - 4.7: `), alpha=self.alpha, n_restarts_optimizer=self.n_re«starts»_optimizer )`
- **f43276** AUROC 0.999 (val), active 1.00 / 0.02; max tokens `'starts'` ×94, `'decay'` ×1, `'valuate'` ×1
  - 7.9: `( kernel=self.kernel, alpha=self.alpha, n_re«starts»_optimizer=self.n_re`
  - 7.9: `= GaussianProcessRegressor(kernel=self.kernel, alpha=self.epsilon, n_re«starts»_optimizer=10`
  - 7.9: `.noise, alpha=self.noise**2, n_re«starts»_optimizer=10`
  - 7.8: `WhiteKernel(), alpha=1e-5, n_re«starts»_optimizer=10`
  - 7.8: `.noise_variance), alpha=self.noise_variance, n_re«starts»_optimizer=10`
- **f39275** AUROC 0.999 (val), active 1.00 / 0.01; max tokens `'restart'` ×53, `'.restart'` ×23, `'starts'` ×8
  - 18.2: `self.decay_rate = decay_rate self.restarts =« restart»s self`
  - 18.1: `ation and restart if np.std(self.fitness) < self«.restart»_threshold: self`
  - 18.0: `self.min_lr = min_lr self.restart_threshold =« restart»_threshold self`
  - 18.0: `if self.last_improvement > self.budget * self«.restart»_threshold: #`
  - 18.0: `self.local_iter = local_iter self.restart_temp =« restart»_temp self`

## L9 · local_search

- **f58976** AUROC 0.932 (val), active 0.88 / 0.08; max tokens `'_search'` ×77, `'search'` ×13, `'Search'` ×5
  - 2.7: `1.5, social_coeff=1.5, local«_search»_steps=30`
  - 2.7: `'''python import numpy as np class PSO_Gradient_Local«Search»: def __`
  - 2.7: `_update_freq = grad_update_freq self.local_search_steps = local«_search»_steps self`
  - 2.6: `_factor self.crossover_rate = crossover_rate self.local«_search»_radius = local_search`
  - 2.6: `, F=0.8, CR=0.9, local«_search»_rate=0.`
- **f13288** AUROC 0.935 (val), active 0.98 / 0.38; max tokens `'_search'` ×35, `'local'` ×9, `'search'` ×9
  - 6.7: `= dim self.population_size = population_size self.initial«_learning»_rate = initial_learning`
  - 6.6: `, 0.1, 1.0) # Local« optimization» within trust region`
  - 6.6: `step size else: # Local exploration around current point « » x = np.random`
  - 6.4: `= x + np.random.uniform(-self.local_search_radius, self.local«_search»_radius, self.dim`
  - 6.4: `python import numpy as np class AdaptiveDE_GuidedLocal«Search»: def __`
- **f46891** AUROC 0.936 (val), active 0.97 / 0.22; max tokens `'local'` ×45, `'.local'` ×21, `'_local'` ×21
  - 9.1: `= f_trial self.x_opt = trial # Perform« local» gradient refinement`
  - 8.9: `_scale *= self.mutation_decay self.local_search_scale *= self«.local»_search_decay`
  - 8.6: `inement(self, func): # Refine the best individual using a« local» search x`
  - 8.4: `.x_opt = x remaining_budget -= 1 #« Local» search around the best`
  - 8.4: `self.step_size = step_size self.num_local_steps = num«_local»_steps def`

## L9 · gradient

- **f39383** AUROC 0.933 (val), active 1.00 / 1.00; max tokens `'self'` ×13, `''` ×11, `'='` ×9
  - 18.7: `'central_diff': h = self.grad_step grad =« np».zeros(self.dim)`
  - 15.4: `= init_step_size self.step_decay = step_decay « self».min_step_size =`
  - 15.1: `# Initialize population self.pop = np.random.uniform(-5.«0», 5.`
  - 15.0: `.linalg.norm(grad_mean) if grad_norm > 1e-«6»: grad_mean`
  - 14.9: `f_x = self.f_pop[idx] # Compute the gradient using« finite» differences grad`
- **f8260** AUROC 0.952 (val), active 0.98 / 0.35; max tokens `''` ×9, `','` ×8, `'_minus'` ×6
  - 8.7: `Numerical gradient estimation using central difference grad = np.zeros_like«(x») eps =`
  - 8.5: `.x_opt def _estimate_gradient(self, x, func): « » # Central difference gradient`
  - 8.5: `x_plus_ij[j] += self.step_size x«_minus»_i = np.copy`
  - 8.1: `grad[i] = (func(x_plus) - func(x«_minus»)) / (2`
  - 8.1: `func(x_plus) - func(x_minus)) / (2 * self«.step»_size) #`
- **f10522** AUROC 0.901 (val), active 1.00 / 0.37; max tokens `'grad'` ×29, `'gradient'` ×19, `'.grad'` ×15
  - 9.3: `5): # Perform 5 gradient steps grad = np«.gradient»(func, x,`
  - 9.3: `_idx].copy() remaining -= self.pop_size # Local« gradient»-based refinement`
  - 9.2: `]] = trial fitness[parent_indices[0]] = trial_fitness«_grad» # Update`
  - 9.2: `, self.pop[j], edge_order=2) # Approximate« gradient» self.pop`
  - 9.1: `# Use gradient information to refine the best solution grad = np«.gradient»(func, self.x`

## L9 · elitism

- **f50288** AUROC 0.999 (val), active 1.00 / 0.00; max tokens `'elite'` ×46, `'ite'` ×33, `'elites'` ×7
  - 27.1: `_rate self.mutation_rate = mutation_rate self.el«ite»_ratio = elite_ratio`
  - 26.9: `self.population = self.population[:self.pop_size] # Elit«ism»: keep the best`
  - 26.9: `_size self.shrink_factor = shrink_factor self.el«ite»_ratio = elite_ratio`
  - 26.8: `selected = population[indices[:int(self.pop_size * self.el«ite»_ratio)]]`
  - 26.6: `self.mutation_rate = mutation_rate self.elite_ratio =« elite»_ratio self`
- **f10240** AUROC 0.999 (val), active 1.00 / 0.00; max tokens `'elite'` ×44, `'ite'` ×36, `'elites'` ×9
  - 11.0: `_rate self.mutation_rate = mutation_rate self.el«ite»_ratio = elite_ratio`
  - 11.0: `self.learning_rate = learning_rate self.elite_ratio =« elite»_ratio self`
  - 10.9: `_rate self.crossover_rate = crossover_rate self.el«ite»_ratio = elite_ratio`
  - 10.9: `self.pop[elite_indices] elite_f = self.f_values[«elite»_indices] #`
  - 10.9: `= np.inf self.elite_size = int(pop_size *« elite»_ratio) def`
- **f63341** AUROC 0.983 (val), active 0.97 / 0.01; max tokens `'ite'` ×32, `'.el'` ×23, `'elite'` ×18
  - 16.2: `keep the best individual if np.random.rand() < self.el«ite»_ratio: new`
  - 15.9: `_inertia self.mutation_rate = mutation_rate self«.el»ite_ratio = elite`
  - 15.8: `self.sigma_increase = sigma_increase self.num_elites = num«_el»ites self`
  - 15.6: `ub) new_pop.append(new_x) else: #« El»itism and gradient`
  - 15.2: `_f = f_vals[elite_indices] # Compute gradient estimate from« elites» self.grad`

## L9 · surrogate

- **f42** AUROC 1.000 (val), active 1.00 / 1.00; max tokens `'.dim'` ×4, `'_size'` ×4, `'f'` ×4
  - 16.7: `acquisition_type self.initial_samples = initial_samples self.no«ise»_var = noise_var`
  - 16.7: `x = np.clip(x, -5.0, 5«.»0) new`
  - 16.7: `scipy.stats import norm from sklearn.gaussian_process import GaussianProcessRegressor« »from sklearn.gaussian`
  - 16.6: `copy() # Train surrogate model self.surrogate_model«.fit»(self.X_samples,`
  - 16.5: `.stats import norm from sklearn.gaussian_process import GaussianProcessRegressor «from» sklearn.gaussian_process`
- **f43353** AUROC 0.917 (val), active 1.00 / 1.00; max tokens `'.g'` ×13, `'p'` ×4, `'next'` ×4
  - 10.1: `_kernel self.gp_alpha = gp_alpha self.g«p»_n_restarts =`
  - 9.3: `_population_fitness = np.vstack([new_population_fitness, [next_x,« next»_f]]) self`
  - 9.1: `# Code: '''python import numpy as np from sklearn.g«aussian»_process import GaussianProcess`
  - 9.1: `.y_samples = self.f_pop for i in range(self.b«udget»): # Evaluate`
  - 8.9: `solutions if step < self.initial_samples: # Initial random« sampling» new_x`
- **f10559** AUROC 0.972 (val), active 1.00 / 1.00; max tokens `''` ×22, `'='` ×8, `','` ×7
  - 3.2: `, n_restarts_optimizer=10) self.f_opt =« np».inf self`
  - 3.2: `, kernel=RBF(length_scale=1.0), noise_var«=»0.01`
  - 3.2: `exploitation. # Code: '''python import numpy as np from« sklearn».gaussian_process import`
  - 3.2: `aussian_process import GaussianProcessRegressor from sklearn.gaussian_process.kernels« import» RBF, Constant`
  - 3.1: `def __call__(self, func): bounds = func.bounds « » population = np.random`

## L9 · adaptive_parameters

- **f22304** AUROC 0.930 (val), active 0.94 / 0.13; max tokens `'AD'` ×30, `'.ad'` ×19, `'_ad'` ×15
  - 12.5: `0.99): """ Adaptive Directional Harmony Search («AD»HS) algorithm with`
  - 12.4: `Initialize momentum v = np.zeros(self.dim) # Perform« AD»AM-like optimization with`
  - 9.3: `Code: '''python import numpy as np class HybridDEGradient«AD»: def __`
  - 9.0: `) # Code: '''python import numpy as np class« AD»ER: def`
  - 7.8: `_rate self.local_update_freq = local_update_freq self«.ad»apt_rate = adapt`
- **f26471** AUROC 0.906 (val), active 1.00 / 0.96; max tokens `''` ×22, `'='` ×20, `'=True'` ×9
  - 4.4: `.2, mutation_scale=0.5, budget_decay«=»0.98`
  - 4.4: `self.restart_interval = restart_interval self.dynamic_inertia« =» dynamic_inertia`
  - 4.3: `utation_factor = mutation_factor self.memory_conservation = memory_cons«ervation» self.dynamic`
  - 4.2: `= F self.CR = CR self.adaptive =« adaptive» self.ex`
  - 4.2: `.linalg.solve(self.hessian, grad) if self.use_hessian_approx«imation» else x -`
- **f10414** AUROC 0.900 (val), active 0.95 / 0.58; max tokens `':'` ×33, `''` ×29, `'='` ×6
  - 8.0: `adjust perturbation rate based on progress if self.dynamic_learning: « » if self.best_fitness`
  - 8.0: `_vector is not None else np.random.rand(dim) self.use_gradient« =» use_gradient`
  - 7.9: `ratio if np.random.rand() < self.exploration_ratio«: » self.sigma *=`
  - 7.9: `=0.1, pop_size=50, multi_objective«=False»): self.b`
  - 7.9: `# Update ensemble of mutation vectors if self.use_ensemble: « » if len(self.`

## L15 · population

- **f31613** AUROC 0.978 (val), active 0.96 / 0.01; max tokens `'.pop'` ×44, `'pop'` ×26, `'_pop'` ×14
  - 16.6: `udget: # Select parents idx = np.argsort(self.f«_pop») parents =`
  - 16.6: `# Apply bounds self.pop[j] = np.clip(self«.pop»[j], -5`
  - 16.5: `, func.bounds.ub) f = func(x) self«.pop».append((x,`
  - 16.5: `.dim) trial = self.pop[i] + phi * (self«.pop»[i] - self`
  - 16.5: `self.best_f = f_sorted[0] self.best_x =« pop»_sorted[0].`
- **f39782** AUROC 0.968 (val), active 0.96 / 0.05; max tokens `'.pop'` ×37, `'population'` ×22, `'pop'` ×11
  - 9.9: `self.grad_freq = grad_freq self.batch_size =« batch»_size self`
  - 8.9: `-6, momentum=0.9, batch_size=«1»0, noise_level`
  - 7.1: `.ub, (self.pop_size, self.dim)) self.f«_pop» = np.array([`
  - 6.9: `.dim) for _ in range(10): # Small« mini»-batch for gradient`
  - 6.9: `np.random.multivariate_normal(self.mean, self.cov_matrix, self«.pop»size) samples`
- **f5266** AUROC 0.968 (val), active 0.96 / 0.02; max tokens `'='` ×17, `'population'` ×15, `'_size'` ×13
  - 20.7: `, func.bounds.ub, (self.population_size, self.dim)) « » fitness = np.array`
  - 20.4: `self.x_opt = None # Initial population self.pop« =» np.random.uniform(l`
  - 20.3: `self.f_opt = np.min(fitness) self.x_opt =« population»[np.argmin(f`
  - 20.3: `call__(self, func): # Initialize population self.population« =» np.random.uniform(func`
  - 20.1: `)) f_pop = np.array([func(x) for x in« pop»]) # Best`

## L15 · differential_evolution

- **f50221** AUROC 0.999 (val), active 1.00 / 0.42; max tokens `'mutant'` ×69, `'ant'` ×20, `'descent'` ×3
  - 6.2: `eigenvectors = np.linalg.eig(self.curvature_matrix) « dominant»_eigenvalue =`
  - 6.1: `+ self.mutation_strength * np.random.normal(size=self.dim) « mutant» = np.clip(mut`
  - 6.0: `self.eta_g * np.outer(grad, grad) # Compute« dominant» gradient direction`
  - 6.0: `= a + self.mutation_factor * (b - c) « mutant» = np.clip(mut`
  - 5.9: `np.random.choice(self.pop_size, 3, replace=False) « mutant» = sorted_pop[idx`
- **f26008** AUROC 0.988 (val), active 0.99 / 0.03; max tokens `'mutant'` ×54, `'gen'` ×9, `'Gen'` ×7
  - 14.4: `# Code: '''python import numpy as np class Adaptive«Gen»eticOptimization:`
  - 13.3: `# Code: '''python import numpy as np class Hybrid«Gen»eticGradientOptimizer:`
  - 13.0: `# Code: '''python import numpy as np class HybridGen«etic»Algorithm: def`
  - 12.9: `.mu_eff) - np.sqrt(np.sum((self.f_pop - np«.min»(self.f_pop))`
  - 12.8: `) a, b, c = self.pop[idxs] « mutant» = a + self`
- **f30546** AUROC 0.974 (val), active 1.00 / 0.58; max tokens `''` ×22, `'='` ×7, `'self'` ×5
  - 28.5: `(self.dim) < self.crossover_rate, mutant, parent1) « » # Evaluate trial`
  - 27.8: `.argmin(fitness)] for i in range(self.budget): « » # Create new population`
  - 27.6: `step in range(self.budget): # Genetic operations #« Select» parents parents`
  - 27.6: `pop_size self.f = f self.cr = cr« » self.mu =`
  - 27.6: `, self.dim) # Crossover mask = np.random«.rand»(self.dim) <`

## L15 · crossover

- **f30546** AUROC 0.984 (val), active 1.00 / 0.60; max tokens `''` ×22, `'='` ×7, `'self'` ×5
  - 28.5: `(self.dim) < self.crossover_rate, mutant, parent1) « » # Evaluate trial`
  - 27.8: `.argmin(fitness)] for i in range(self.budget): « » # Create new population`
  - 27.6: `step in range(self.budget): # Genetic operations #« Select» parents parents`
  - 27.6: `pop_size self.f = f self.cr = cr« » self.mu =`
  - 27.6: `, self.dim) # Crossover mask = np.random«.rand»(self.dim) <`
- **f4142** AUROC 0.982 (val), active 1.00 / 0.04; max tokens `'rossover'` ×60, `'crossover'` ×31, `'.cross'` ×1
  - 26.6: `k in range(self.dim): if np.random.rand() < self«.cross»_prob: trial`
  - 24.8: `pop[b] - pop[c]) # Crossover cross« =» np.random.rand(self`
  - 22.9: `0.1) if self.adapt_cross: # Adjust« crossover» rate based on exploration`
  - 22.8: `(population) x_cross = np.concatenate((x_mut[:c«rossover»_point], parent[c`
  - 22.1: `crossed = np.where(np.random.rand(self.dim) < self.c«rossover»_rate,`
- **f15396** AUROC 0.968 (val), active 1.00 / 0.07; max tokens `'crossover'` ×45, `'rossover'` ×43, `'up'` ×7
  - 10.8: `_steps: return self.step_size * (t / self.warm«up»_steps) else`
  - 10.6: `_sigma self.max_sigma = max_sigma self.warm«up»_steps = warmup`
  - 10.4: `.cos(np.pi * t / (self.budget - self.warm«up»_steps))) elif`
  - 10.3: `Number of steps with no improvement to trigger shrinkage. - warm«up»_steps: Number of`
  - 8.5: `_factor = budget_shift_factor self.warmup_steps = warm«up»_steps self`

## L15 · particle_swarm

- **f19723** AUROC 0.982 (val), active 0.97 / 0.06; max tokens `'.velocity'` ×41, `'='` ×29, `')'` ×12
  - 26.0: `_opt = np.inf self.x_opt = None self«.v» = np.zeros(self`
  - 25.9: `init_step self.rand_step_size = sigma self.velocity« =» np.zeros(dim)`
  - 25.9: `np.random.normal(0, 1, self.dim) self«.velocity» *= self.mutation`
  - 25.8: `.decay_rate = decay_rate self.velocity = np.zeros(dim«) » def __call`
  - 25.8: `ise_scale = noise_scale self.velocity = np.zeros(dim) « » self.best_x =`
- **f4673** AUROC 0.978 (val), active 1.00 / 1.00; max tokens `''` ×10, `'self'` ×9, `'='` ×5
  - 16.1: `udget // 2)) self.velocities = np.random.uniform(-«1».0,`
  - 16.0: `2 = np.random.rand(self.dim), np.random.rand(self.dim) « » velocity = (self`
  - 15.3: `_velocities[j] = (self.inertia_weight * self.particle«_vel»ocities[j] +`
  - 15.3: `ocities[j] = (self.inertia_weight * self.velocities[j«]» + self.c`
  - 15.2: `udget - self.init_pop): # Estimate gradients for all points in« the» population grads`
- **f30578** AUROC 0.959 (val), active 0.99 / 0.12; max tokens `'.vel'` ×24, `'momentum'` ×19, `'velocity'` ×10
  - 6.7: `_opt + perturbation else: # Gradient-based update with« momentum» grad =`
  - 6.7: `numpy as np class MomentumAdaptiveSearch: def __init«__(»self, budget=`
  - 6.6: `) * delta_f # Update velocity with momentum self«.velocity» = self.momentum`
  - 6.5: `grad_step self.grad_freq = grad_freq self.m«omentum» = momentum`
  - 6.5: `f = func(x) history.append(f) #« Update» the best solution`

## L15 · covariance

- **f5776** AUROC 0.990 (val), active 1.00 / 0.17; max tokens `'np'` ×14, `''` ×13, `'.c'` ×9
  - 22.5: `self.population - mu cov_matrix = np.cov(deviations«,» rowvar=False)`
  - 22.2: `itness)] # Initialize covariance matrix and mean self.mean =« np».mean(self.pop,`
  - 22.2: `Code: '''python import numpy as np class MemoryCov«ariance»HybridAlgorithm:`
  - 22.1: `np.linalg.pinv(cov_matrix) # Regularized inverse « » # Generate offspring using`
  - 22.1: `) else: cov_matrix = np.eye(self.dim) *« »0.1`
- **f18918** AUROC 0.980 (val), active 1.00 / 0.05; max tokens `'ov'` ×35, `'ariance'` ×23, `'covariance'` ×23
  - 18.0: `self.sigma_inc = sigma_inc self.cov_matrix =« cov»_matrix if cov_matrix`
  - 17.8: `(grads.T, grads) / self.mu self.cov«ariance» = np.clip(self`
  - 17.7: `ans = np.mean(X, axis=0) self.cov«ariance» = np.cov`
  - 17.7: `self.x_opt def update_covariance(self): # Calculate« covariance» matrix from elite samples`
  - 17.6: `_lengthscale=1.0, noise_var=0.1,« covariance»_update_rate=0`
- **f51900** AUROC 0.976 (val), active 1.00 / 0.31; max tokens `'np'` ×12, `''` ×10, `'='` ×9
  - 7.6: `covariance matrix is positive definite eigenvalues, eigenvectors =« np».linalg.eigh(self`
  - 6.8: `self.direction_update_freq = direction_update_freq self.covariance«_update»_freq = covariance_update`
  - 6.7: `curvature magnitude as Frobenius norm of Hessian return np«.linalg».norm(hessian)`
  - 6.6: `_values.append(best_f) # Update the covariance matrix and sigma « » if self.adapt`
  - 6.5: `: # Decrease exploration by narrowing covariance self.cov«ariance» *= (1 -`

## L15 · step_size_adaptation

- **f3752** AUROC 0.719 (val), active 0.98 / 0.83; max tokens `'*='` ×38, `'+='` ×33, `'-='` ×14
  - 20.9: `* grad # Apply learning rate and decay self.velocity« *=» self.decay_rate`
  - 20.8: `# Decay momentum vector to prevent saturation self.momentum_vector« *=» self.momentum_decay`
  - 20.7: `traction_factor # Reset velocity if no improvement self.velocity« *=» 0.5`
  - 20.7: `momentum vector (optional, can be removed if desired) self.v« *=» self.decay_rate`
  - 20.6: `oration_rate * 1.1) self.exploration_counter« +=» 1`
- **f51418** AUROC 0.707 (val), active 0.88 / 0.58; max tokens `'size'` ×62, `'function'` ×4, `'vector'` ×3
  - 4.7: `step size adaptation self.beta = beta # Momentum for step« size» adaptation def`
  - 4.2: `self.step_size = 0.5 # Initial step« size» # Initial`
  - 4.2: `candidate_f = func(candidate_x) # Acceptance based on cooling« schedule» if candidate`
  - 4.1: `, func.bounds.lb, func.bounds.ub) # Adaptive step« size» based on improvement`
  - 4.1: `parents for differential evolution self.sigma = sigma # Initial step« size» for CMA-`
- **f65527** AUROC 0.735 (val), active 1.00 / 1.00; max tokens `''` ×14, `'self'` ×7, `'.'` ×5
  - 13.9: `# Perform random search with reduced step size x_rand =« self».x_opt + self`
  - 10.7: `: best_step = s best_f_step = f_test« » # Update`
  - 10.5: `* (self.expand_factor ** self.level) else: self«.level» = max(1`
  - 10.5: `.learning_rate *= self.decay_rate self.learning_rate_history.append«(self».learning_rate)`
  - 10.1: `erturbation_scale, self.perturbation_scale * 0.5«,» self.perturbation`

## L15 · restart

- **f21759** AUROC 1.000 (val), active 1.00 / 0.06; max tokens `'starts'` ×51, `'0'` ×14, `'='` ×9
  - 9.7: `ProcessRegressor(kernel=self.kernel, alpha=self.alpha, n_restarts_optimizer«=»10)`
  - 9.6: `0, dim=10, init_points=10, n«_re»starts_optimizer=2`
  - 9.5: `( kernel=self.kernel, alpha=self.alpha, n_re«starts»_optimizer=self.n_re`
  - 9.5: `0.1: # 10% chance of random« restart» x =`
  - 9.5: `1e2)), alpha=self.alpha, n_restarts_optimizer«=»10`
- **f3526** AUROC 0.999 (val), active 1.00 / 0.01; max tokens `'restart'` ×47, `'.restart'` ×23, `'Restart'` ×10
  - 11.7: `evals = self.pop_size # Track restarts « restart»s = 0`
  - 11.7: `5, adaptive_mutation=True, adaptive_crossover=True,« restart»_threshold=0.`
  - 11.7: `.1 else: if np.random.rand() < self«.restart»_threshold: self`
  - 11.7: `, adaptive_mutation=True, dynamic_crossover=True, « restart»_threshold=0.`
  - 11.6: `=0.5, elite_ratio=0.2,« restart»_threshold=0.`
- **f60537** AUROC 0.998 (val), active 1.00 / 0.02; max tokens `'restart'` ×48, `'.restart'` ×32, `'Restart'` ×12
  - 8.6: `self.mu = mu self.noise = noise self«.restart»_interval = restart_interval`
  - 8.2: `.1 else: if np.random.rand() < self«.restart»_threshold: self`
  - 8.2: `= grad_step self.min_lr = min_lr self«.restart»_threshold = restart_threshold`
  - 8.0: `_weight=0.5, decay_rate=0.99,« restart»_threshold=0.`
  - 8.0: `i > 0: if np.std(fitness) < self«.restart»_threshold: #`

## L15 · local_search

- **f31876** AUROC 0.953 (val), active 0.97 / 0.22; max tokens `'local'` ×51, `'.local'` ×26, `'Local'` ×7
  - 14.8: `self.mutation_decay = mutation_decay self.local_search_decay =« local»_search_decay`
  - 12.8: `= f_trial self.x_opt = trial # Perform« local» gradient refinement`
  - 12.8: `= grad_steps self.grad_eps = grad_eps self«.local»_search_prob = local`
  - 12.7: `self.x_opt + self.dir_scale * step_size * directions « local»_samples = np.clip`
  - 12.7: `2 self.neighborhood = neighborhood self.local_iter =« local»_iter def`
- **f43578** AUROC 0.857 (val), active 0.98 / 0.84; max tokens `'self'` ×11, `'best'` ×9, `'search'` ×8
  - 11.4: `0, num_onlooker=20, pheromone«_decay»=0.9`
  - 9.3: `to find the point that maximizes acquisition # This is a« heuristic» and could be replaced`
  - 9.1: `class HybridAdaptiveHarmonySearch: def __init__(«self», budget=1`
  - 9.0: `, 0.8) self.learning_rate = np.clip(self«.learning»_rate + np.random`
  - 9.0: `=15, hmcr=0.95,« par»=0.8`
- **f51490** AUROC 0.875 (val), active 0.97 / 0.22; max tokens `'local'` ×60, `'.local'` ×21, `'Local'` ×9
  - 13.9: `with random perturbations if np.random.rand() < self«.local»_search_rate:`
  - 13.7: `self.social_coeff = social_coeff self.local_search_rate =« local»_search_rate`
  - 13.3: `). trust_region_radius (float): Radius of the trust region for« local» search. grad`
  - 13.3: `local_search_points = np.random.uniform(gbest_position - self«.local»_search_radius,`
  - 13.2: `= cognitive_coeff self.social_coeff = social_coeff self«.local»_search_radius = local`

## L15 · gradient

- **f6720** AUROC 0.958 (val), active 1.00 / 0.91; max tokens `''` ×8, `'x'` ×7, `')'` ×6
  - 45.4: `[j] += eps f_plus = func(x_plus) « x»_minus = np.copy`
  - 43.4: `- self.learning_rate * grad # Perturb the solution « » for _ in range`
  - 43.2: `in range(self.dim): x_plus = np.copy(x) « x»_plus[i] +=`
  - 43.1: `func) if np.linalg.norm(grad) > 1e-«6»: new_x`
  - 42.5: `# Finite difference gradient estimator eps = 1e-7« » grad = np`
- **f22900** AUROC 0.925 (val), active 0.98 / 0.22; max tokens `'('` ×17, `''` ×8, `'*'` ×6
  - 8.1: `f_candidate grad[k] = (f_plus - f_minus«)» / (2 *`
  - 7.9: `self.f_values[i]: grad = np.gradient(self.f_values[i«],» self.population[i])`
  - 7.8: `change in step size) if len(self.curvature_history) >« »1: #`
  - 7.7: `np.array(self.history[-2][0])) / (self.step_size«) » # Compute approximate`
  - 7.6: `.gradient(self.f_history[-1], self.x_opt) direction =« grad» / np.linalg.norm`
- **f25747** AUROC 0.939 (val), active 0.98 / 0.50; max tokens `''` ×17, `'.dim'` ×15, `'np'` ×8
  - 11.0: `Estimate gradient using finite differences grad = np.zeros(self.dim) « » for d in range`
  - 11.0: `_minus) grad[d] = (f_plus - f_minus)« /» (2 * eps`
  - 11.0: `6 grad = np.zeros(self.dim) for d in« range»(self.dim):`
  - 10.9: `udget): # Estimate gradients for all points gradients = np«.zeros»((self.population_size`
  - 10.8: `in range(self.dim): perturb = np.zeros(self.dim) « » perturb[d]`

## L15 · elitism

- **f17108** AUROC 0.999 (val), active 1.00 / 0.00; max tokens `'elite'` ×50, `'ite'` ×28, `'elites'` ×10
  - 15.3: `elite_fitness = fitness[elite_indices] # Gradient estimation for« elites» using finite differences`
  - 15.3: `_rate self.elite_ratio = elite_ratio self.el«ite»_count = int(self`
  - 15.2: `_elites = self.pop[non_elite_indices] non_el«ite»_f = self.f`
  - 15.1: `=0.8, mutation_rate=0.1,« elite»_ratio=0.`
  - 15.0: `best individuals elite_count = int(self.pop_size * self.el«ite»_ratio) self`
- **f37819** AUROC 0.999 (val), active 1.00 / 0.00; max tokens `'elite'` ×59, `'ite'` ×26, `'elites'` ×9
  - 9.2: `* self.gbest_value self.particles = np.vstack((«elite»_positions, self.p`
  - 9.0: `.elite_count] elite_pop = population[elite_indices] « elite»_fitness = fitness[`
  - 9.0: `self.mutation_rate = mutation_rate self.elite_ratio =« elite»_ratio self`
  - 9.0: `.argsort(new_fitness)[:elite_size] elite_population = new_population[«elite»_indices] elite`
  - 8.9: `self.mutation_rate = mutation_rate self.elite_ratio =« elite»_ratio self`
- **f6700** AUROC 0.998 (val), active 1.00 / 0.00; max tokens `'elite'` ×51, `'ite'` ×34, `'Elite'` ×4
  - 10.2: `, 0.3, 1.0) self.de«cline»_count = 0`
  - 5.8: `=0.8, mutation_rate=0.1,« elite»_ratio=0.`
  - 5.7: `new_fitness # Optional: preserve elite if self.el«ite»_preservation and new`
  - 5.6: `.mutation_rate = mutation_rate self.elite_preservation =« elite»_preservation`
  - 5.6: `_size * 0.2)] elite_pop = self.X[«elite»_idx]`

## L15 · surrogate

- **f55511** AUROC 1.000 (val), active 1.00 / 1.00; max tokens `''` ×14, `'self'` ×11, `'='` ×8
  - 7.6: `gp.fit(self.X, self.y) self.gp_model =« gp» # Predict`
  - 7.3: `[np.argmin(self.Y), :] # Initialize Gaussian Process Reg«ress»or kernel`
  - 7.3: `np.array([func(x) for x in self.X]) self«.g»pr.fit(self.X`
  - 7.2: `self.X = None self.y = None « self».model = None`
  - 7.2: `= beta self.noise = noise self.kernel =« kernel» self.max`
- **f44640** AUROC 0.999 (val), active 1.00 / 0.13; max tokens `'_process'` ×20, `'model'` ×12, `'.fit'` ×11
  - 7.0: `search gradient = coeff[1:] # Gradient from linear« model» step =`
  - 5.9: `): # Use historical population and fitness to build a simple linear surrogate« model» X =`
  - 5.9: `(X)), X)) try: theta = np.linalg.lstsq«(X», y, r`
  - 5.7: `using surrogate predicted_fitness = np.array([self._predict_sur«rogate»(ind) for ind`
  - 5.6: `perturbation_decay=0.99, surrogate_model_size=«1»00, surrogate`
- **f38726** AUROC 1.000 (val), active 1.00 / 0.09; max tokens `''` ×12, `'.g'` ×7, `'__('` ×6
  - 8.8: `_rate self.elite_ratio = elite_ratio self.s«ur»rogate_kernel = surrogate`
  - 8.5: `self.y = initial_fitness self.gp = Gaussian«Process»Regressor(kernel=self.kernel`
  - 8.4: `.X_train) > 1: predicted_x = self.sur«rogate»_model.predict(np.array`
  - 8.4: `0, alpha=1e-5, exploration_decay=«0».995`
  - 8.3: `=RBF(), noise_var=1e-2, grad_tol=«1»e-5,`

## L15 · adaptive_parameters

- **f55215** AUROC 0.913 (val), active 1.00 / 0.88; max tokens `'ive'` ×30, `'Adaptive'` ×18, `'adaptive'` ×16
  - 14.0: `sklearn.gaussian_process.kernels import RBF, WhiteKernel class« Adaptive»CovarianceEv`
  - 13.3: `np.random.rand() < self.exploration_ratio: # Predict«ive» mean and variance`
  - 13.2: `func(x) for x in self.X]) self.model = mult«ivariate»_normal(mean=np.mean`
  - 13.2: `_evaluated, self.f_evaluated) # Update sigma using simple« adaptive» learning self`
  - 13.1: `optimization phase while remaining_budget > 0: # Predict«ive» mean and variance`
- **f21299** AUROC 0.891 (val), active 0.97 / 0.63; max tokens `''` ×23, `'='` ×14, `'.ad'` ×9
  - 7.8: `= local_radius self.learning_rate = learning_rate self«.ad»aptive = adaptive`
  - 7.6: `self.noise_level = noise_level self.adaptive«_ex»ploration = adaptive`
  - 7.5: `_remaining = budget self.elite_preservation = elite_preservation« » def __call`
  - 7.4: `self.adaptive_exploration = adaptive_exploration « » self.perturbation`
  - 7.4: `.3, self.exploration_weight - 0.02«) » # With some`
- **f14313** AUROC 0.897 (val), active 0.93 / 0.50; max tokens `''` ×17, `'='` ×11, `','` ×8
  - 9.9: `learning_rate=0.05, multi_objective=False«,» diversity_weight=0`
  - 9.8: `_factor = mu_factor # Learning rate for crossover rate self«.multi»_objective = multi`
  - 9.7: `self.pop_size = pop_size self.multi_objective« =» multi_objective`
  - 9.6: `# Adaptive mutation based on dimension scaling if self.dim_scaling«: » self.mutation`
  - 9.5: `not None else np.random.rand(dim) self.use_gradient = use«_gradient» self.grad`

## L21 · population

- **f42170** AUROC 0.960 (val), active 0.93 / 0.01; max tokens `'.pop'` ×78, `'pop'` ×8, `'_pop'` ×5
  - 23.2: `if len(self.gradient_history) >= 5: self.gradient_history«.pop»(0)`
  - 22.8: `itness[i] < self.best_fitness: self.best_solution = self«.pop»[i] self`
  - 22.7: `pcov += np.outer((pop - mean) / sigma, («pop» - mean) /`
  - 22.5: `# Estimate covariance matrix self.cov = np.cov(self«.pop»[self.weights <`
  - 22.5: `ivariate_normal(self.mean, self.covariance * self.sigma, self«.pop»_size) self`
- **f49803** AUROC 0.969 (val), active 0.96 / 0.07; max tokens `'population'` ×26, `'.population'` ×18, `'_size'` ×16
  - 39.1: `_opt = np.min(self.f_pop) self.x_opt = self«.pop»[np.argmin(self`
  - 38.3: `3, replace=False) a, b, c = self«.pop»[idxs]`
  - 38.2: `new_fitness = np.hstack((fitness, offspring_fitness)) new«_population» = new_population[np`
  - 38.0: `= budget self.dim = dim self.population_size =« population»_size self`
  - 37.8: `, n_restarts_optimizer=10 ) self«.pop» = None`
- **f52869** AUROC 0.969 (val), active 0.97 / 0.07; max tokens `'.pop'` ×20, `'population'` ×15, `'swarm'` ×13
  - 15.6: `np.random.uniform(func.bounds.lb, func.bounds.ub, (self«.pop»_size, self.dim`
  - 15.3: `of the problem. swarm_size (int): Size of the particle« swarm». inertia_weight`
  - 15.1: `= [] def __call__(self, func): # Initialize« swarm» for _`
  - 15.1: `, axis=0) self.cov = np.cov(self«.pop», rowvar=False`
  - 14.4: `(x, func.bounds.lb, func.bounds.ub) for x in« new»_population] #`

## L21 · differential_evolution

- **f1323** AUROC 0.998 (val), active 1.00 / 0.17; max tokens `'mutant'` ×77, `'ant'` ×17, `'dominant'` ×2
  - 13.8: `eigenvectors = np.linalg.eig(self.curvature_matrix) « dominant»_eigenvalue =`
  - 13.8: `self.eta_g * np.outer(grad, grad) # Compute« dominant» gradient direction`
  - 11.2: `# Update covariance matrix using diversity of the population diff =« mutant» - pop`
  - 11.2: `05, 0.5, 1.0) « mutant» = pop[a]`
  - 11.0: `(pop[b] - pop[c]) # Apply bounds « mutant» = np.clip(mut`
- **f28429** AUROC 0.990 (val), active 0.98 / 0.00; max tokens `'mutant'` ×96
  - 6.3: `, b, c = indices # Create mutant vector « mutant» = self.pop[a`
  - 6.3: `replace=False) if self.mutation_strategy == 'random': « mutant» = population[idxs`
  - 6.2: `range(self.budget): # Generate mutant vectors using DE « mutant»_pop = []`
  - 6.2: `.dim)) for i in range(self.budget): # Generate« mutant» vectors for`
  - 6.1: `=False) a, b, c = indices # Create« mutant» vector mutant`
- **f25508** AUROC 0.956 (val), active 0.94 / 0.16; max tokens `'three'` ×52, `'3'` ×34, `'_size'` ×6
  - 12.4: `parents parent_indices = np.random.choice(pop_size, size=«3», replace=False)`
  - 11.7: `for j in range(self.pop_size): # Tournament selection for« three» distinct individuals`
  - 11.1: `evolution for j in range(self.pop_size): # Select« three» distinct individuals`
  - 11.0: `individuals idxs = np.random.choice(self.pop_size, «3», replace=False)`
  - 10.9: `if j in elite_idx: continue # Select« three» distinct individuals`

## L21 · crossover

- **f24722** AUROC 0.987 (val), active 1.00 / 0.05; max tokens `'crossover'` ×48, `'rossover'` ×43, `'cross'` ×3
  - 39.4: `ut_scale * (pop[b] - pop[c]) # C«rossover» trial =`
  - 35.8: `= np.random.rand(self.pop_size, self.dim) < self.c«rossover»_rate trial`
  - 35.7: `self.step_size, self.dim) * self.mutation_rate « crossed» = np.clip(c`
  - 33.5: `[i]) # Combine gradient update and mutation candidate =« crossed» + gradient_update`
  - 33.1: `_min self.mutation_rate = mutation_rate self.c«rossover»_rate = crossover_rate`
- **f6499** AUROC 0.971 (val), active 1.00 / 0.20; max tokens `'rossover'` ×54, `'crossover'` ×31, `'offspring'` ×4
  - 11.9: `= np.random.uniform(bounds.lb, bounds.ub) # Apply« bandwidth» new_h`
  - 11.5: `_harmony[i] = np.random.uniform(func.bounds.lb, func«.bounds».ub)`
  - 10.5: `in range(self.dim): if np.random.rand() < self.c«rossover»_rate: trial`
  - 10.5: `F # Mutation factor self.CR = CR # C«rossover» probability self`
  - 10.4: `= a + self.F * (b - c) # C«rossover» trial =`
- **f14886** AUROC 0.969 (val), active 0.99 / 0.08; max tokens `'rossover'` ×19, `'crossover'` ×13, `'.'` ×10
  - 11.8: `# Code: '''python import numpy as np class HybridGen«etic»Algorithm: def`
  - 11.5: `np.sum(self.f_pop)) selected = self.pop[selected] « offspring» = np.random.choice`
  - 11.1: `# Crossover if np.random.rand() < self.crossover«_rate»: child =`
  - 11.0: `range(self.pop_size - elite_size): # Select parents « parent»1_idx = np`
  - 10.7: `= [] for i in range(self.pop_size - elite_count): « » # Select parents`

## L21 · particle_swarm

- **f21412** AUROC 0.984 (val), active 0.97 / 0.09; max tokens `'.velocity'` ×59, `'velocity'` ×14, `'='` ×13
  - 30.8: `_opt = None self.v = np.zeros(dim) #« Velocity» vector self`
  - 30.5: `= noise_scale self.step = init_step self.velocity« =» np.zeros(dim)`
  - 30.4: `x_new = self.x_opt - self.step_size * self«.velocity» else:`
  - 30.3: `self.dim_weights += self.learning_rate * improvement * np.sign(self«.velocity») # Decay`
  - 30.2: `: momentum-based update self.velocity = self.momentum * self«.velocity» + (self.best`
- **f3901** AUROC 0.955 (val), active 0.99 / 0.40; max tokens `'))'` ×10, `'.vel'` ×10, `'velocities'` ×10
  - 15.8: `(func.bounds.lb, func.bounds.ub, (self.pop_size,« self».dim)) self`
  - 15.0: `_coeff * r2 * (self.gbest_position - self.particles«)) » self.particles`
  - 14.9: `= np.zeros((self.dim,)) # personal best « self».gbest = np`
  - 14.9: `) for _ in range(self.swarm_size)] self.velocities« =» [np.random.uniform`
  - 14.9: `_prob self.population = None self.velocities = None« » self.best =`
- **f6565** AUROC 0.962 (val), active 0.95 / 0.14; max tokens `'.velocity'` ×32, `'velocity'` ×26, `'_vel'` ×14
  - 26.9: `udget): for j in range(self.pop_size): # Update« velocity» r1`
  - 11.9: `_opt = np.inf self.x_opt = None self«.velocity» = np.zeros(dim`
  - 11.7: `learning self.velocity = np.zeros(dim) # Momentum-based« velocity» def __`
  - 10.7: `np.random.rand(dim) # Initial search direction self.vel«ocities» = np.random.rand`
  - 10.2: `self.grad_step_size) # Update velocity with momentum self«.velocity» = self.momentum`

## L21 · covariance

- **f13556** AUROC 0.984 (val), active 1.00 / 0.06; max tokens `'ov'` ×49, `'covariance'` ×19, `'cov'` ×14
  - 23.6: `loop for gen in range(self.budget): # Co«ev»olutionary learning phase`
  - 21.3: `self.learning_rate *= self.decay_rate # Co«ev»olutionary directional update`
  - 21.0: `, func): """ Optimize the given function using the Adaptive« Co»evolutionary Genetic`
  - 20.9: `# Code: '''python import numpy as np class AdaptiveC«oe»volutionaryGradientDE`
  - 20.2: `urbation_ratio = perturbation_ratio self.coevolution_rate =« co»evolution_rate`
- **f8192** AUROC 0.928 (val), active 0.90 / 0.16; max tokens `''` ×18, `'np'` ×10, `'='` ×7
  - 19.3: `samples - mean_sample self.cov = np.cov(dev«iations», rowvar=False`
  - 18.9: `_opt = pop[j] # Update covariance matrix mean =« np».mean(pop, axis`
  - 18.3: `from the mean deviations = pop - self.mean #« Compute» covariance matrix`
  - 17.6: `_size self.sigma = np.sqrt((self.sigma**2 *« (»self.pop_size -`
  - 17.5: `- np.mean(self.population, axis=0) cov_matrix =« np».cov(deviations`
- **f31308** AUROC 0.977 (val), active 1.00 / 0.05; max tokens `'ov'` ×67, `'covariance'` ×21, `'cov'` ×4
  - 15.8: `# Update covariance matrix self.covariance_matrix *= self.c«ov»ariance_decay`
  - 15.8: `parent1) * self.covariance_learning_rate self.c«ov»ariance_matrix += update`
  - 15.7: `_opt + np.random.multivariate_normal(np.zeros(self.dim), self.c«ov»ariance) * self`
  - 15.2: `_update_rate: self.covariance = (1 - self.c«ov»ariance_update_rate)`
  - 15.1: `0) centered_population = self.population - mean self.c«ov»ariance = (center`

## L21 · step_size_adaptation

- **f23794** AUROC 0.750 (val), active 0.97 / 0.60; max tokens `'*='` ×82, `'*'` ×8, `''` ×2
  - 10.2: `candidate += self.step_size * np.random.randn(self.dim) candidate« *=» np.random.uniform(`
  - 8.4: `is enabled if self.adaptive_learning: self.learning_rate« *=» self.learning_rate_decay`
  - 8.4: `_factor *= self.mutation_increase else: self.mutation«_factor» *= self.mutation`
  - 8.3: `_diff = np.abs(grad - self.velocity) self.curvature_estimate« *=» self.curvature_decay`
  - 8.3: `_opt = x remaining_budget -= 1 learning_rate« *=» self.decay_rate`
- **f58699** AUROC 0.712 (val), active 0.98 / 0.75; max tokens `'*='` ×23, `'9'` ×20, `'.'` ×10
  - 24.0: `# Combine momentum with random exploration self.velocity *= self.dec«ay»_rate self`
  - 23.1: `Generate a new candidate point using momentum self.velocity *= self.dec«ay»_rate self`
  - 23.1: `_noise: self.noise_scale = max(self.noise_scale *« (»1 - 1`
  - 22.8: `_momentum # Decay trust region radius self.current_radius« *=» self.radius_decay`
  - 22.4: `, pop_size=20, decay_rate=0.«9»9, mutation_rate`
- **f10348** AUROC 0.716 (val), active 0.72 / 0.31; max tokens `'*='` ×86, `'.multiply'` ×5, `'_multiplier'` ×3
  - 21.8: `, (personal_best[j] - swarm[j])) social = np«.multiply»(r2, (`
  - 21.6: `[j] = np.multiply(self.inertia, velocities[j]) + np«.multiply»(self.pso_weight`
  - 21.6: `, (personal_best[j] - swarm[j])) social = np«.multiply»(r2, (`
  - 21.5: `ias[s][j], velocities[s][j]) + \ np«.multiply»(self.pso_weight`
  - 21.0: `.rand(self.dim), np.random.rand(self.dim) cognitive = np«.multiply»(r1, (`

## L21 · restart

- **f43600** AUROC 1.000 (val), active 1.00 / 0.00; max tokens `'restart'` ×46, `'.restart'` ×23, `'Restart'` ×12
  - 25.9: `ation and restart if np.std(self.fitness) < self«.restart»_threshold: self`
  - 25.6: `self.local_iter = local_iter self.restart_temp =« restart»_temp self`
  - 25.6: `.bounds.ub, self.dim) new_temp = temperature * self«.restart»_temp_ratio`
  - 25.4: `self.decay_rate = decay_rate self.restarts =« restart»s self`
  - 25.3: `else: self.learning_rate *= self.lr_decay # Early« restart» if no improvement for`
- **f55795** AUROC 0.995 (val), active 1.00 / 0.06; max tokens `'starts'` ×71, `'enerate'` ×7, `'balance'` ×3
  - 15.8: `(diversity) < self.diversity_threshold: # Re«int»roduce diversity by pert`
  - 12.9: `0 # Reset step count if new best found #« Rein»force direction`
  - 12.8: `np.std(self.f) > self.rebalance_threshold: # Re«balance» by adding random points`
  - 12.7: `() if diversity < self.diversity_threshold: # Reg«enerate» population to maintain diversity`
  - 12.5: `]) if diversity < self.diversity_threshold: # Reg«enerate» population to maintain diversity`
- **f28450** AUROC 0.911 (val), active 1.00 / 1.00; max tokens `'.random'` ×11, `')'` ×9, `'range'` ×9
  - 24.6: `.budget = budget self.dim = dim self.learning«_rate» = learning_rate`
  - 24.1: `.choice(self.pop_size, int(self.pop_size * self.exploration«_ratio»), replace=False)`
  - 23.9: `Update exploration weight self.exploration_weight *= self.decay«_rate» # Update`
  - 23.4: `(self.swarm_size): r1, r2 = np.random«.rand»(self.dim), np`
  - 23.3: `.rand() < self.exploration_rate: exploration = np.random«.randn»(self.dim) *`

## L21 · local_search

- **f52337** AUROC 0.890 (val), active 0.97 / 0.22; max tokens `'Local'` ×37, `'local'` ×28, `'.local'` ×20
  - 35.8: `- pop[j]) elif strategy_idx == 1: #« Local» best strategy`
  - 35.2: `). trust_region_radius (float): Radius of the trust region for« local» search. grad`
  - 35.1: `self.f_opt = f_local self.x_opt = x«_local» # Early`
  - 35.0: `# Perform local gradient descent for _ in range(self.num«_local»_steps): if`
  - 34.8: `aussian_process.kernels import RBF, WhiteKernel class BayesianDE«Local»SearchOpt:`
- **f31918** AUROC 0.812 (val), active 1.00 / 1.00; max tokens `'='` ×15, `'_size'` ×11, `'_freq'` ×7
  - 14.4: `[i]) for j in range(self.dim): if np.random«.rand»() < self.c`
  - 13.3: `.f_opt: self.f_opt = x_f self.x«_opt» = x`
  - 13.2: `_size self.init_step = init_step self.decay«_rate» = decay_rate`
  - 13.2: `self.x_opt.copy() for _ in range(self.grad_refinement«_steps»): # Check`
  - 13.1: `_exploration = adaptive_exploration self.multi_step_ref«inement» = multi_step_ref`
- **f6404** AUROC 0.804 (val), active 1.00 / 0.95; max tokens `'.local'` ×13, `'.dim'` ×7, `'.f'` ×6
  - 16.8: `_learning_rate self.refine_rate = refine_rate self«.step»_decay_rate = step`
  - 16.0: `.99, adaptive_factor=0.05, « self»_adapt_rate=`
  - 16.0: `=0.2, noise_level=0.1, « self»_organization_rate=0`
  - 15.0: `.lb, func.bounds.ub) f_perturbed = func«(x»_perturbed)`
  - 14.7: `local refinement on elite solutions if np.random.rand() < self«.local»_refinement_rate:`

## L21 · gradient

- **f23754** AUROC 0.942 (val), active 0.99 / 0.41; max tokens `''` ×23, `'eps'` ×9, `'func'` ×5
  - 14.0: `numpy as np class HybridRandomGradientSearch: def __init«__(»self, budget=`
  - 13.9: `x_plus = np.copy(x) x_plus[i] += eps« » x_minus =`
  - 13.5: `grad[i] = (func(x_plus) - func(x«_minus»)) / (2`
  - 13.4: `def _update_gradient(self, func, x, grad_info): « eps» = 1e`
  - 13.4: `[j] = (func(x_plus) - func(x_minus)) /« (»2 * eps)`
- **f63055** AUROC 0.942 (val), active 0.98 / 0.24; max tokens `''` ×16, `'('` ×5, `'-'` ×4
  - 18.0: `] = (func(x_plus) - func(x_minus)) / («2» * eps)`
  - 17.2: `Numerical gradient estimation using central difference eps = 1e«-»7 grad`
  - 17.2: `grad[i] = (func(x_plus) - func(x_minus«))» / (2 *`
  - 16.9: `step grad = np.gradient(func, current_x, method='«central»', axis=0`
  - 16.9: `x_minus = x.copy() x_minus[i] -= h« » grad[i]`
- **f15727** AUROC 0.907 (val), active 1.00 / 0.41; max tokens `'grad'` ×25, `'gradient'` ×22, `'Gradient'` ×14
  - 15.3: `995, perturbation_scale=0.1,« grad»_tolerance=1e`
  - 15.0: `gradient clipping grad_norm = np.linalg.norm(grad) if« grad»_norm > self.grad`
  - 14.8: `step_decay (float): Decay rate for step size over iterations. « grad»_tolerance (float):`
  - 14.7: `gradient information try: grad = func.gradient(self.x_opt«)» # Assume this`
  - 14.6: `learning rate for gradient descent. momentum (float): Momentum coefficient for« gradient» descent. exploration`

## L21 · elitism

- **f56831** AUROC 0.960 (val), active 0.93 / 0.05; max tokens `'.el'` ×82, `'(el'` ×5, `'elites'` ×4
  - 18.9: `= decay_rate self.init_samples = init_samples self«.el»ite_ratio = elite`
  - 18.8: `_opt = np.inf self.x_opt = None self«.el»ite_memory = []`
  - 18.7: `= exploration_prob self.history_length = history_length self«.el»ite_ratio = elite`
  - 18.5: `_freq self.exploration_ratio = exploration_ratio self«.el»ite_ratio = elite`
  - 18.3: `# Elite selection elite_size = int(self.pop_size * self«.el»ite_ratio)`
- **f58435** AUROC 0.998 (val), active 1.00 / 0.07; max tokens `'elite'` ×40, `'ite'` ×37, `'elites'` ×8
  - 14.5: `, beta=0.95, memory_size=5,« elite»_decay=0.`
  - 14.1: `, adaptive_exploration_rate=0.95,« elite»_preservation=True,`
  - 14.1: `if self.elite_preservation: if len(self.el«ite»_pop) < self`
  - 14.0: `gradient steps elite_samples = int(remaining_budget * self.el«ite»_ratio) random`
  - 13.7: `=0.3, grad_update_freq=10, « elite»_preservation=0`
- **f21842** AUROC 0.999 (val), active 1.00 / 0.01; max tokens `'ite'` ×42, `'elite'` ×40, `'elites'` ×10
  - 13.5: `_fitness[np.argsort(self.elite_fitness)[:int(pop_size * self.el«ite»_ratio)]]`
  - 13.3: `solution best_history = [self.f_opt] # Track« elites» for preservation`
  - 13.1: `-= 1 continue # Select elite individuals « elite»_idx = np.argsort`
  - 12.9: `'''python import numpy as np class AdaptiveDifferentialEvolution«Elite»: def __`
  - 12.8: `= np.inf self.x_opt = None self.el«ites» = [] def`

## L21 · surrogate

- **f25145** AUROC 1.000 (val), active 1.00 / 1.00; max tokens `''` ×20, `','` ×10, `'_model'` ×7
  - 18.8: `expected_improvement(x): mean, std = self.gp«_model».predict(x.reshape(`
  - 18.7: `(mu - self.f_opt) * np.exp(-0.5« *» (mu - self`
  - 18.4: `, Y, rcond=None)[0] return lambda x:« np».dot(x, weights`
  - 17.3: `mu, sigma = self.gpr.predict(x.reshape(1, -«1»), return_std=True`
  - 17.3: `import numpy as np from sklearn.gaussian_process import GaussianProcess«Regressor» from sklearn.g`
- **f2037** AUROC 0.999 (val), active 1.00 / 0.14; max tokens `'.g'` ×13, `'Process'` ×7, `'rogate'` ×7
  - 11.7: `self.social_coeff = social_coeff self.gp_kernel =« gp»_kernel self`
  - 11.6: `ind in self.bayesian_candidates]) # Update Gaussian process« model» self.X`
  - 11.5: `. # Code: '''python import numpy as np from sklearn«.g»aussian_process import Gaussian`
  - 10.7: `self.x_opt = population[best_idx] # Initialize Gaussian« Process» model self`
  - 10.6: `5.0) # Crossover trial = np.copy«(self».pop[j])`
- **f23970** AUROC 0.999 (val), active 1.00 / 0.24; max tokens `'Process'` ×6, `'noise'` ×5, `'Kernel'` ×5
  - 15.7: `_steps self.restart_freq = restart_freq self.sigma_m«ixture» = sigma_mixture`
  - 12.3: `Regressor( kernel=RBF(length_scale=kernel_lengthscale) +« White»Kernel(noise_level`
  - 12.2: `.gp_kernel, alpha=1e-5, n_restarts«_optimizer»=10)`
  - 12.2: `ersity), length_scale_bounds=(1e-2, 1«e»2)) self`
  - 12.2: `# Train initial GP model self.gp_model = Gaussian«Process»Regressor( kernel`

## L21 · adaptive_parameters

- **f35776** AUROC 0.926 (val), active 1.00 / 0.88; max tokens `'ive'` ×30, `'aptive'` ×22, `'adaptive'` ×16
  - 22.6: `: # Use Bayesian optimization to suggest new points # Predict«ive» mean and variance for`
  - 22.6: `sklearn.gaussian_process.kernels import RBF, WhiteKernel class« Adaptive»DE_Bayesian`
  - 22.6: `if len(self.X) > 1: # Predict«ive» mean and variance`
  - 22.2: `_opt = mutant.copy() # Bayesian optimization step # Predict«ive» mean and variance`
  - 22.1: `np.random.rand() < self.exploration_ratio: # Predict«ive» mean and variance`
- **f2295** AUROC 0.904 (val), active 1.00 / 0.59; max tokens `'.ad'` ×53, `'Adaptive'` ×18, `'adaptive'` ×11
  - 19.1: `) / self.budget) * self.reduction_factor #« Adaptive» random perturbation`
  - 19.0: `.98 # Adaptive step size control if self«.ad»aptive_steps < self`
  - 18.4: `self.adaptation_factor * self.momentum + (1 - self«.ad»aptation_factor)`
  - 18.2: `# Adapt population size and step size periodically if i % self«.ad»apt_interval ==`
  - 18.0: `+ self.diversity_weight * np.eye(self.dim) #« Adaptive» sigma control based on`
- **f32285** AUROC 0.927 (val), active 0.98 / 0.49; max tokens `'aptive'` ×16, `'apt'` ×9, `'adapt'` ×7
  - 16.0: `_rate self.penalty_coeff = penalty_coeff self.ad«apt»_perturbation =`
  - 15.6: `=0.9, pop_size=50, local«_search»_radius=0.`
  - 14.7: `: increase perturbation scale if diversity is low if self.ad«apt»_perturbation:`
  - 14.6: `_perturbation = max_perturbation self.use_bay«esian» = use_bay`
  - 14.2: `, surrogate_kernel=RBF(length_scale=1.0),« use»_surrogate=True):`

## L27 · population

- **f18650** AUROC 0.995 (val), active 1.00 / 0.17; max tokens `'.pop'` ×36, `'pop'` ×13, `'population'` ×10
  - 79.0: `itness[i] < self.best_fitness: self.best_solution = self«.pop»[i] self`
  - 76.6: `__(self, func): # Initial random sampling for _« in» range(self.init_samples`
  - 76.5: `self, func): # Initialize population with random samples self«.population» = [np.random`
  - 76.2: `[0], self.bounds[1], (self.mu, self.dim«)) » self.fitness`
  - 75.8: `func(x) self.f_evals += 1 self«.pop»ulations.append((x`
- **f33202** AUROC 0.988 (val), active 0.98 / 0.01; max tokens `'.pop'` ×57, `'.population'` ×9, `'pop'` ×7
  - 152.0: `if len(self.memory) > 0: self.memory«.pop»(0)`
  - 135.7: `if len(self.memory) >= self.memory_size: self.memory.pop«(»0) self`
  - 135.4: `np.random.uniform(func.bounds.lb, func.bounds.ub, (self«.population»_size, self.dim`
  - 131.9: `- sigma: Standard deviation for random perturbations. -« pop»_size: Size of`
  - 131.9: `self.pop_size * self.elite_ratio: self.elites.append«(pop»ulation[0])`
- **f28044** AUROC 0.972 (val), active 0.96 / 0.04; max tokens `'population'` ×17, `'.pop'` ×17, `'_size'` ×13
  - 105.4: `[] # Keep elites for idx in elites: new«_pop».append(self.pop[idx`
  - 103.8: `self.sigma = sigma self.pop_size = pop_size« » self.adapt`
  - 103.6: `self.dim = dim self.pop_size = pop_size« » self.learning_rate`
  - 103.3: `func.bounds.ub self._initialize() # Initial« population» self.pop`
  - 100.1: `self.budget = budget self.dim = dim self«.pop»_size = pop_size`

## L27 · differential_evolution

- **f40999** AUROC 0.999 (val), active 1.00 / 0.38; max tokens `'mutant'` ×61, `')'` ×13, `'ant'` ×8
  - 39.9: `.random.choice(range(self.pop_size), 3, replace=False) « mutant»[j] = population`
  - 39.6: `() for i in range(self.budget): # Generate« mutant» vectors mutant`
  - 39.3: `0.5, 1.0) # Generate« mutant» mutant =`
  - 39.1: `(pop[b] - pop[c]) # Apply bounds « mutant» = np.clip(mut`
  - 38.6: `.normal(0, 1, self.dim)) else: « mutant» = population[a]`
- **f26721** AUROC 0.957 (val), active 0.98 / 0.33; max tokens `''` ×22, `'='` ×9, `'.'` ×9
  - 43.0: `of tournament for selection self.crossover_rate = crossover_rate « #» Probability of crossover`
  - 42.0: `winner_fitness = np.min(selected_fitness) # Crossover:« Blend» crossover partner`
  - 41.5: `.budget): # Genetic operations # Select parents « parents» = np.random.choice`
  - 40.0: `min(self.fitness)] while evaluations_used < self.budget: « » # Evaluate all individuals`
  - 39.3: `population self.pop = np.random.uniform(-5.0,« »5.0,`
- **f6921** AUROC 0.933 (val), active 1.00 / 1.00; max tokens `'mutant'` ×17, `')'` ×16, `'s'` ×5
  - 55.4: `sensitivity_factor self.momentum = momentum self.expl«oration»_rate = exploration_rate`
  - 54.9: `0, local_radius=0.5): self.budget =« budget» self.dim`
  - 53.4: `s = np.random.choice(self.pop_size, 3, replace=False«) » a, b`
  - 52.9: `'''python import numpy as np class AdaptiveHybridEvolution«ary»Search: def`
  - 52.7: `xs)]) # Evaluate trial trial_fitness = func(trial«) » if trial_fitness`

## L27 · crossover

- **f6049** AUROC 0.983 (val), active 1.00 / 0.77; max tokens `'rossover'` ×41, `'crossover'` ×24, `'.pop'` ×4
  - 72.2: `, pop_size=20, mutation_factor=0.5,« crossover»_rate=0.`
  - 70.0: `_size=20, mut_scale=0.5, cross_prob«=»0.7,`
  - 62.8: `.random.rand(self.pop_size, self.dim) < self.crossover_rate« » trial = np`
  - 62.6: `(pop[b] - pop[c]) # Crossover « cross» = np.random.rand`
  - 61.5: `self.mut_decay = mut_decay self.cross_decay =« cross»_decay self`
- **f54342** AUROC 0.980 (val), active 0.99 / 0.05; max tokens `'rossover'` ×57, `'crossover'` ×31, `'.'` ×4
  - 34.9: `, pop_size=20, mut_scale=0.5,« cross»_prob=0.`
  - 33.3: `in range(self.dim): if np.random.rand() < self.c«rossover»_rate: trial`
  - 32.5: `=0.2, mutation_factor=0.5, « crossover»_rate=0.`
  - 32.1: `ant, func.bounds.lb, func.bounds.ub) # C«rossover» trial =`
  - 31.9: `self.mutation_strategies = mutation_strategies self.c«rossover»_strategies = crossover`
- **f58845** AUROC 0.976 (val), active 0.99 / 0.16; max tokens `'rossover'` ×58, `'crossover'` ×32, `'Inter'` ×1
  - 71.4: `memory_fit = self.memory[np.random.randint(len(self.memory))] #« Inter»polate between current population`
  - 48.4: `_fitness pop[j] = trial # Adaptive mutation and« crossover» if self`
  - 38.1: `.lb, func.bounds.ub) # Evaluate f_c«rossover» = func(crossover`
  - 37.5: `.pop[random_idx]) / 2 # Check bounds « crossover»_x = np.clip`
  - 37.3: `armony, -5.0, 5.0) « crossover»_fitness = func(c`

## L27 · particle_swarm

- **f46928** AUROC 0.994 (val), active 1.00 / 0.29; max tokens `'.velocity'` ×18, `'momentum'` ×16, `'omentum'` ×16
  - 73.5: `amp, self.velocity_clamp, (self.pop_size, self.dim«)) » self.f_pop`
  - 72.0: `.randn(self.dim) direction /= np.linalg.norm(direction) self«.velocity»[j] = self`
  - 71.1: `self.momentum * self.velocity[j] + self.learning_rate * grad« » current_x =`
  - 70.5: `func.bounds.ub) # Velocity update for momentum self«.velocity»[j] = self`
  - 70.5: `self.velocity + (self.x_opt - self.population.mean(axis=0«)) » self.cov`
- **f50643** AUROC 0.995 (val), active 1.00 / 0.70; max tokens `'ocities'` ×33, `'.vel'` ×20, `'momentum'` ×7
  - 100.7: `(self.dim) velocity = (self.inertia_weight * self.vel«ocities»[j] +`
  - 99.5: `ersity = 0.0 self.velocities = None« » def __call`
  - 97.7: `self.particles)) self.particles = self.particles + self«.vel»ocities #`
  - 97.3: `self.velocities[j] = self.momentum * self.vel«ocities»[j] + (`
  - 96.9: `# Update position self.particles[i] += self.vel«ocities»[i] #`
- **f38837** AUROC 0.933 (val), active 1.00 / 1.00; max tokens `''` ×18, `'self'` ×15, `'omentum'` ×5
  - 127.1: `.cognitive_coeff = cognitive_coeff self.social_coeff = social_coeff« » self.mutation`
  - 125.6: `def __call__(self, func): # Initialize swarm « » for _ in range`
  - 125.1: `* eps) self.velocities[i] = self.momentum *« self».velocities[i]`
  - 123.3: `, func.bounds.ub, (self.swarm_size, self.dim«)) » velocities = np`
  - 121.7: `v = self.beta * v - self.sigma * grad « » x_new = x`

## L27 · covariance

- **f11916** AUROC 0.973 (val), active 1.00 / 0.07; max tokens `'ariance'` ×26, `'ov'` ×24, `'covariance'` ×12
  - 154.9: `mean cov = np.cov(deviations, rowvar=False«) » self.sigma =`
  - 151.2: `.lb, bounds.ub) return trial def _cov«ariance»_matrix_adaptation`
  - 150.6: `self.mutation_rate = mutation_rate self.cov«ariance»_matrix = np.eye`
  - 150.4: `self.pop[j] self.best_index = j # Update« covariance» matrix and step size`
  - 149.3: `3): self.budget = budget self.dim = dim« » self.population_size`
- **f51420** AUROC 0.973 (val), active 1.00 / 0.69; max tokens `''` ×13, `','` ×9, `'self'` ×5
  - 168.3: `np.dot(K_inv, values) return alpha, K, K«_inv» def _`
  - 162.8: `mu = np.dot(K_s, np.dot(self.model[0«],» self.model[1`
  - 160.9: `self.f_opt = np.inf self.x_opt =« None» self.m`
  - 157.2: `np.dot(mean, np.linalg.pinv(K)) @ K @ np«.linalg».pinv(K)`
  - 153.6: `K_star = np.zeros((X_star.shape[0], X«.shape»[0]))`
- **f31440** AUROC 0.924 (val), active 1.00 / 1.00; max tokens `''` ×14, `'='` ×8, `'self'` ×7
  - 80.7: `.f_opt = f_trial self.x_opt = trial « #» Track success rate`
  - 78.8: `func.bounds.ub) f_perturb = func(x_p«ert»urb) grad`
  - 70.5: `_idx] # Initialize covariance matrix self.covariance_matrix« =» np.eye(self.dim`
  - 69.5: `.mean(axis=0) centered_population = self.population - mean « » self.covariance`
  - 69.3: `= f_trial self.x_opt = trial # Update« covariance» matrix using the population`

## L27 · step_size_adaptation

- **f45320** AUROC 0.729 (val), active 0.85 / 0.51; max tokens `''` ×11, `'#'` ×7, `'region'` ×6
  - 31.8: `Perturb the current best point with random search x = self«.x»_opt + np.random`
  - 31.4: `_center - self.trust_radius, self.trust_center + self.tr«ust»_radius) else`
  - 30.4: `self.trust_radius = trust_radius # Initial trust region« radius» self.tr`
  - 29.1: `ust_radius = trust_radius self.trust_decay = trust_decay« » self.trust`
  - 28.6: `_opt self.trust_radius *= self.trust_decay « #» Increase trust radius if`
- **f23270** AUROC 0.683 (val), active 0.69 / 0.30; max tokens `'*='` ×96
  - 16.5: `2 * 1e-5) self.velocities[i]« *=» self.momentum`
  - 16.0: `5.0, 5.0) self.momentum_vector« *=» 0.5`
  - 15.3: `else: # No improvement, reduce velocity self.velocity« *=» 0.5`
  - 15.3: `_opt = x_candidate # Update temperature self.current_temp« *=» self.temp_decay`
  - 15.2: `x_candidate # Update direction momentum self.direction_momentum« *=» self.momentum`
- **f21212** AUROC 0.695 (val), active 1.00 / 0.98; max tokens `''` ×18, `'.'` ×7, `'1'` ×7
  - 66.4: `self.min_sigma = min_sigma self.max_sigma =« max»_sigma self`
  - 64.0: `for _ in range(self.local_search_steps): perturbation = np«.random».uniform(-0.`
  - 62.0: `_opt) for _ in range(self.local_search_steps): pert«urbation» = np.random.uniform`
  - 59.8: `*= 1.1 else: std *= 0«.»99`
  - 59.7: `= dim self.initial_step = initial_step self.p«ert»urbation_ratio = pert`

## L27 · restart

- **f20316** AUROC 1.000 (val), active 1.00 / 0.02; max tokens `'starts'` ×82, `'.reshape'` ×12, `'('` ×1
  - 54.4: `self.gp_alpha = gp_alpha self.gp_n_re«starts» = gp_n_re`
  - 53.3: `kernel=self.gp_kernel, alpha=self.gp_alpha, n_re«starts»_optimizer=self.gp`
  - 53.2: `acquisition_type='ei', alpha=0.05, n_re«starts»=10):`
  - 52.6: `= GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha, n_re«starts»_optimizer=10`
  - 52.2: `= GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha, n_re«starts»_optimizer=10`
- **f793** AUROC 0.996 (val), active 1.00 / 0.03; max tokens `'starts'` ×68, `'rest'` ×21, `')'` ×1
  - 90.7: `.random.uniform(-5.0, 5.0, self.dim«) » restart_f =`
  - 78.8: `from scipy.stats import norm class CMA_Bayesian_R«est»art: def`
  - 73.9: `'''python import numpy as np class CMAESWithRandom«Rest»arts: def`
  - 72.5: `self.population[:elite_size] rest = self.population[elite_size«:] » # Compute gradients`
  - 51.3: `_fitness = combined_fitness[elite_idx] # Select remaining from the« rest» remaining_idx`
- **f44846** AUROC 0.995 (val), active 1.00 / 0.72; max tokens `'restart'` ×17, `'starts'` ×12, `'.re'` ×10
  - 121.1: `0 # Reset step count if new best found # Rein«force» direction self`
  - 99.4: `, alpha=0.5, noise_scale=0.3,« reb»alance_freq=1`
  - 99.0: `) # Rebalance population periodically if i % self.re«balance»_freq == 0`
  - 98.2: `_size = int(self.pop_size * self.shrink_ratio) #« Re»-initialize population with`
  - 96.4: `self.noise_scale = noise_scale self.rebalance_freq =« reb»alance_freq`

## L27 · local_search

- **f39575** AUROC 0.963 (val), active 0.97 / 0.17; max tokens `'local'` ×38, `'.local'` ×37, `'Local'` ×9
  - 49.1: `self.last_improvement = 0 # Gradient-based« local» search on the best`
  - 47.0: `_rate self.elite_ratio = elite_ratio self.local«_search»_freq = local_search`
  - 46.7: `mutation_factor self.crossover_rate = crossover_rate self«.local»_search_radius = local`
  - 46.6: `999, sigma_decay=0.9995,« local»_search_prob=0`
  - 46.4: `1, trust_radius=1.0, adaptive=True,« local»_search_prob=0`
- **f19256** AUROC 0.904 (val), active 1.00 / 0.90; max tokens `'_rate'` ×16, `'.linalg'` ×11, `'_search'` ×8
  - 88.0: `# We use scipy's minimize with a L-BFGS«-B» method and use the`
  - 73.2: `result = minimize(objective, self.x_opt, method='L«-B»FGS-B',`
  - 73.2: `0, 5.0, self.dim), method='L-B«FG»S-B', bounds`
  - 72.5: `], b[1]) for b in bounds], method='L«-B»FGS-B')`
  - 71.6: `esian - self.learning_rate * grad x_local = np.clip«(x»_local, func.bounds`
- **f44706** AUROC 0.922 (val), active 1.00 / 0.98; max tokens `'local'` ×18, `'.local'` ×18, `'ust'` ×5
  - 43.4: `.crossover_rate: new_harmony[i] = self.har«mony»_memory[idx][i`
  - 43.1: `self.crossover_rate = initial_crossover_rate self.m«utation»_decay_rate = mutation`
  - 39.1: `.linalg.norm(step) * grad # Perform local search with probability« local»_search_rate`
  - 38.6: `# Move towards trust region center direction = (self.tr«ust»_region_center - current`
  - 38.3: `999, min_step=1e-6, « random»_search_ratio=0`

## L27 · gradient

- **f49680** AUROC 0.918 (val), active 1.00 / 0.60; max tokens `''` ×16, `'grad'` ×10, `'np'` ×5
  - 39.8: `range(self.budget): # Stochastic Gradient Descent step « » grad = np.zeros`
  - 36.8: `(x - e)) / (2 * 1e-5) « » return f, grad`
  - 36.2: `_decay = grad_decay # Decay factor for gradient influence def« __»call__(self,`
  - 35.9: `(mutant, func.bounds.lb, func.bounds.ub) #« Gradient»-informed mutation`
  - 35.8: `.linalg.norm(grad) > 1e-8: grad =« grad» / np.linalg.norm`
- **f2271** AUROC 0.936 (val), active 1.00 / 0.83; max tokens `''` ×13, `'='` ×7, `'x'` ×6
  - 199.6: `] = (func(x_plus) - func(x_minus)) / («2» * eps)`
  - 198.7: `= (func(self.x_opt + e) - func(self.x_opt« -» e)) / (`
  - 192.6: `5.0, 5.0) # Evaluate « f» = func(x)`
  - 190.5: `# Compute gradient using finite differences grad = np.gradient(func,« x», 1e`
  - 190.0: `Evaluate gradient at current point # Use finite difference approximation for gradient« » grad = np`
- **f25797** AUROC 0.920 (val), active 1.00 / 0.31; max tokens `'grad'` ×17, `'gradient'` ×14, `'Gradient'` ×7
  - 57.2: `self.curvature_cache: return self.curvature_cache[x] « grad» = self._approx`
  - 55.3: `np.random.rand() < self.exploration_rate: # Estimate« curvature» (second derivative)`
  - 52.8: `) # Adjust step size based on curvature if abs(cur«vature») > self.cur`
  - 52.6: `[i] += self.step_size x_plus_ij[j] +=« self».step_size`
  - 52.3: `# Reduce step size for high curvature elif np.abs(h«essian»_diag).mean()`

## L27 · elitism

- **f29867** AUROC 0.999 (val), active 1.00 / 0.00; max tokens `'ite'` ×36, `'elite'` ×33, `'Elite'` ×6
  - 43.7: `, mutation_rate=0.1, adaptive_learning=True, « elite»_ratio=0.`
  - 42.0: `par self.mutation_rate = mutation_rate self.el«ite»_ratio = elite_ratio`
  - 41.9: `)[:int(self.elite_ratio * self.pop_size)] pop[«elite»_idx] = pop`
  - 41.9: `self.mutation_factor - 0.01) #« Elite» preservation elite`
  - 41.4: `fraction of population sorted_indices = np.argsort(fitness) « elite»_size = int(self`
- **f752** AUROC 0.999 (val), active 1.00 / 0.03; max tokens `'elite'` ×42, `'ite'` ×22, `''` ×5
  - 37.8: `_threshold=1e-6, pop_size=5,« elite»_ratio=0.`
  - 37.2: `) new_pop.append(new_x) else: # El«it»ism and gradient-based`
  - 37.0: `0.15, gradient_smoothing=0.95,« elite»_ratio=0.`
  - 36.8: `0.5, mutation_factor_max=1.5,« elite»_ratio=0.`
  - 36.5: `in range(self.pop_size): # Elitism: keep the« best» individual if`
- **f46445** AUROC 0.999 (val), active 1.00 / 0.01; max tokens `'.el'` ×18, `'it'` ×14, `'ism'` ×12
  - 65.1: `.init_samples = init_samples self.elite_ratio = elite_ratio« » self.f_opt`
  - 63.8: `new_pop = np.zeros((pop_size, self.dim)) « elites» = np.argsort(f`
  - 63.7: `Select elites and replace population self.current_pop = self._select«_el»ites(new_pop,`
  - 62.6: `self.population = self.population[:self.pop_size] # El«it»ism: keep the`
  - 61.3: `() self.elite_fitness = self.fitness.copy() #« El»itism-based refinement`

## L27 · surrogate

- **f63904** AUROC 1.000 (val), active 1.00 / 1.00; max tokens `''` ×16, `'.g'` ×6, `','` ×5
  - 99.7: `self.f_opt = np.inf self.x_opt = None « » self.surrogate`
  - 95.7: `population = np.random.uniform(-5.0, 5«.»0, (self`
  - 95.6: `), alpha=self.noise_var, n_restarts_optimizer=«1»0 )`
  - 93.3: `_opt) self.y_train.append(self.f_opt) self.s«ur»rogate_model.fit(np`
  - 91.8: `_surrogate(self, model, X, f): """Update the« R»BF surrogate model with`
- **f62130** AUROC 1.000 (val), active 1.00 / 0.65; max tokens `''` ×17, `'self'` ×11, `'Gaussian'` ×4
  - 57.4: `: return lambda x: np.inf weights = np.linalg«.lst»sq(X, Y`
  - 55.1: `1: # Fit a linear regression model to the surrogate points « » X = np.column`
  - 54.1: `urrogate_points) y = np.array(self.surrogate_values«) » X = np`
  - 53.6: `_sol def _predict_surrogate(self, point): if« self».surrogate_points`
  - 51.5: `surrogate model.""" X = np.hstack((X, np.ones((«X».shape[0],`
- **f3488** AUROC 1.000 (val), active 1.00 / 0.97; max tokens `''` ×20, `'='` ×5, `'#'` ×5
  - 110.2: `# Upper Confidence Bound acquisition function mu, sigma = self._«predict»(X) return`
  - 106.8: `value and uncertainty X = np.array([x]) y«_mean», y_std =`
  - 106.4: `)) / (2 * epsilon) return grad def _«get»_uncertainty(self,`
  - 106.3: `self.acquisition_function == 'ei': acq_values = («mean» - np.min(self`
  - 106.0: `np.clip(x, -5.0, 5.0) « » if np.any(np`

## L27 · adaptive_parameters

- **f61505** AUROC 0.942 (val), active 1.00 / 0.83; max tokens `'apt'` ×37, `'Adaptive'` ×23, `'Adapt'` ×9
  - 98.5: `.f[idx] self.x_opt = self.pop[idx] #« Adaptive» noise control based on`
  - 94.2: `[i] elite_fitness[i] = trial_fitness[i] #« Adapt» mutation factor based on`
  - 92.4: `Balance # Code: '''python import numpy as np class« Adaptive»CovarianceM`
  - 91.6: `= np.std(self.population, axis=0).mean() #« Adapt» mutation and crossover based`
  - 91.6: `Code: '''python import numpy as np class HybridGradientAd«aptive»MultiScaleSearch:`
- **f25158** AUROC 0.910 (val), active 0.99 / 0.80; max tokens `'aptive'` ×14, `'.ad'` ×10, `'replace'` ×6
  - 94.0: `= trial # Adaptive mutation and crossover if self.ad«aptive»: # Update`
  - 93.3: `self.CR = CR # Crossover probability self«.ad»aptive = adaptive`
  - 90.6: `= trial # Adaptive step size control if self.ad«aptive»: # Calculate`
  - 88.3: `, F=0.8, CR=0.9, adaptive«=True»): self.b`
  - 85.6: `] = trial # Adaptive mutation and crossover if self«.ad»aptive: #`
- **f30567** AUROC 0.900 (val), active 1.00 / 0.99; max tokens `'adaptive'` ×11, `'aptive'` ×9, `'finite'` ×8
  - 31.1: `self.fitness = None self.mutation_rate =« adaptive»_mutation_rate`
  - 30.1: `self.x_opt = pop[np.argmin(fitness)] self«.best»_fitness_history = [`
  - 30.0: `self.f_opt, self.x_opt def _select_diver«se»_candidates(self):`
  - 29.8: `exploitation. # Code: '''python import numpy as np class« Adaptive»EvolutionaryHy`
  - 29.3: `.elite, self.f_min, self.f_max) # Adaptive« local» search rate based on`
