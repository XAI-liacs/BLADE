import numpy as np

class AdaptiveSubspaceCovarianceSearch:
    """
    Adaptive Subspace Covariance Search (ASCS)
    - Hybrid global-local sampler that alternates covariance-adapted multivariate
      sampling with targeted principal-direction line searches and adaptive step-size.
    - Budget-aware; respects provided function-evaluation budget and bound constraints.
    """
    def __init__(self, budget=10000, dim=10, pop_size=None, seed=None):
        self.budget = int(budget)
        self.dim = int(dim)
        if pop_size is None:
            # default population scales with dimension
            self.pop_size = max(4 * self.dim, 12)
        else:
            self.pop_size = int(pop_size)
        self.rng = np.random.RandomState(None if seed is None else int(seed))

    def __call__(self, func):
        # Robustly get bounds as arrays
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        if lb.ndim == 0:
            lb = np.full(self.dim, lb.item())
        if ub.ndim == 0:
            ub = np.full(self.dim, ub.item())

        # local helpers
        def reflect_clip(x):
            # single symmetric reflection then clip to bounds (safe and simple)
            x = np.array(x, dtype=float)
            # reflect values above upper bound
            over = x > ub
            if np.any(over):
                x[over] = ub[over] - (x[over] - ub[over])
            under = x < lb
            if np.any(under):
                x[under] = lb[under] - (lb[under] - x[under])
            # final clip to ensure in bounds
            return np.clip(x, lb, ub)

        # bookkeeping
        evals = 0
        f_opt = np.inf
        x_opt = None

        # initialize a small random sampling seed (n_init) while respecting budget
        n_init = min(max(6, 2 * self.dim), max(1, self.budget // 10))
        n_init = min(n_init, self.budget)
        samples_x = []
        samples_f = []
        for _ in range(n_init):
            x = self.rng.uniform(lb, ub)
            f = func(x)
            evals += 1
            samples_x.append(x.copy())
            samples_f.append(float(f))
            if f < f_opt:
                f_opt = float(f)
                x_opt = x.copy()
            if evals >= self.budget:
                break

        # If no evaluations were possible (budget==0), return
        if evals == 0:
            return f_opt, x_opt

        # center and its fitness
        center = x_opt.copy()
        f_center = f_opt

        # initial covariance: anisotropic diagonal proportional to (range/4)^2
        range_vec = ub - lb
        init_var = (range_vec / 4.0) ** 2
        init_var = np.maximum(init_var, 1e-12)
        C = np.diag(init_var)

        # initial global step-size multiplier (sigma) scale ~ quarter of average range
        sigma = 0.25 * np.mean(range_vec)
        sigma = max(sigma, 1e-8)

        # adaptation parameters
        alpha_cov = 0.18   # covariance mixing rate
        inc_factor = 1.2
        dec_factor = 0.88
        min_sigma = 1e-8
        max_sigma = 5.0 * np.mean(range_vec)

        # stagnation control
        iter_count = 0
        last_improve_iter = 0
        patience = max(10, 5 * self.dim)

        # main loop: sample around center, adapt covariance and sigma, and occasional 1D probes
        while evals < self.budget:
            rem = self.budget - evals
            bs = min(self.pop_size, rem)
            batch_x = []
            batch_f = []

            # Sample candidates and evaluate sequentially (respect budget)
            for k in range(bs):
                # draw from N(0, C) scaled by sigma
                try:
                    # sample zero-mean multivariate normal with covariance C
                    z = self.rng.multivariate_normal(np.zeros(self.dim), C)
                except Exception:
                    # fallback to isotropic/directional using diag of C
                    diag = np.maximum(np.diag(C), 1e-20)
                    z = self.rng.randn(self.dim) * np.sqrt(diag)
                x = reflect_clip(center + sigma * z)
                f = func(x)
                evals += 1

                batch_x.append(x.copy())
                batch_f.append(float(f))

                # immediate global best update
                if f < f_opt:
                    f_opt = float(f)
                    x_opt = x.copy()

                if evals >= self.budget:
                    break

            if len(batch_f) == 0:
                break

            # find best in batch
            best_idx = int(np.argmin(batch_f))
            best_x = batch_x[best_idx]
            best_f = batch_f[best_idx]

            improved = False
            # If the batch produced an improvement vs center -> move center toward the best
            if best_f < f_center:
                improved = True
                lr = 0.6  # learning rate toward best
                prev_center = center.copy()
                center = lr * best_x + (1 - lr) * center
                center = reflect_clip(center)
                f_center = float(best_f)
                sigma = min(max_sigma, sigma * inc_factor)
                last_improve_iter = iter_count
                # small random perturb to escape local saddle
                center = reflect_clip(center + 0.03 * sigma * self.rng.randn(self.dim))
            else:
                # no immediate improvement: modest shrinkage
                sigma = max(min_sigma, sigma * dec_factor)

            # Build covariance update from top-performing samples in the batch
            bs_avail = len(batch_x)
            top_k = max(2, bs_avail // 3)
            # indices of best top_k samples
            idxs = np.argsort(batch_f)[:top_k]
            X = np.vstack([batch_x[i] for i in idxs]) - center  # deviations from new center

            # compute sample covariance (biased estimate ok) and scale to maintain average scale
            if X.shape[0] == 1:
                S = np.outer(X[0], X[0])
            else:
                # rowvar=False equivalent: each row is an observation
                # Use covariance = (X^T X) / n_samples
                S = (X.T @ X) / float(X.shape[0])

            # normalize S so its mean variance equals current average diag(C)
            mean_var_C = np.mean(np.diag(C))
            mean_var_S = np.trace(S) / float(self.dim) if np.trace(S) > 0 else 0.0
            if mean_var_S <= 0:
                S_scaled = S * 0.0 + np.diag(init_var) * 1e-4
            else:
                S_scaled = S * (mean_var_C / (mean_var_S + 1e-16))

            # mix into covariance matrix
            C = (1 - alpha_cov) * C + alpha_cov * S_scaled
            # stabilize covariance (small ridge)
            C += 1e-12 * np.eye(self.dim)

            # Occasionally perform directed 1D line probes along principal axis
            if (iter_count % max(3, self.dim // 2) == 0) and (evals < self.budget):
                # principal eigenvector
                try:
                    w, v = np.linalg.eigh(C)
                    pv = v[:, np.argmax(w.real)].real
                except Exception:
                    # fallback to random direction
                    pv = self.rng.randn(self.dim)
                    pv /= np.linalg.norm(pv) + 1e-12

                # try a set of relative step sizes along pv and -pv, prefer smaller first
                probe_steps = [0.5, 1.0, 2.0]
                for s in probe_steps + [-p for p in probe_steps]:
                    if evals >= self.budget:
                        break
                    x_try = reflect_clip(center + sigma * s * pv)
                    f_try = func(x_try)
                    evals += 1
                    if f_try < f_center:
                        center = x_try.copy()
                        f_center = float(f_try)
                        # encourage covariance to align with pv by rank-1 boost
                        rank1 = np.outer(pv, pv) * (np.mean(np.diag(C)) + 1e-12)
                        C = (1 - 0.5 * alpha_cov) * C + (0.5 * alpha_cov) * rank1
                        sigma = min(max_sigma, sigma * (inc_factor ** 1.1))
                        last_improve_iter = iter_count
                        # update global best if needed
                        if f_try < f_opt:
                            f_opt = float(f_try)
                            x_opt = x_try.copy()
                        # after a successful probe, stop other probes to conserve budget
                        break

            # stagnation handling: if no improvement for long, increase exploratory jitter
            if (iter_count - last_improve_iter) > patience:
                # random restart/large jump with low probability
                if self.rng.rand() < 0.15:
                    jump = 0.5 * np.mean(range_vec) * (1.0 + self.rng.randn(self.dim) * 0.5)
                    center = reflect_clip(center + jump)
                    # reinitialize sigma to encourage global exploration
                    sigma = min(max_sigma, sigma * 2.0)
                    last_improve_iter = iter_count
                else:
                    # gradually cool down sigma to focus
                    sigma = max(min_sigma, sigma * dec_factor)

            iter_count += 1

            # safety cap sigma
            sigma = np.clip(sigma, min_sigma, max_sigma)

        return float(f_opt), np.array(x_opt, dtype=float)