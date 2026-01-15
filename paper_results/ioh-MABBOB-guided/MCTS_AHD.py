import numpy as np

class MirrorAdaptiveDECovInject:
    """
    MirrorAdaptiveDECovInject:
    - Hybrid of mirrored Gaussian sampling and adaptive Differential Evolution (rand/1/bin).
    - Maintains a covariance matrix to shape Gaussian proposals and updates it from selected elites.
    - Uses exponential rank weights (different from log-weights in the reference).
    - Adaptive F and CR per trial (sampled from heavy-tailed / normal priors).
    - Eigenvalue clamping + covariance injection instead of plain floor blend; opportunistic eigen-inflation
      / small multivariate-t probes on stagnation (different equations and parameter settings).
    """

    def __init__(self, budget=10000, dim=10, seed=None, pop_base=None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = seed
        # Algorithmic knobs (deliberately different choices from the provided HybridCMADE)
        self.pop_base = pop_base  # if None, computed below (uses sqrt scaling instead of log)
        self.cov_lr = 0.10               # slower covariance learning than reference's 0.25
        self.sigma_adapt_rate = 0.15     # milder step-size adaptation
        self.success_target = 0.25       # slightly higher target success rate
        self.archive_size = min(50, max(5, 6 * self.dim))  # larger archive scaling
        self.stagnation_iters = max(15, int(0.03 * self.budget))  # different stagnation trigger
        self.levy_prob = 0.10            # rarer heavy-tailed probes
        self.F_base = 0.6                # base differential weight
        self.CR_base = 0.9               # base crossover prob
        self.max_eig_ratio = 1e6         # maximal condition number allowed
        self.min_eig_scale = 1e-12       # absolute minimum eigenvalue floor

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)

        # bounds (Many BBOB uses [-5,5], but honor func bounds)
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        if lb.size == 1:
            lb = np.full(self.dim, lb.item(), dtype=float)
        if ub.size == 1:
            ub = np.full(self.dim, ub.item(), dtype=float)
        assert lb.shape[0] == self.dim and ub.shape[0] == self.dim
        bounds_scale = (ub - lb)

        # adaptive population size (different from original log formula)
        if self.pop_base is None:
            lam = max(8, int(6 + 2.0 * np.sqrt(max(1, self.dim))))
        else:
            lam = int(self.pop_base)
        lam = min(lam, max(2, self.budget))

        evals = 0
        # initial uniform seeding (consume up to lam evaluations)
        init_n = min(lam, self.budget - evals)
        X0 = rng.uniform(lb, ub, size=(init_n, self.dim))
        f0 = np.array([func(x) for x in X0])
        evals += init_n

        # best-known
        best_idx = int(np.argmin(f0))
        f_best = float(f0[best_idx])
        x_best = X0[best_idx].copy()

        # archive (store as arrays)
        archive_X = X0.copy()
        archive_f = f0.copy()

        def archive_add(x, fx):
            nonlocal archive_X, archive_f
            if len(archive_f) < self.archive_size:
                archive_X = np.vstack([archive_X, x.reshape(1, -1)])
                archive_f = np.concatenate([archive_f, np.array([fx])])
            else:
                worst_idx = int(np.argmax(archive_f))
                if fx < archive_f[worst_idx]:
                    archive_X[worst_idx] = x
                    archive_f[worst_idx] = fx

        # initialize mean by exponential rank-weighted average of the top half
        mu0 = max(1, init_n // 2)
        order0 = np.argsort(f0)
        elites0 = X0[order0[:mu0]]
        # exponential weights (different equation)
        ranks = np.arange(mu0)
        weights0 = np.exp(-ranks / max(1.0, mu0 / 3.0))
        weights0 = weights0 / weights0.sum()
        m = (weights0.reshape(-1, 1) * elites0).sum(axis=0)

        # initial covariance: isotropic scaled relative to bounds but different divisor
        C = np.diag(((bounds_scale / 6.0) ** 2).clip(min=1e-16))
        sigma = 0.20 * np.mean(bounds_scale)
        sigma = max(sigma, 1e-12)

        # strategy state
        p_succ = self.success_target
        stagn_iters = 0
        iter_count = 0

        # helper: safe decomposition to get transform A with A^T A = C
        def chol_like(Cmat):
            # prefer eigh and form sqrt(V D V^T) to allow rectangular A
            vals, vecs = np.linalg.eigh(Cmat)
            vals_clipped = np.clip(vals, self.min_eig_scale, None)
            # enforce condition limit
            if vals_clipped.max() / vals_clipped.min() > self.max_eig_ratio:
                # rescale largest eigenvalues to maintain condition number
                max_allowed = vals_clipped.min() * self.max_eig_ratio
                vals_clipped = np.minimum(vals_clipped, max_allowed)
            A = (vecs * np.sqrt(vals_clipped)).T  # A @ A.T = Cmat, A^T @ A = ...
            return A

        # main loop: generate up to lam candidates per generation
        while evals < self.budget:
            iter_count += 1
            remaining = self.budget - evals
            lam_iter = min(lam, remaining)
            mu = max(1, lam_iter // 2)

            # recompute exponential rank weights (different equation)
            ranks = np.arange(mu)
            weights = np.exp(-ranks / max(1.0, mu / 3.0))
            weights = weights / np.sum(weights)

            A = chol_like(C)

            # candidate composition: half mirrored Gaussian, half adaptive DE
            n_gauss = lam_iter // 2
            n_de = lam_iter - n_gauss
            Xcand = np.empty((lam_iter, self.dim), dtype=float)

            # 1) mirrored Gaussian proposals: produce pairs (x, 2m - x) for variance reduction
            if n_gauss > 0:
                # generate n_gauss/2 unique normals and mirror them; if odd handle last separately
                Z = rng.normal(size=(n_gauss, self.dim))
                Y = Z @ (A.T)
                Xg = m + sigma * Y
                # mirrored
                Xg_mirror = 2.0 * m - Xg
                # mix sequence Xg and mirrors to fill n_gauss slots
                for i in range(n_gauss):
                    if i < len(Xg):
                        cand = Xg[i]
                    else:
                        cand = Xg_mirror[i - len(Xg)]
                    # clamp
                    Xcand[i] = np.minimum(np.maximum(cand, lb), ub)

            # 2) adaptive DE rand/1/bin proposals using archive
            if n_de > 0:
                # ensure sorted archive for selection bias towards elites
                if len(archive_f) >= 3:
                    idx_sort = np.argsort(archive_f)
                    sorted_X = archive_X[idx_sort]
                    sorted_f = archive_f[idx_sort]
                else:
                    sorted_X = archive_X
                    sorted_f = archive_f

                for i in range(n_de):
                    # sample F from Cauchy centered at base, heavy-tailed but clipped
                    F = rng.standard_cauchy() * 0.1 + self.F_base
                    F = float(np.clip(F, 0.2, 1.0))
                    # sample CR from normal near base
                    CR = float(np.clip(rng.normal(loc=self.CR_base, scale=0.15), 0.0, 1.0))

                    if len(sorted_X) < 3:
                        # fallback gaussian
                        z = rng.normal(size=self.dim)
                        y = z @ (A.T)
                        trial = m + sigma * y
                    else:
                        # pick three distinct indices for rand/1
                        idxs = rng.choice(len(sorted_X), size=3, replace=False)
                        xr = sorted_X[idxs[0]]
                        xa = sorted_X[idxs[1]]
                        xb = sorted_X[idxs[2]]
                        mutant = xr + F * (xa - xb)
                        # target vector chosen randomly between m and a random elite (introduce diversity)
                        if rng.rand() < 0.5:
                            target = m
                        else:
                            target = sorted_X[rng.randint(len(sorted_X))]
                        # binomial crossover
                        mask = rng.rand(self.dim) < CR
                        if not np.any(mask):
                            mask[rng.randint(self.dim)] = True
                        trial = np.where(mask, mutant, target)
                        # small Gaussian jitter scaled by sigma and relative to bounds
                        trial += rng.normal(scale=0.3 * sigma, size=self.dim)

                    # clamp
                    Xcand[n_gauss + i] = np.minimum(np.maximum(trial, lb), ub)

            # evaluate candidates (respect budget)
            f_cand = np.empty(lam_iter, dtype=float)
            for i in range(lam_iter):
                if evals >= self.budget:
                    # Should not happen as lam_iter chosen <= remaining, but guard anyway
                    f_cand[i] = np.inf
                    continue
                f_cand[i] = float(func(Xcand[i]))
            evals += lam_iter

            # update archive
            for i in range(lam_iter):
                archive_add(Xcand[i].copy(), float(f_cand[i]))

            # generation best
            gen_best_idx = int(np.argmin(f_cand))
            gen_best_f = float(f_cand[gen_best_idx])
            gen_best_x = Xcand[gen_best_idx].copy()

            improved = False
            if gen_best_f < f_best:
                f_best = gen_best_f
                x_best = gen_best_x.copy()
                improved = True
                stagn_iters = 0
            else:
                stagn_iters += 1

            # selection: choose top-mu candidates
            order = np.argsort(f_cand)
            X_mu = Xcand[order[:mu]]

            # recompute weighted mean (exponential rank weights)
            m_new = (weights.reshape(-1, 1) * X_mu).sum(axis=0)

            # deltas normalized by sigma for covariance update
            deltas = (X_mu - m) / (sigma + 1e-20)
            W = weights.reshape(-1, 1)
            weighted_cov = (deltas * W).T @ deltas  # (dim x dim)

            # include small rank-one update from mean shift
            mean_shift = ((m_new - m) / max(sigma, 1e-20)).reshape(-1, 1)
            rank_one = (mean_shift @ mean_shift.T) * 0.5

            # covariance update with injection term (different blend than reference)
            C = (1.0 - self.cov_lr) * C + self.cov_lr * (weighted_cov + 0.5 * rank_one)

            # eigen-regularize: clamp eigenvalues to reasonable range relative to bounds_scale
            vals, vecs = np.linalg.eigh(C)
            # set minimum eigenvalue relative to typical squared scale of bounds
            min_eig = max(self.min_eig_scale, (np.mean(bounds_scale) * 1e-3) ** 2)
            vals_clipped = np.clip(vals, min_eig, None)
            # enforce max condition number
            max_allowed = vals_clipped.min() * self.max_eig_ratio
            vals_clipped = np.minimum(vals_clipped, max_allowed)
            C = (vecs * vals_clipped) @ vecs.T

            # update mean
            m = m_new.copy()

            # step-size adaptation via smoothed success-rate but with different smoothing
            p_succ = 0.85 * p_succ + 0.15 * float(improved)
            sigma *= np.exp(self.sigma_adapt_rate * (p_succ - self.success_target))
            # clip sigma to sensible bounds (bounded by domain size)
            sigma = float(np.clip(sigma, 1e-12, 1.5 * np.max(bounds_scale)))

            # stagnation handling: covariance injection or small multivariate-t probes
            if stagn_iters >= self.stagnation_iters and evals < self.budget:
                if rng.rand() < self.levy_prob:
                    # heavy-tailed multivariate-t probe centered at one of the top elites
                    idx_sort = np.argsort(archive_f)
                    topK = min(len(idx_sort), max(3, self.dim))
                    anchor = archive_X[idx_sort[rng.randint(topK)]]
                    # multivariate t-like by sampling normal and dividing by sqrt(gamma)
                    # gamma drawn from chi2 with df small to create heavy tails
                    df = 2.0
                    gamma = max(1e-8, rng.chisquare(df))
                    z = rng.normal(size=self.dim)
                    scale = max(0.4 * np.mean(bounds_scale), sigma * 3.0)
                    jump = z / np.sqrt(gamma / df) * scale
                    x_jump = anchor + jump
                    x_jump = np.minimum(np.maximum(x_jump, lb), ub)
                    fj = float(func(x_jump))
                    evals += 1
                    archive_add(x_jump.copy(), fj)
                    if fj < f_best:
                        f_best = fj
                        x_best = x_jump.copy()
                        stagn_iters = 0
                        m = x_jump.copy()
                        sigma = max(sigma, 0.4 * np.mean(bounds_scale))
                        C = np.diag(((bounds_scale / 5.0) ** 2).clip(min=1e-16))
                    else:
                        # perform eigen-inflation: increase small eigenvalues moderately to encourage new directions
                        vals, vecs = np.linalg.eigh(C)
                        vals = np.maximum(vals, min_eig)
                        inflation = 1.5
                        vals = vals * inflation
                        C = (vecs * vals) @ vecs.T
                        # nudge mean slightly toward the best
                        m = 0.85 * m + 0.15 * x_best
                        sigma = min(1.5 * np.max(bounds_scale), sigma * 1.4)
                else:
                    # milder restart around the best with small Gaussian jitter (not full reinitialization)
                    jitter = rng.normal(scale=0.08 * np.maximum(bounds_scale, 1.0), size=self.dim)
                    m = x_best + jitter
                    m = np.minimum(np.maximum(m, lb), ub)
                    C = np.diag(((bounds_scale / 7.0) ** 2).clip(min=1e-16))
                    sigma = max(sigma, 0.25 * np.mean(bounds_scale))
                stagn_iters = 0
                p_succ = self.success_target  # reset smoothed success estimate

        # finished budget
        self.f_opt = float(f_best)
        self.x_opt = np.asarray(x_best, dtype=float)
        return self.f_opt, self.x_opt