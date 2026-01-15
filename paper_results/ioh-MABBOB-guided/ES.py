import numpy as np

class DARESv2_Plus:
    """
    DARESv2_Plus

    Improvements over the provided DARESv2:
    - Explicit trust-region style local quadratic surrogate in a small directional subspace
      (spanned by directional memory + principal components of recent elites). The surrogate
      is a lightweight diagonal-quadratic + linear model in that subspace, fit with regularized
      least-squares and used to propose a budget-aware model step.
    - Adaptive anisotropic per-dimension scaling (cov_diag) with stabilized updates.
    - Budget-aware usage of parabolic probes and surrogate evaluations (never exceed budget).
    - Lévy-style occasional escape jumps that scale with stagnation to leave deceptive basins.
    - Improved restart policy that preserves dominant memory directions and shrinks covariance.
    - Careful mirrored/antithetic sampling and momentum-smoothed sigma adaptation.
    """

    def __init__(self, budget, dim, population=None, init_sigma=0.2, mem_size=6,
                 seed=None, max_elite=80, model_min_points=10):
        """
        Required:
        - budget: total function evaluations allowed
        - dim: problem dimensionality

        Optional:
        - population: initial population size heuristic override
        - init_sigma: initial step-size fraction of domain mean span
        - mem_size: number of directional memory vectors to keep (low-rank)
        - seed: RNG seed
        - max_elite: number of elite points to keep for surrogate fitting
        - model_min_points: minimum elite points to attempt surrogate fit
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.population = population
        self.init_sigma = float(init_sigma)
        self.mem_size = int(mem_size)
        self.seed = seed
        self.max_elite = int(max_elite)
        self.model_min_points = max(6, int(model_min_points))
        self.rng = np.random.RandomState(seed)

    def _get_bounds(self, func):
        try:
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        except Exception:
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)
        if lb.shape == ():
            lb = np.full(self.dim, float(lb))
        if ub.shape == ():
            ub = np.full(self.dim, float(ub))
        return lb, ub

    def _orthonormalize_memory(self, M):
        # Orthonormalize rows of M via Gram-Schmidt -> return rows as orthonormal basis
        if len(M) == 0:
            return np.zeros((0, self.dim))
        A = np.array(M, dtype=float)
        Q = []
        for i in range(A.shape[0]):
            v = A[i].copy()
            for q in Q:
                v -= np.dot(q, v) * q
            nrm = np.linalg.norm(v)
            if nrm > 1e-12:
                Q.append(v / nrm)
        if len(Q) == 0:
            return np.zeros((0, self.dim))
        return np.vstack(Q)

    def _levy_jump(self, scale):
        # Simple heavy-tailed jump using Cauchy scaled by provided scale
        return self.rng.standard_cauchy(self.dim) * scale

    def __call__(self, func):
        evals = 0
        budget = int(self.budget)
        dim = int(self.dim)
        lb, ub = self._get_bounds(func)
        span = ub - lb
        span_mean = float(np.mean(np.abs(span)))
        span_max = float(np.max(np.abs(span)))
        span_mean = max(span_mean, 1e-12)

        # safe evaluator that enforces budget
        def safe_eval(x):
            nonlocal evals
            if evals >= budget:
                return None
            # ensure contiguous float64
            x = np.asarray(x, dtype=float)
            val = float(func(x))
            evals += 1
            return val

        # Initialize a small random sample to seed mean and best
        init_n = min(max(10, 2 * dim), max(1, budget // 12))
        f_opt = np.inf
        x_opt = None
        samples = []
        for _ in range(init_n):
            if evals >= budget:
                break
            x = self.rng.uniform(lb, ub)
            f = safe_eval(x)
            if f is None:
                break
            samples.append((x.copy(), f))
            if f < f_opt:
                f_opt = f
                x_opt = x.copy()

        # If nothing evaluated yet (very small budget), return trivial
        if evals >= budget:
            return f_opt, x_opt

        if x_opt is None:
            mean = self.rng.uniform(lb, ub)
            f_mean = safe_eval(mean)
            if f_mean is None:
                return f_opt, x_opt
            f_opt = f_mean
            x_opt = mean.copy()
        else:
            mean = x_opt.copy()
            # ensure f_mean known
            f_mean = f_opt

        # population settings
        if self.population is None:
            lam = int(max(4, 4 + int(3 * np.log(max(1, dim)))))
        else:
            lam = int(self.population)
        lam = min(lam, max(2, budget // 20))
        if lam % 2 == 1:
            lam += 1
        lam_min = 2
        lam_max = max(4, min(300, budget // 4))

        # recombination weights
        mu = max(1, lam // 2)
        def compute_weights(mu):
            ranks = np.arange(1, mu + 1)
            w = np.log(mu + 0.5) - np.log(ranks)
            w = np.maximum(w, 0.0)
            if w.sum() <= 0:
                w = np.ones(mu)
            return w / w.sum()
        weights = compute_weights(mu)

        # strategy state
        sigma = max(1e-12, self.init_sigma * span_mean)
        sigma_min = 1e-12
        sigma_max = span_max * 4.0 if span_max > 0 else 1.0

        cov_diag = np.ones(dim)
        cov_eps = 1e-12
        cov_beta = 0.06  # EMA for covariance diag
        rank_beta = 0.5
        momentum = 0.65

        success_target = 0.2
        success_increase = 1.18
        success_decrease = 0.86

        # memory and archive
        mem = []
        mem_capacity = max(1, min(self.mem_size, dim))
        elite_X = [x.copy() for x, f in samples]  # historical elites (positions)
        elite_F = [f for x, f in samples]
        # keep recent samples separately for PCA
        recent_X = [x.copy() for x, f in samples]
        recent_F = [f for x, f in samples]

        stagnation_limit = max(8, int(25 + dim / 2))
        no_improve = 0
        prev_best = f_opt

        # trust-region parameters for surrogate proposals
        trust_radius = sigma * (1.0 + np.sqrt(dim))
        trust_shrink = 0.7
        trust_expand = 1.2
        trust_min = 1e-8 * span_mean

        # main loop
        while evals < budget:
            remaining = budget - evals

            # dynamic lam adaptation: expand population if stagnating, shrink when progressing
            if no_improve > 0 and no_improve % max(1, (stagnation_limit // 3)) == 0:
                lam = min(lam * 2, lam_max)
            else:
                lam = max(lam_min, int(lam * (0.999 if no_improve == 0 else 1.0)))
            if lam % 2 == 1:
                lam += 1
            lam = max(2, min(lam, remaining))  # cannot exceed remaining evals
            mu = max(1, lam // 2)
            weights = compute_weights(mu)

            half = lam // 2

            # generate mirrored samples with memory bias
            if len(mem) > 0:
                Z_core = self.rng.randn(half, dim)
                bias_strength = 0.6 / max(1.0, len(mem))
                mem_mat = np.array(mem)  # (k, dim)
                coeff = self.rng.randn(half, mem_mat.shape[0]) * bias_strength
                bias_part = coeff.dot(mem_mat)
                Z = Z_core + bias_part
            else:
                Z = self.rng.randn(half, dim)

            scales = sigma * np.sqrt(cov_diag + cov_eps)
            X = mean + np.vstack([Z, -Z]) * scales[np.newaxis, :]

            # clip
            np.clip(X, lb, ub, out=X)

            lam_eff = X.shape[0]
            fs = np.full(lam_eff, np.inf)
            X_evaluated = []
            Z_evaluated = []

            # sequentially evaluate candidates until budget
            for i in range(lam_eff):
                if evals >= budget:
                    break
                xi = X[i]
                fi = safe_eval(xi)
                if fi is None:
                    break
                fs[i] = fi
                X_evaluated.append(xi.copy())
                Z_evaluated.append((Z[i % half].copy() if i < half else (-Z[i - half].copy())))
                # improvement found -> cheap parabolic probe (budget-aware)
                if fi < f_opt - 1e-16:
                    f_opt = fi
                    x_opt = xi.copy()
                    elite_X.append(xi.copy())
                    elite_F.append(fi)
                    recent_X.append(xi.copy())
                    recent_F.append(fi)
                    # parabolic probe along xi - mean direction (small exploitation)
                    dir_vec = xi - mean
                    dnorm = np.linalg.norm(dir_vec)
                    if dnorm > 1e-12 and evals < budget:
                        dir_unit = dir_vec / dnorm
                        # probe further and closer points, but budget-aware: only one more probe if room
                        # choose a farther probe factor adaptively
                        probe_factor = 1.4 if remaining > 3 else 1.2
                        probe = xi + dir_unit * (dnorm * (probe_factor - 1.0))
                        np.clip(probe, lb, ub, out=probe)
                        fp = safe_eval(probe)
                        if fp is not None:
                            # simple quadratic fit using points (0,f_mean),(s,fi),(s2,fp)
                            s1 = dnorm
                            s2 = dnorm * probe_factor
                            y0 = f_mean
                            y1 = fi
                            y2 = fp
                            # build system as in previous algorithm with regularization guard
                            try:
                                A = np.array([[s1 * s1, s1, 1.0],
                                              [s2 * s2, s2, 1.0],
                                              [0.0, 0.0, 1.0]])
                                b = np.array([y1, y2, y0])
                                coeffs = np.linalg.solve(A, b)
                                a = coeffs[0]
                                b_lin = coeffs[1]
                                if abs(a) > 1e-16:
                                    s_opt = -b_lin / (2.0 * a)
                                    if 0.0 < s_opt < s2 * 1.2 and evals < budget:
                                        trial = mean + dir_unit * s_opt
                                        np.clip(trial, lb, ub, out=trial)
                                        ft = safe_eval(trial)
                                        if ft is not None and ft < f_opt:
                                            f_opt = ft
                                            x_opt = trial.copy()
                                            elite_X.append(trial.copy())
                                            elite_F.append(ft)
                            except Exception:
                                pass

                    # deposit direction into memory
                    if dnorm > 1e-12:
                        mem.append((xi - mean) / (dnorm))
                        mem = list(self._orthonormalize_memory(mem))
                        if len(mem) > mem_capacity:
                            mem = list(mem[:mem_capacity])

            if len(X_evaluated) == 0:
                break

            fs_arr = np.array([fs[i] for i in range(len(fs))])
            idx_sorted = np.argsort(fs_arr)[:len(X_evaluated)]
            top_idx = idx_sorted[:mu]
            X_top = np.array([X_evaluated[i] for i in top_idx])
            Z_top = np.array([Z_evaluated[i] for i in top_idx])
            f_top = fs_arr[top_idx]

            # ensure f_mean known
            if f_mean is None or not np.isfinite(f_mean):
                f_mean = safe_eval(mean)
                if f_mean is None:
                    f_mean = f_opt

            # recombine
            if mu == 1:
                new_mean = X_top[0].copy()
            else:
                new_mean = np.sum((weights[:, np.newaxis] * X_top), axis=0)

            # success measurement
            success_count = np.sum(f_top < f_mean - 1e-12)
            success_rate = float(success_count) / float(mu)

            # sigma adaptation with momentum smoothing
            adapt_factor = (success_increase if success_rate > success_target else success_decrease)
            sigma_new = sigma * adapt_factor
            sigma = float(max(sigma_min, min(sigma_max, momentum * sigma + (1.0 - momentum) * sigma_new)))

            # update diag covariance via mean step and memory contributions
            mean_step = new_mean - mean
            if np.any(np.abs(mean_step) > 0):
                normed = (mean_step / max(1e-12, sigma)) ** 2
                rank_contrib = np.zeros(dim)
                if len(mem) > 0:
                    for i_m, d in enumerate(mem):
                        w = rank_beta * (1.0 / (1 + i_m))
                        rank_contrib += w * (d ** 2)
                    # scale rank_contrib to sensible magnitude
                    if np.sum(rank_contrib) > 0:
                        rank_contrib = rank_contrib / np.sum(rank_contrib) * np.mean(rank_contrib + 1e-12) * len(mem)
                cov_diag = (1 - cov_beta) * cov_diag + cov_beta * (normed + rank_contrib + 1e-6)
                cov_diag = np.maximum(cov_diag, 1e-8)

            # update mean and evaluate it if budget allows (keep consistency)
            prev_mean = mean.copy()
            prev_f_mean = f_mean
            mean = new_mean.copy()
            if evals < budget:
                f_mean_new = safe_eval(mean)
                if f_mean_new is not None:
                    f_mean = f_mean_new
                    # record if improved
                    if f_mean < f_opt:
                        f_opt = f_mean
                        x_opt = mean.copy()
                        elite_X.append(mean.copy())
                        elite_F.append(f_mean)
                        recent_X.append(mean.copy())
                        recent_F.append(f_mean)
            else:
                f_mean = min(f_mean, f_opt)

            # If recombination improved mean, add to memory
            if f_mean < prev_f_mean - 1e-12:
                step = mean - prev_mean
                s_norm = np.linalg.norm(step)
                if s_norm > 1e-12:
                    mem.append(step / s_norm)
                    mem = list(self._orthonormalize_memory(mem))
                    if len(mem) > mem_capacity:
                        mem = list(mem[:mem_capacity])

            # maintain elite and recent pools
            if len(elite_F) == 0 or (x_opt is not None and f_opt < np.inf):
                if x_opt is not None and (len(elite_X) == 0 or not any(np.allclose(x_opt, ex) for ex in elite_X)):
                    elite_X.append(x_opt.copy())
                    elite_F.append(f_opt)
            # trim elite
            if len(elite_F) > self.max_elite:
                idxs = np.argsort(elite_F)[:self.max_elite // 2]
                elite_X = [elite_X[i] for i in idxs]
                elite_F = [elite_F[i] for i in idxs]
            # recent buffer trim
            if len(recent_F) > max(2 * dim, 100):
                recent_X = recent_X[-100:]
                recent_F = recent_F[-100:]

            # surrogate modeling in a low-rank subspace
            # proceed only if enough elite points and some budget left for trial
            attempt_model = (len(elite_F) >= self.model_min_points and evals + 2 < budget)
            if attempt_model:
                # Build subspace basis B: prioritize mem directions, then PCA on recent samples residuals
                B_rows = []
                mem_orth = self._orthonormalize_memory(mem)
                for r in mem_orth:
                    B_rows.append(r.copy())
                # add PCA components if space left
                k_needed = min(dim, max(1, mem_capacity + 2))
                if len(B_rows) < k_needed:
                    # center recent samples at current mean
                    Xmat = np.array(recent_X)
                    if Xmat.shape[0] >= 2:
                        diffs = (Xmat - mean)
                        # PCA via SVD of diffs
                        try:
                            U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
                            comps = Vt[:max(0, k_needed - len(B_rows))]
                            for row in comps:
                                # orthonormalize new comp against existing rows
                                v = row.copy()
                                for q in B_rows:
                                    v -= np.dot(q, v) * q
                                nrm = np.linalg.norm(v)
                                if nrm > 1e-12:
                                    B_rows.append(v / nrm)
                                if len(B_rows) >= k_needed:
                                    break
                        except Exception:
                            pass
                # build B (k x dim), want orthonormal rows; if none, fallback to standard basis directions subset
                if len(B_rows) == 0:
                    # choose top coordinate directions by cov_diag
                    idxs = np.argsort(-cov_diag)[:min(k_needed, dim)]
                    for i in idxs:
                        e = np.zeros(dim)
                        e[i] = 1.0
                        B_rows.append(e)
                # final orthonormalize to be safe
                B = self._orthonormalize_memory(B_rows)
                k = B.shape[0]
                if k > 0 and len(elite_F) >= max(self.model_min_points, k + 3):
                    # prepare design matrix for diagonal-quadratic model in subspace:
                    # f = a + b^T s + 0.5 * sum(c_i * s_i^2) + noise
                    # For each elite point compute s = B.dot(x - mean)
                    Xe = np.array(elite_X)
                    Fe = np.array(elite_F)
                    S = (B.dot((Xe - mean).T)).T  # shape (n_points, k)
                    n_pts = S.shape[0]
                    # Build design: [1, s1,...,sk, 0.5*s1^2,...,0.5*sk^2]
                    A = np.zeros((n_pts, 1 + k + k))
                    A[:, 0] = 1.0
                    A[:, 1:1 + k] = S
                    A[:, 1 + k:] = 0.5 * (S ** 2)
                    # regularized least-squares solve
                    lambda_reg = 1e-6 * (1.0 + np.var(Fe))
                    try:
                        ATA = A.T.dot(A) + np.eye(A.shape[1]) * lambda_reg
                        ATb = A.T.dot(Fe)
                        coeffs = np.linalg.solve(ATA, ATb)
                        a0 = coeffs[0]
                        b_vec = coeffs[1:1 + k]
                        c_vec = coeffs[1 + k:]
                        # propose optimum in subspace: solve gradient = b + c * s = 0 -> s = -b / c (elementwise)
                        # handle near-zero c by damping
                        denom = c_vec.copy()
                        small_mask = np.abs(denom) < 1e-12
                        denom[small_mask] = np.sign(denom[small_mask]) * 1e-12 + 1e-12
                        s_opt = -b_vec / denom
                        # limit step by trust radius in ambient space
                        x_candidate = mean + (B.T.dot(s_opt))
                        step_norm = np.linalg.norm(x_candidate - mean)
                        # scale-back if too large
                        tr = max(trust_min, trust_radius)
                        if step_norm > tr:
                            s_opt = s_opt * (tr / max(1e-12, step_norm))
                            x_candidate = mean + B.T.dot(s_opt)
                        # ensure within bounds
                        np.clip(x_candidate, lb, ub, out=x_candidate)
                        # evaluate candidate if we have budget
                        if evals < budget:
                            f_cand = safe_eval(x_candidate)
                            if f_cand is not None:
                                # if promising, accept and update trust radius
                                if f_cand < f_opt:
                                    f_opt = f_cand
                                    x_opt = x_candidate.copy()
                                    elite_X.append(x_candidate.copy())
                                    elite_F.append(f_cand)
                                    recent_X.append(x_candidate.copy())
                                    recent_F.append(f_cand)
                                    trust_radius = min(sigma * (1.0 + np.sqrt(dim)) * 4.0, trust_radius * trust_expand)
                                else:
                                    # if not improving, shrink trust region slightly
                                    trust_radius = max(trust_min, trust_radius * trust_shrink)
                    except Exception:
                        # if model fails, continue
                        pass

            # stagnation book-keeping
            if f_opt < prev_best - 1e-12:
                no_improve = 0
                prev_best = f_opt
                # gentle shrink of sigma on success
                sigma = max(sigma_min, sigma * 0.98)
                trust_radius = max(trust_min, trust_radius * 1.05)
            else:
                no_improve += 1
                # occasional Lévy jump when stagnating but we have budget
                if no_improve > (stagnation_limit // 3) and (self.rng.rand() < min(0.25, 0.02 * no_improve)) and evals + 1 < budget:
                    scale = span_mean * (1.0 + 0.5 * (no_improve / max(1, stagnation_limit)))
                    jump = self._levy_jump(scale)
                    candidate = mean + jump
                    np.clip(candidate, lb, ub, out=candidate)
                    fc = safe_eval(candidate)
                    if fc is not None and fc < f_opt:
                        f_opt = fc
                        x_opt = candidate.copy()
                        elite_X.append(candidate.copy())
                        elite_F.append(fc)
                        recent_X.append(candidate.copy())
                        recent_F.append(fc)
                        # moderate sigma increase to explore newly found basin
                        sigma = min(sigma_max, sigma * 2.0)

            # adaptive restart on long stagnation
            if no_improve >= stagnation_limit:
                no_improve = 0
                # choose new center from elites or random
                if len(elite_X) > 0:
                    pick = self.rng.randint(len(elite_X))
                    mean = elite_X[pick].copy() + self.rng.randn(dim) * 0.2 * span_mean
                elif x_opt is not None:
                    mean = x_opt.copy() + self.rng.randn(dim) * 0.2 * span_mean
                else:
                    mean = self.rng.uniform(lb, ub)
                np.clip(mean, lb, ub, out=mean)
                if evals < budget:
                    fm = safe_eval(mean)
                    if fm is not None:
                        f_mean = fm
                        if fm < f_opt:
                            f_opt = fm
                            x_opt = mean.copy()
                            elite_X.append(mean.copy())
                            elite_F.append(fm)
                # increase sigma and reset some covariance memory but keep primary directions
                sigma = min(sigma_max, max(sigma * 2.5, self.init_sigma * span_mean))
                cov_diag = cov_diag * 0.6 + 0.4 * np.ones_like(cov_diag)
                # preserve primary mem directions only
                if len(mem) > 0:
                    mem = list(self._orthonormalize_memory(mem)[:max(1, len(mem)//2)])
                else:
                    mem = []
                # widen population to explore after restart
                lam = min(lam_max, max(lam, 2 * lam))
                if lam % 2 == 1:
                    lam += 1
                # reset trust radius
                trust_radius = sigma * (1.0 + np.sqrt(dim))

            # enforce numeric bounds on sigma and cov_diag
            sigma = max(sigma_min, min(sigma, sigma_max))
            cov_diag = np.clip(cov_diag, 1e-8, 1e8)

            # final guard
            if evals >= budget:
                break

        return f_opt, x_opt