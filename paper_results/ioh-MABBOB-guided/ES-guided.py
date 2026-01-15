import numpy as np

class EntropyBoostedSubspaceAdaptiveSearch:
    """
    Entropy-Boosted Subspace Adaptive Search (EBSAS)

    Main ideas and novelties:
    - Monitor the eigenvalue (variance) spectrum entropy from a recent-step archive to measure
      directional diversity ("degree entropy"). Use this to adaptively tune sampling strategy:
        * Low entropy -> concentrate sampling in a smaller subspace (exploit)
        * High entropy -> broaden sampling (explore)
    - Multi-mode mirrored sampling: subspace-focused, orthogonal-only, global isotropic,
      coordinate/top-eigen perturbations, and amplified long jumps. Mode probabilities adapt
      based on measured entropy and stagnation to keep a high effective behavioral entropy.
    - Rank-mu covariance adaptation with importance-sampling correction and CMA-like paths.
    - Lightweight quasi-Newton / linear-probe steps in the learned subspace to accelerate local improvement.
    - Budget-aware population sizing and occasional small restarts when stagnation is detected.
    """

    def __init__(self, budget=10000, dim=10, seed=None,
                 init_sigma=None,
                 lambda_base=None,          # base population size
                 p_random=0.02,             # occasional global random probe
                 cov_eps=1e-8,
                 memory_size=120,
                 target_entropy=0.7,        # desired normalized entropy (0..1)
                 entropy_smooth=0.2):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.init_sigma = init_sigma

        if lambda_base is None:
            # small to moderate base population; will be scaled with dimension
            lambda_base = int(max(6, np.floor(4 + 3.0 * np.log(max(2, dim)))))
        self.lambda_base = max(4, int(lambda_base))

        self.p_random = float(p_random)
        self.cov_eps = float(cov_eps)
        self.memory_size = int(max(40, memory_size))

        self.target_entropy = float(np.clip(target_entropy, 0.01, 0.99))
        self.entropy_smooth = float(np.clip(entropy_smooth, 0.0, 1.0))

        # internal best
        self.f_opt = None
        self.x_opt = None

    def __call__(self, func):
        # expect func to have bounds .lb and .ub like in the framework
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        assert lb.shape[0] == self.dim
        assert ub.shape[0] == self.dim
        dim = self.dim

        # initialize center randomly and evaluate
        x_mean = self.rng.uniform(lb, ub)
        f_mean = float(func(x_mean))
        evals = 1

        # best overall
        x_best = x_mean.copy()
        f_best = f_mean

        # initialize sigma
        span = np.max(ub - lb)
        sigma = self.init_sigma if self.init_sigma is not None else 0.25 * np.mean(ub - lb)
        sigma = float(max(sigma, 1e-12))

        # initialize covariance
        C = np.eye(dim)
        def decompose_cov(Cmat):
            # robust eigen-decomposition ensuring positive definiteness
            vals, vecs = np.linalg.eigh((Cmat + Cmat.T) * 0.5)
            vals = np.where(vals > 1e-16, vals, 1e-16)
            sqrtvals = np.sqrt(vals)
            inv_sqrtvals = 1.0 / sqrtvals
            B = (vecs * sqrtvals).dot(vecs.T)          # sqrt(C)
            invB = (vecs * inv_sqrtvals).dot(vecs.T)  # inv_sqrt(C)
            return B, invB, vals, vecs

        B, invB, eigvals, eigvecs = decompose_cov(C)

        # population weighting (rank-mu like)
        lam = max(4, self.lambda_base)
        # ensure even for mirrored sampling
        if lam % 2 == 1:
            lam += 1
        mu = max(1, lam // 2)
        ranks = np.arange(1, mu + 1)
        weights = np.log(mu + 0.5) - np.log(ranks)
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # CMA-like adaptation constants
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))
        c1 = max(c1, 1e-6)
        cmu = max(cmu, 1e-6)

        # evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # archive for PCA, step statistics, and probes
        archive_X = [x_mean.copy()]
        archive_f = [f_mean]
        step_archive = []  # store recent successful step vectors (x_new - x_old)

        generation = 0
        stagnation = 0
        stagnation_limit = max(30, 6 * dim)

        # expected length of N(0,I)
        expected_norm = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # initialize entropy estimate
        entropy_est = 1.0  # normalized (0..1)

        # helper: compute normalized eigenvalue entropy from an empirical covariance
        def normalized_entropy(eigvals_arr):
            vals = np.array(eigvals_arr, dtype=float)
            vals = np.where(vals > 0, vals, 1e-16)
            p = vals / np.sum(vals)
            H = -np.sum(p * np.log(p + 1e-16)) / np.log(len(p) + 1e-16)
            return float(np.clip(H, 0.0, 1.0))

        # main optimization loop
        while evals < self.budget:
            generation += 1

            # occasional purely random global probe (forced exploration)
            if self.rng.rand() < self.p_random and evals < self.budget:
                x_probe = self.rng.uniform(lb, ub)
                f_probe = float(func(x_probe))
                evals += 1
                archive_X.append(x_probe.copy()); archive_f.append(f_probe)
                if len(archive_X) > self.memory_size:
                    archive_X.pop(0); archive_f.pop(0)
                if f_probe < f_best:
                    x_best, f_best = x_probe.copy(), f_probe
                # maybe re-center to a good probe if it improves mean
                if f_probe < f_mean:
                    x_mean, f_mean = x_probe.copy(), f_probe
                # continue but still allow regular generation below

            # Build adaptive subspace from successful steps in step_archive and archive
            use_for_pca = min(len(archive_X), self.memory_size)
            if use_for_pca >= 2:
                Xmat = np.array(archive_X[-use_for_pca:])
                centered = (Xmat - np.mean(Xmat, axis=0))
                # compute empirical covariance (robust)
                cov_est = np.cov(centered, rowvar=False) + 1e-12 * np.eye(dim)
                vvals, vvecs = np.linalg.eigh(cov_est)
                idx = np.argsort(vvals)[::-1]
                vvals = vvals[idx]; vvecs = vvecs[:, idx]
                # normalized eigenvalue entropy
                H = normalized_entropy(vvals)
                # smooth entropy estimate
                entropy_est = (1.0 - self.entropy_smooth) * entropy_est + self.entropy_smooth * H
                # choose subspace dimension k based on entropy: lower entropy -> smaller k
                # map entropy in [0,1] to k in [1, dim]
                k = int(np.clip(np.round((0.5 + 0.8 * entropy_est) * dim), 1, dim))
                # we bias to capture at least top variance directions but allow adaptive k
                sub_basis = vvecs[:, :k]
            else:
                # not enough data: default small subspace
                entropy_est = 1.0
                k = max(1, int(0.25 * dim))
                sub_basis = np.eye(dim)[:, :k]

            # adapt sampling mode probabilities to encourage maintenance of entropy
            # Modes: 'sub' (focused), 'orth' (complement), 'global', 'coord' (top-eigen axes), 'jump' (amplified)
            # If entropy_est < target, encourage orth/global to increase entropy; otherwise encourage subspace exploitation.
            p_target = self.target_entropy
            # base probabilities
            p_sub = 0.35
            p_orth = 0.20
            p_global = 0.15
            p_coord = 0.15
            p_jump = 0.15
            # adjust based on difference
            diff = entropy_est - p_target
            # if entropy below target, increase orth/global/jump to inject diversity
            if diff < 0:
                boost = min(0.5, -diff)
                p_orth += 0.4 * boost
                p_global += 0.3 * boost
                p_jump += 0.3 * boost
                p_sub = max(0.05, p_sub * (1 - 0.8 * boost))
            else:
                # entropy high -> concentrate more in subspace
                boost = min(0.6, diff)
                p_sub = min(0.9, p_sub + 0.6 * boost)
                p_coord = max(0.05, p_coord * (1 - 0.5 * boost))
            # normalize probabilities
            probs = np.array([p_sub, p_orth, p_global, p_coord, p_jump], dtype=float)
            probs = np.maximum(probs, 1e-6)
            probs = probs / np.sum(probs)

            # population size scaling with remaining budget & dimension (budget-aware)
            remaining = max(1, self.budget - evals)
            lam = self.lambda_base
            # scale up lambda moderately if many evaluations remain and dim smallish
            if remaining > 10 * lam and dim <= 50:
                lam = int(min(4 * self.lambda_base, lam + remaining // 200))
            if lam % 2 == 1:
                lam += 1
            half = lam // 2

            # generate half independent z then mirror; but we might need fewer candidates due to budget
            z_list = [self.rng.normal(size=dim) for _ in range(half)]
            cand_list = []
            z_store = []
            modes_store = []

            for z in z_list:
                mode = self.rng.choice(5, p=probs)
                # transform base z via B (sqrt(C)) later; we adjust z components
                if mode == 0:  # subspace-focused
                    proj = sub_basis.dot(sub_basis.T.dot(z))
                    off = z - proj
                    # amplify subspace, shrink orthogonal
                    z_new = proj * (1.8 + 0.6 * self.rng.rand()) + off * (0.3 + 0.4 * self.rng.rand())
                elif mode == 1:  # orthogonal exploration
                    proj = sub_basis.dot(sub_basis.T.dot(z))
                    off = z - proj
                    # emphasize orthogonal component
                    if np.linalg.norm(off) < 1e-12:
                        # fallback: perturb random orth direction
                        off = z.copy()
                    z_new = off * (1.6 + 0.8 * self.rng.rand()) + proj * (0.2 + 0.3 * self.rng.rand())
                elif mode == 2:  # global isotropic
                    z_new = z * (0.8 + 0.8 * self.rng.rand())
                elif mode == 3:  # coordinate / eigen-axis perturbation (sparse)
                    # perturb only along one or a few top eigenvectors of current C
                    ntop = max(1, min(k, int(1 + self.rng.randint(0, min(4, k)))))
                    axes = eigvecs[:, :k]
                    coeffs = np.zeros(k)
                    idxs = self.rng.choice(k, size=ntop, replace=False)
                    coeffs[idxs] = self.rng.normal(scale=2.0, size=ntop)
                    z_new = axes.dot(coeffs)
                    # add small noise in orth directions
                    z_new = z_new + 0.1 * z
                else:  # jump (long-range)
                    # amplify whole vector, occasionally flip direction to push far
                    z_new = z * (2.5 + 2.0 * self.rng.rand())
                    if self.rng.rand() < 0.1:
                        z_new = -z_new

                # whiten (approximately) using invB to keep sampling scale consistent
                # We'll apply B later when generating actual candidate
                y = B.dot(z_new)
                cand_list.append(x_mean + sigma * y)
                z_store.append(z_new)
                modes_store.append(mode)
                # mirrored
                cand_list.append(x_mean - sigma * y)
                z_store.append(-z_new)
                modes_store.append(mode)

                if len(cand_list) >= remaining:
                    break

            # trim to budget
            to_eval = min(len(cand_list), remaining)
            cand_list = cand_list[:to_eval]
            z_store = z_store[:to_eval]
            modes_store = modes_store[:to_eval]

            # Evaluate candidates in an order that prioritizes diversity: interleave modes
            # Create indices grouped per mode and sample one from each group in round-robin
            mode_indices = {}
            for i, m in enumerate(modes_store):
                mode_indices.setdefault(m, []).append(i)
            interleaved = []
            pointers = {m: 0 for m in mode_indices}
            while len(interleaved) < len(cand_list):
                for m in list(mode_indices.keys()):
                    p = pointers[m]
                    if p < len(mode_indices[m]):
                        interleaved.append(mode_indices[m][p])
                        pointers[m] = p + 1
                    if len(interleaved) >= len(cand_list):
                        break
            # Evaluate in interleaved order
            cand_X = []
            cand_f = []
            for idx in interleaved:
                x_cand = cand_list[idx]
                x_c = np.minimum(np.maximum(x_cand, lb), ub)
                f_c = float(func(x_c))
                evals += 1
                cand_X.append(x_c.copy()); cand_f.append(f_c)
                archive_X.append(x_c.copy()); archive_f.append(f_c)
                if len(archive_X) > self.memory_size:
                    archive_X.pop(0); archive_f.pop(0)
                # store step vectors for successful improvements relative to mean
                step = x_c - x_mean
                step_archive.append(step.copy())
                if len(step_archive) > self.memory_size:
                    step_archive.pop(0)
                if f_c < f_best:
                    # accept best immediately
                    x_best, f_best = x_c.copy(), f_c
                if evals >= self.budget:
                    break

            # selection: rank by fitness among candidates
            if len(cand_f) == 0:
                break
            idx_sorted = np.argsort(cand_f)
            top_count = min(mu, len(idx_sorted))
            top_idx = idx_sorted[:top_count]

            # importance-sampling weights: prefer diversity-weighted contributions
            ys = []
            weighted_shift = np.zeros(dim)
            is_weights = []
            for rank_j, jj in enumerate(top_idx):
                # base recombination weight (shape)
                w = weights[rank_j] if rank_j < len(weights) else 0.0
                # diversity bonus: penalize candidates too close to mean to encourage exploration
                dist = np.linalg.norm((cand_X[jj] - x_mean) / (sigma + 1e-12))
                diversity_bonus = 1.0 + 0.3 * np.tanh(dist / (1.0 + 0.5 * dim ** 0.5))
                w_eff = w * diversity_bonus
                y = (cand_X[jj] - x_mean) / (sigma + 1e-16)
                ys.append((w_eff, y))
                weighted_shift += w_eff * (cand_X[jj] - x_mean)
                is_weights.append(w_eff)
            wsum = sum(is_weights) if len(is_weights) > 0 else 0.0
            if wsum > 0:
                delta_mean = weighted_shift / (wsum + 1e-16)
            else:
                delta_mean = np.zeros(dim)

            # update mean and approximate mean fitness conservatively
            x_mean = x_mean + delta_mean
            f_mean = min([cand_f[jj] for jj in top_idx]) if len(top_idx) > 0 else f_mean

            # evolution path and sigma adaptation
            z_for_ps = invB.dot(delta_mean / (sigma + 1e-16))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_for_ps
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / (expected_norm + 1e-16) - 1.0))

            # conjugate path for covariance
            denom = np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * generation))
            if denom <= 0:
                denom = 1e-8
            h_sig = 1.0 if (norm_p_sigma / denom) < (1.4 + 2.0 / (dim + 1.0)) * expected_norm else 0.0
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (delta_mean / (sigma + 1e-16))

            # compute rank-mu update with importance-sampling normalization
            rank_mu = np.zeros((dim, dim))
            # normalize importance weights so that sum(weights) == 1 for covariance update
            if len(ys) > 0:
                wsum_norm = sum([abs(w) for w, _ in ys]) + 1e-16
                for w, y in ys:
                    w_norm = w / wsum_norm
                    rank_mu += w_norm * np.outer(y, y)

            # covariance update with small damping when h_sig==0 (to handle bad signals)
            damp = 0.5 if h_sig < 0.5 else 1.0
            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) * damp + cmu * rank_mu
            # regularize
            C += self.cov_eps * np.eye(dim)

            # re-decompose every generation (cheap for modest dims)
            B, invB, eigvals, eigvecs = decompose_cov(C)

            # update eigenvalue-based entropy estimate and influence sampling next loop done above

            # local quasi-Newton / gradient probe every few gens if budget allows
            if generation % max(3, int(5 + dim / 10)) == 0 and evals < self.budget and len(archive_X) >= min(8, dim):
                # build linear model in current subspace using recent archive
                recent_k = min(len(archive_X), self.memory_size)
                Xr = np.array(archive_X[-recent_k:])
                fr = np.array(archive_f[-recent_k:])
                Xc = Xr - np.mean(Xr, axis=0)
                fr_c = fr - np.mean(fr)
                # project to subspace
                Z = Xc.dot(sub_basis)  # Nxk
                if Z.shape[1] >= 1 and Z.shape[0] >= Z.shape[1]:
                    # ridge least squares to get gradient approx in subspace
                    reg = 1e-6 * max(1.0, np.linalg.norm(fr_c))
                    try:
                        A = Z.T.dot(Z) + reg * np.eye(Z.shape[1])
                        b = Z.T.dot(fr_c)
                        sol = np.linalg.solve(A, b)
                        grad_sub = sub_basis.dot(sol)  # approximate gradient in original space
                        gnorm = np.linalg.norm(grad_sub)
                        if gnorm > 1e-12:
                            step_dir = -grad_sub / gnorm
                            step_size = 0.9 * sigma * np.sqrt(dim) * (1.0 + 0.5 * self.rng.rand())
                            probe = x_mean + step_dir * step_size
                            probe = np.minimum(np.maximum(probe, lb), ub)
                            # evaluate probe
                            f_probe = float(func(probe))
                            evals += 1
                            archive_X.append(probe.copy()); archive_f.append(f_probe)
                            if len(archive_X) > self.memory_size:
                                archive_X.pop(0); archive_f.pop(0)
                            # incorporate successful probe
                            if f_probe < f_best:
                                step_vec = probe - x_mean
                                s2 = np.dot(step_vec, step_vec)
                                if s2 > 1e-16:
                                    rank_one = np.outer(step_vec, step_vec) / (s2 + 1e-16)
                                    C = (1 - 0.5 * cmu) * C + (0.5 * cmu) * rank_one
                                    B, invB, eigvals, eigvecs = decompose_cov(C)
                                x_best, f_best = probe.copy(), f_probe
                            # small chance to re-center mean if probe is good
                            if f_probe < f_mean:
                                x_mean, f_mean = probe.copy(), f_probe
                    except Exception:
                        pass

            # stagnation handling
            if f_best < f_mean:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation > stagnation_limit:
                stagnation = 0
                # diversify by re-centering around a good archived point or making a randomized restart
                if len(archive_X) > 0 and self.rng.rand() < 0.8:
                    # pick among top archived points
                    top_count = min(len(archive_f), 10)
                    idxs = np.argsort(archive_f)[:top_count]
                    pick = int(self.rng.choice(idxs))
                    x_mean = archive_X[pick].copy()
                    f_mean = archive_f[pick]
                else:
                    x_mean = self.rng.uniform(lb, ub)
                    f_mean = float(func(x_mean))
                    evals += 1
                    archive_X.append(x_mean.copy()); archive_f.append(f_mean)
                    if f_mean < f_best:
                        x_best, f_best = x_mean.copy(), f_mean
                # inflate sigma to encourage exploration
                sigma = min(0.8 * span, sigma * (1.5 + 0.5 * self.rng.rand()))
                # slightly isotropize covariance to avoid becoming too narrow
                C = 0.5 * C + 0.5 * np.eye(dim)
                B, invB, eigvals, eigvecs = decompose_cov(C)

            # track global best
            self.f_opt = float(f_best)
            self.x_opt = x_best.copy()

        return float(self.f_opt), self.x_opt