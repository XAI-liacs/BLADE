import random, math

class AdaptiveLevySA(OptAlg):
    """
    Adaptive Lévy‐Flight Simulated Annealing:
    Exponential cooling + dynamic global jump probability + multi‐neighbor local search.
    """

    def __init__(self, budget=5000, T_init=1.0, T_final=1e-3,
                 p0=0.5, beta=1.5, local_k=3, stagnate=50):
        """
        budget    : approximate number of iterations
        T_init    : initial temperature
        T_final   : final temperature
        p0        : initial probability of global Lévy jump
        beta      : Pareto exponent (>1) for Lévy step‐sizes
        local_k   : number of local neighbors to sample each iteration
        stagnate  : iterations without improvement before restart
        """
        self.budget = budget
        self.T0 = T_init
        self.Tf = T_final
        self.p0 = p0
        self.beta = beta
        self.local_k = local_k
        self.stagnate = stagnate

    def __call__(self, func, searchspace):
        sp = searchspace
        # initial sample
        x = list(sp.get_random_sample(1)[0])
        f_x = func(x)
        best, f_best = x[:], f_x

        dim = len(x)
        it = 0
        no_imp = 0
        max_it = max(1, int(self.budget))

        while func.budget_spent_fraction < 1.0:
            # temperature: exponential schedule
            frac = it / max_it
            T = self.T0 * (self.Tf / self.T0) ** frac
            # dynamic jump probability: decays to zero
            p_jump = self.p0 * (1 - frac)

            # propose
            if random.random() < p_jump:
                # Lévy‐flight global jump
                y = list(sp.get_random_sample(1)[0])
                # sample Pareto step‐size k
                u = random.random()
                k = int((u ** (-1.0 / self.beta)))
                k = max(1, min(k, dim))
                idxs = random.sample(range(dim), k)
                cand = x[:]
                for i in idxs:
                    cand[i] = y[i]
            else:
                # local exploitation: sample several neighbors, pick best
                best_local = None
                best_floc = float('inf')
                for _ in range(self.local_k):
                    neigh = sp.get_neighbors(tuple(x), neighbor_method="Hamming") or []
                    if not neigh:
                        # fallback: tweak one dimension
                        j = random.randrange(dim)
                        tmp = x[:]
                        tmp[j] = sp.get_random_sample(1)[0][j]
                    else:
                        tmp = list(random.choice(neigh))
                    if not sp.is_param_config_valid(tuple(tmp)):
                        continue
                    f_tmp = func(tmp)
                    if f_tmp < best_floc:
                        best_floc, best_local = f_tmp, tmp
                if best_local is None:
                    # fallback to random valid
                    best_local = list(sp.get_random_sample(1)[0])
                cand = best_local
                f_cand = best_floc if 'best_floc' in locals() and best_local else None

            # ensure validity
            if not sp.is_param_config_valid(tuple(cand)):
                # quick repair: find any valid neighbor
                for method in ["Hamming", "adjacent", "strictly-adjacent"]:
                    neigh = sp.get_neighbors(tuple(cand), neighbor_method=method) or []
                    valid = [n for n in neigh if sp.is_param_config_valid(n)]
                    if valid:
                        cand = list(random.choice(valid))
                        break

            # evaluate if not yet
            if 'f_cand' not in locals() or locals().get('f_cand') is None:
                f_cand = func(cand)

            # acceptance criterion
            delta = f_cand - f_x
            if delta < 0 or random.random() < math.exp(-delta / max(T,1e-12)):
                x, f_x = cand, f_cand
                no_imp += 1
                if f_x < f_best:
                    best, f_best = x[:], f_x
                    no_imp = 0
            else:
                no_imp += 1

            # restart if stagnated
            if no_imp >= self.stagnate:
                xr = list(sp.get_random_sample(1)[0])
                if sp.is_param_config_valid(tuple(xr)):
                    x, f_x = xr, func(xr)
                no_imp = 0

            it += 1
            # clear f_cand flag
            if 'f_cand' in locals(): del f_cand

        return tuple(best), f_best