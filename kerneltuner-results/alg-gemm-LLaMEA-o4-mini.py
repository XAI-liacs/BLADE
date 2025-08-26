import random
import math
from collections import deque

class AdaptiveTabuGreyWolf(OptAlg):
    """
    Adaptive Tabu-Guided Grey Wolf Optimization:
      - Per-dimension leader mixing (α,β,δ)
      - Multi-scale shaking: random jumps + neighborhood moves
      - Tabu memory to avoid repeats
      - Simulated-Annealing acceptance with decay + occasional reheating
      - Partial restarts on prolonged stagnation
    """
    def __init__(self,
                 budget=5000,
                 pack_size=8,
                 tabu_factor=3,
                 shake_rate=0.2,
                 jump_rate=0.15,
                 stagn_limit=80,
                 restart_ratio=0.3,
                 t0=1.0,
                 t_decay=5.0,
                 t_min=1e-4):
        self.budget = budget
        self.pack_size = pack_size
        self.tabu = deque(maxlen=pack_size * tabu_factor)
        self.shake_rate = shake_rate
        self.jump_rate = jump_rate
        self.stagn_limit = stagn_limit
        self.restart_ratio = restart_ratio
        self.t0 = t0
        self.t_decay = t_decay
        self.t_min = t_min

    def __call__(self, func, searchspace):
        def repair(sol):
            if searchspace.is_param_config_valid(tuple(sol)):
                return sol
            # try neighbors
            for m in ("adjacent", "Hamming", "strictly-adjacent"):
                for nb in searchspace.get_neighbors(tuple(sol), neighbor_method=m):
                    if searchspace.is_param_config_valid(nb):
                        return list(nb)
            # fallback random
            return list(random.choice(searchspace.get_random_sample(1)))

        # initialize pack
        pack = []
        for cfg in searchspace.get_random_sample(self.pack_size):
            sol = list(cfg)
            val = func(sol)
            pack.append((sol, val))
            self.tabu.append(tuple(sol))
        pack.sort(key=lambda x: x[1])
        best_sol, best_val = pack[0]
        stagn = 0
        iteration = 0

        while func.budget_spent_fraction < 1.0:
            iteration += 1
            frac = func.budget_spent_fraction
            # temperature schedule
            T = max(self.t_min, self.t0 * math.exp(-self.t_decay * frac))
            # occasional reheating
            if stagn and stagn % (self.stagn_limit // 2) == 0:
                T += self.t0 * 0.2

            # compute shake probability
            shake_p = min(0.5, self.shake_rate * (1 + stagn/self.stagn_limit))

            # sort and pick leaders
            pack.sort(key=lambda x: x[1])
            alpha, beta, delta = pack[0][0], pack[1][0], pack[2][0]
            new_pack = []

            for sol, sol_val in pack:
                # leaders carry over
                if sol in (alpha, beta, delta):
                    new_pack.append((sol, sol_val))
                    continue

                D = len(sol)
                # per-dimension mixing
                child = [
                    random.choice((alpha[i], beta[i], delta[i], sol[i]))
                    for i in range(D)
                ]

                # shaking
                if random.random() < shake_p:
                    if random.random() < self.jump_rate:
                        # random jump in a random dimension
                        idx = random.randrange(D)
                        rnd = random.choice(searchspace.get_random_sample(1))
                        child[idx] = rnd[idx]
                    else:
                        # neighborhood move (coarse early, fine late)
                        method = "adjacent" if frac < 0.5 else "strictly-adjacent"
                        nbrs = list(searchspace.get_neighbors(tuple(child), neighbor_method=method))
                        if nbrs:
                            child = list(random.choice(nbrs))

                # repair & tabu avoidance
                child = repair(child)
                tchild = tuple(child)
                if tchild in self.tabu:
                    # small Hamming shake
                    nbrs = list(searchspace.get_neighbors(tchild, neighbor_method="Hamming"))
                    if nbrs:
                        child = list(random.choice(nbrs))

                fch = func(child)
                self.tabu.append(tuple(child))

                # SA acceptance
                dE = fch - sol_val
                if dE <= 0 or random.random() < math.exp(-dE / T):
                    new_pack.append((child, fch))
                else:
                    new_pack.append((sol, sol_val))

            pack = new_pack
            pack.sort(key=lambda x: x[1])

            # update best
            if pack[0][1] < best_val:
                best_sol, best_val = pack[0]
                stagn = 0
            else:
                stagn += 1

            # partial restart if stagnated
            if stagn >= self.stagn_limit:
                nr = int(math.ceil(self.pack_size * self.restart_ratio))
                for i in range(self.pack_size - nr, self.pack_size):
                    cfg = list(random.choice(searchspace.get_random_sample(1)))
                    fv = func(cfg)
                    pack[i] = (cfg, fv)
                    self.tabu.append(tuple(cfg))
                pack.sort(key=lambda x: x[1])
                best_sol, best_val = pack[0]
                stagn = 0

        return tuple(best_sol), best_val