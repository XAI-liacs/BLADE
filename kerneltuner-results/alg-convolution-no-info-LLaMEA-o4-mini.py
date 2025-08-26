import random
import math

class TabuHarmonySearch(OptAlg):
    """
    Adaptive Tabu‐Harmony Search combining harmony memory,
    dynamic HMCR/PAR tuning, and tabu‐based move restriction.
    """
    def __init__(self,
                 budget=5000,
                 memory_size=10,
                 HMCR=0.7,
                 PAR=0.2,
                 HMCR_delta=0.3,
                 PAR_delta=0.3,
                 tabu_tenure=50,
                 stagnation_limit=100):
        # total budget of function evaluations
        self.budget = budget
        # harmony memory size
        self.memory_size = memory_size
        # initial Harmony Memory Considering Rate
        self.HMCR = HMCR
        # initial Pitch Adjustment Rate
        self.PAR = PAR
        # how much to increase HMCR and PAR over time
        self.HMCR_delta = HMCR_delta
        self.PAR_delta = PAR_delta
        # tabu tenure (iterations)
        self.tabu_tenure = tabu_tenure
        # stagnation threshold before random re‐seed
        self.stagnation_limit = stagnation_limit

    def __call__(self, func, searchspace):
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()
        self.methods = ["strictly-adjacent", "adjacent", "Hamming"]

        # initialize harmony memory with random valid samples
        mem = []
        for cfg in self.searchspace.get_random_sample(self.memory_size):
            if self.searchspace.is_param_config_valid(cfg):
                f = func(list(cfg))
                mem.append((list(cfg), f))
        # if some invalid or duplicates, top‐up
        while len(mem) < self.memory_size:
            cfg = self.searchspace.get_random_sample(1)[0]
            if self.searchspace.is_param_config_valid(cfg):
                f = func(list(cfg))
                mem.append((list(cfg), f))
        # sort by objective
        mem.sort(key=lambda x: x[1])
        # best so far
        best_cfg, best_f = mem[0]
        # tabu list: config tuple -> expiration iteration
        tabu = {}
        iteration = 0
        no_improve = 0

        # main loop
        while func.budget_spent_fraction < 1.0:
            iteration += 1
            frac = min(1.0, func.budget_spent_fraction)
            # dynamic HMCR and PAR
            HMCR_t = min(1.0, self.HMCR + self.HMCR_delta * frac)
            PAR_t  = min(1.0, self.PAR  + self.PAR_delta  * frac)

            # generate one new harmony
            # get a single random reference for all 'random' draws
            rand_ref = self.searchspace.get_random_sample(1)[0]
            dim = len(rand_ref)
            new_cfg = [None] * dim
            # memory consideration vs random
            for j in range(dim):
                if random.random() < HMCR_t:
                    # pick some memory harmony
                    pick = random.choice(mem)[0]
                    new_cfg[j] = pick[j]
                else:
                    # random from global sample
                    new_cfg[j] = rand_ref[j]

            # pitch adjustment: small neighbor hop
            if random.random() < PAR_t:
                method = random.choice(self.methods)
                nbrs = self.searchspace.get_neighbors(tuple(new_cfg), neighbor_method=method)
                # filter valid and non‐tabu
                valid = [list(nb) for nb in nbrs
                         if self.searchspace.is_param_config_valid(nb)
                         and tuple(nb) not in tabu]
                if valid:
                    new_cfg = random.choice(valid)

            # check tabu
            tup = tuple(new_cfg)
            if tup in tabu and tabu[tup] > iteration:
                # skip this harmony, count as no improvement
                no_improve += 1
            else:
                # evaluate
                f_new = func(new_cfg)
                # compare to worst in memory
                worst_cfg, worst_f = mem[-1]
                if f_new < worst_f:
                    # replace worst
                    replaced = mem.pop(-1)
                    mem.append((new_cfg, f_new))
                    # add replaced to tabu
                    tabu[tuple(replaced[0])] = iteration + self.tabu_tenure
                    # keep sorted
                    mem.sort(key=lambda x: x[1])
                    # update best
                    if f_new < best_f:
                        best_cfg, best_f = list(new_cfg), f_new
                        no_improve = 0
                    else:
                        # slight improvement in memory but not global
                        no_improve = 0
                else:
                    no_improve += 1

            # clean expired tabu entries
            tabu = {c: exp for c, exp in tabu.items() if exp > iteration}

            # if stagnated, reseed one harmony
            if no_improve >= self.stagnation_limit:
                # remove worst
                worst_cfg, worst_f = mem.pop(-1)
                tabu[tuple(worst_cfg)] = iteration + self.tabu_tenure
                # generate a fresh random valid
                while True:
                    candidate = self.searchspace.get_random_sample(1)[0]
                    if (self.searchspace.is_param_config_valid(candidate)
                        and tuple(candidate) not in tabu):
                        break
                f_cand = func(list(candidate))
                mem.append((list(candidate), f_cand))
                mem.sort(key=lambda x: x[1])
                if f_cand < best_f:
                    best_cfg, best_f = list(candidate), f_cand
                no_improve = 0

        return tuple(best_cfg), best_f