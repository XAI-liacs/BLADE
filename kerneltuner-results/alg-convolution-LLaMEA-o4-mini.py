import random
import math
import numpy as np
from collections import deque

class PRTS(OptAlg):
    """
    PRTS: Pheromone‐reinforced Tabu Search
    Combines UCB‐guided multi‐neighborhood exploration, a short tabu memory
    to avoid cycles, and pheromone‐based restarts sampling promising values
    from an elite archive.
    """
    def __init__(self, budget=5000,
                 tabu_size=50,
                 elite_size=5,
                 no_improve_limit=100,
                 ucb_c=1.0,
                 epsilon=0.1):
        self.budget = budget
        self.tabu_size = tabu_size
        self.elite_size = elite_size
        self.no_improve_limit = no_improve_limit
        self.ucb_c = ucb_c
        self.epsilon = epsilon

    def __call__(self, func, searchspace):
        # extract parameter ordering
        param_names = list(searchspace.tune_params.keys())
        methods = ["Hamming", "adjacent", "strictly-adjacent"]
        M = len(methods)

        # UCB statistics
        counts  = np.ones(M, dtype=float)   # one initial pull each
        rewards = np.zeros(M, dtype=float)

        # initial solution
        current = list(searchspace.get_random_sample(1)[0])
        f_curr = func(current)
        best, f_best = list(current), f_curr

        # elite archive: list of (config_tuple, score), sorted by score
        elites = [(tuple(current), f_curr)]

        # tabu memory for configs
        tabu = deque(maxlen=self.tabu_size)
        tabu.append(tuple(current))

        no_improve = 0

        def pheromone_restart():
            # build a restart candidate by sampling each param from elite frequencies
            if elites and random.random() < 0.7:
                # sample from pheromone
                cfg = []
                for idx in range(len(param_names)):
                    vals = [e[0][idx] for e in elites]
                    uniq, cnts = np.unique(vals, return_counts=True)
                    probs = cnts / cnts.sum()
                    choice = np.random.choice(uniq, p=probs)
                    cfg.append(choice)
                tcfg = tuple(cfg)
                if searchspace.is_param_config_valid(tcfg):
                    return list(cfg)
            # fallback to a uniform random sample
            return list(searchspace.get_random_sample(1)[0])

        # main loop
        while func.budget_spent_fraction < 1.0:
            # select neighborhood operator by UCB or random exploration
            if random.random() < self.epsilon:
                m = random.randrange(M)
            else:
                ucb_scores = rewards / counts + self.ucb_c * np.sqrt(
                    np.log(counts.sum() + 1e-12) / counts
                )
                m = int(np.argmax(ucb_scores))

            # generate candidate
            nbrs = searchspace.get_neighbors(tuple(current),
                                             neighbor_method=methods[m])
            valid = [n for n in nbrs
                     if searchspace.is_param_config_valid(n)
                     and n not in tabu]
            if valid:
                candidate = list(random.choice(valid))
            else:
                candidate = pheromone_restart()

            f_new = func(candidate)
            counts[m] += 1

            # accept only improvements
            if f_new < f_curr:
                rewards[m] += 1
                current, f_curr = candidate, f_new
                no_improve = 0

                # global best update
                if f_new < f_best:
                    best, f_best = list(candidate), f_new

                # update elite archive
                tup = tuple(candidate)
                # insert if archive not full or better than worst
                if (len(elites) < self.elite_size or
                    f_new < elites[-1][1]) and tup not in [e[0] for e in elites]:
                    elites.append((tup, f_new))
                    elites = sorted(elites, key=lambda x: x[1])[:self.elite_size]
            else:
                no_improve += 1

            # update tabu
            tabu.append(tuple(candidate))

            # stagnation → diversify
            if no_improve >= self.no_improve_limit:
                current = pheromone_restart()
                f_curr = func(current)
                no_improve = 0
                tabu.clear()
                tabu.append(tuple(current))

        return tuple(best), f_best