import random
import math

class ThompsonVNS(OptAlg):
    """
    Adaptive Variable-Neighborhood Search with Thompson Sampling and self-tuning Simulated Annealing.
    """

    def __init__(self, budget=5000, init_temp=1.0, final_temp=0.01):
        """
        budget:         maximum number of evaluations (stops when func.budget_spent_fraction hits 1.0)
        init_temp:      starting temperature for SA
        final_temp:     final temperature for SA
        """
        self.budget = budget
        self.init_temp = init_temp
        self.final_temp = final_temp
        # after this many non-improving steps, perform a random restart
        self.restart_limit = max(50, int(budget * 0.05))

    def __call__(self, func, searchspace):
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params

        # available neighborhood move types
        neighborhoods = ["strictly-adjacent", "adjacent", "Hamming"]
        # Initialize Thompson Sampling priors (alpha=1, beta=1 for each)
        alpha = {nm: 1.0 for nm in neighborhoods}
        beta  = {nm: 1.0 for nm in neighborhoods}

        # initialize with a random valid config
        x = list(searchspace.get_random_sample(1)[0])
        f_x = func(x)
        best_x, best_f = x.copy(), f_x
        no_improve = 0

        # main optimization loop
        while func.budget_spent_fraction < 1.0:
            frac = func.budget_spent_fraction
            # self‐tuning temperature: exponential schedule
            T = self.init_temp * ((self.final_temp / self.init_temp) ** frac)

            # Thompson Sampling: sample a belief for each neighborhood
            theta = {}
            for nm in neighborhoods:
                theta[nm] = random.betavariate(alpha[nm], beta[nm])
            # pick the neighborhood with highest sampled value
            nm = max(theta, key=theta.get)

            # sample one random neighbor in that neighborhood
            nbrs = self.searchspace.get_neighbors(tuple(x), neighbor_method=nm)
            if not nbrs:
                # penalize empty neighborhoods
                beta[nm] += 1.0
                continue

            cand = list(random.choice(nbrs))
            if not self.searchspace.is_param_config_valid(tuple(cand)):
                # invalid move
                beta[nm] += 1.0
                continue

            f_cand = func(cand)
            # acceptance test: always accept improvements, else SA probability
            delta = f_cand - f_x
            accept = (delta < 0) or (random.random() < math.exp(-delta / max(1e-8, T)))

            # update bandit
            if accept:
                alpha[nm] += 1.0
                x, f_x = cand, f_cand
                # track global best
                if f_cand < best_f:
                    best_x, best_f = x.copy(), f_x
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                beta[nm] += 1.0
                no_improve += 1

            # random restart if stuck
            if no_improve >= self.restart_limit:
                no_improve = 0
                x = list(searchspace.get_random_sample(1)[0])
                f_x = func(x)
                if f_x < best_f:
                    best_x, best_f = x.copy(), f_x

        return tuple(best_x), best_f