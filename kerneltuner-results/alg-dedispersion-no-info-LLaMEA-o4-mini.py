import math, random
import numpy as np

class AdaptiveBanditNeighborhoodSearch(OptAlg):
    """
    Adaptive Bandit Neighborhood Search:
    Combines simulated annealing with UCB-driven neighborhood selection over multiple neighbor operators,
    and uses periodic random restarts when stuck.
    """

    def __init__(self, budget=5000, init_temp=1.0, cooling_rate=0.995, restart_thresh=100):
        self.max_budget = budget
        self.init_temp = init_temp
        self.cooling_rate = cooling_rate
        self.restart_thresh = restart_thresh

    def __call__(self, func, searchspace):
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()

        # initialize best
        self.f_opt = float('inf')
        self.x_opt = None

        # neighborhood operators
        neigh_ops = ["strictly-adjacent", "adjacent", "Hamming"]
        counts = {op: 0 for op in neigh_ops}
        rewards = {op: 0.0 for op in neigh_ops}

        def select_op(t):
            # UCB1 selection
            total = sum(counts.values()) or 1
            ucb_scores = {}
            for op in neigh_ops:
                if counts[op] == 0:
                    return op
                avg = rewards[op] / counts[op]
                bonus = math.sqrt(2 * math.log(total) / counts[op])
                ucb_scores[op] = avg + bonus
            return max(ucb_scores, key=ucb_scores.get)

        def get_neighbor(sol, op):
            neigh = self.searchspace.get_neighbors(tuple(sol), neighbor_method=op)
            if not neigh:
                return None
            cand = list(random.choice(neigh))
            if not self.searchspace.is_param_config_valid(tuple(cand)):
                return None
            return cand

        # initial solution
        current = list(searchspace.get_random_sample(1)[0])
        f_current = func(current)
        if f_current < self.f_opt:
            self.f_opt, self.x_opt = f_current, tuple(current)

        temperature = self.init_temp
        no_improve = 0
        iteration = 0

        while func.budget_spent_fraction < 1.0:
            iteration += 1
            op = select_op(iteration)
            neighbor = get_neighbor(current, op)
            if neighbor is None:
                # restart if no valid neighbor
                neighbor = list(searchspace.get_random_sample(1)[0])
            f_neighbor = func(neighbor)

            delta = f_neighbor - f_current
            accept = (delta < 0) or (random.random() < math.exp(-delta / max(1e-8, temperature)))
            counts[op] += 1
            if accept:
                current, f_current = neighbor, f_neighbor
                reward = 1.0 if delta < 0 else 0.1
                rewards[op] += reward
                no_improve = 0
                if f_current < self.f_opt:
                    self.f_opt, self.x_opt = f_current, tuple(current)
            else:
                rewards[op] -= 0.05  # small penalty
                no_improve += 1

            temperature *= self.cooling_rate

            if no_improve >= self.restart_thresh:
                current = list(searchspace.get_random_sample(1)[0])
                f_current = func(current)
                no_improve = 0

        return self.x_opt, self.f_opt