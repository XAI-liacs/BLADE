import random
import math
from copy import deepcopy

class HierarchicalBanditVNS(OptAlg):
    """
    A memetic GA employing a hierarchical MAB to choose subspace sizes and neighbor methods.
    """
    def __init__(self, budget=5000, pop_size=12, crossover_rate=0.8,
                 base_mutation_rate=0.3, c_ucb=1.2, local_iter=15, stagnation_limit=60):
        self.budget = budget
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.base_mutation_rate = base_mutation_rate
        self.c_ucb = c_ucb
        self.local_iter = local_iter
        self.stagnation_limit = stagnation_limit

    def __call__(self, func, searchspace):
        self.searchspace = searchspace
        methods = ["strictly-adjacent", "adjacent", "Hamming"]
        dims = len(searchspace.tune_params)
        # choose a small set of subspace sizes
        subsizes = sorted(set([1, max(1, dims//4), max(1, dims//2), dims]))
        # build combined strategies
        strategies = [(s, m) for s in subsizes for m in methods]
        counts = {strat: 0 for strat in strategies}
        rewards = {strat: 0.0 for strat in strategies}
        total_uses = 0

        # initialize population
        pop = [list(p) for p in searchspace.get_random_sample(self.pop_size)]
        scores = [func(ind) for ind in pop]
        best_idx = min(range(len(scores)), key=lambda i: scores[i])
        x_opt, f_opt = tuple(pop[best_idx]), scores[best_idx]
        stagnation = 0

        def select_strategy(sol):
            nonlocal total_uses
            total_uses += 1
            # compute UCB value for each strat=(subsize,method)
            ucb_vals = {}
            for strat in strategies:
                cnt = counts[strat]
                if cnt == 0:
                    ucb_vals[strat] = float('inf')
                else:
                    avg = rewards[strat] / cnt
                    ucb_vals[strat] = avg + self.c_ucb * math.sqrt(math.log(total_uses) / cnt)
            strat = max(strategies, key=lambda s: ucb_vals[s])
            counts[strat] += 1
            s_size, method = strat
            neighbors = searchspace.get_neighbors(tuple(sol), neighbor_method=method)
            if not neighbors:
                return strat, []
            # if subspace < dims, filter neighbors that differ only in s_size dims
            if s_size < dims:
                # pick a random subspace for this call
                subset = set(random.sample(range(dims), s_size))
                filtered = []
                for n in neighbors:
                    diff = [i for i, (a,b) in enumerate(zip(sol, n)) if a!=b]
                    if set(diff).issubset(subset):
                        filtered.append(n)
                if filtered:
                    return strat, [list(n) for n in filtered]
            # fallback: return all
            return strat, [list(n) for n in neighbors]

        # main loop
        while func.budget_spent_fraction < 1.0:
            # 1) generate offspring via GA
            # tournament selection
            def tourney():
                k = min(3, len(pop))
                idxs = random.sample(range(len(pop)), k)
                return pop[min(idxs, key=lambda i: scores[i])]
            p1, p2 = tourney(), tourney()
            if random.random() < self.crossover_rate:
                child = [random.choice(pair) for pair in zip(p1, p2)]
            else:
                child = p1.copy()
            # mutation: hierarchical bandit strategy
            if random.random() < self.base_mutation_rate:
                strat, neighs = select_strategy(child)
                if neighs:
                    child = random.choice(neighs)
            # repair if invalid
            if not searchspace.is_param_config_valid(tuple(child)):
                # try a one-step neighbor in any method
                for m in methods:
                    nb = searchspace.get_neighbors(tuple(child), neighbor_method=m)
                    if nb:
                        child = list(nb[0])
                        break
            # evaluate
            score = func(child)
            # update best
            improved = False
            if score < f_opt:
                f_opt, x_opt = score, tuple(child)
                improved = True
                stagnation = 0
            else:
                stagnation += 1
            # reward to the last strat used
            if 'strat' in locals():
                reward = max(0.0, (f_opt - score) if improved else 0.0)
                rewards[strat] += reward
            # replace worst in population
            worst = max(range(len(scores)), key=lambda i: scores[i])
            if score < scores[worst]:
                pop[worst], scores[worst] = child, score

            # 2) periodic intensified local search on best
            if random.random() < 0.25 and func.budget_spent_fraction < 1.0:
                current, curr_score = list(x_opt), f_opt
                for _ in range(self.local_iter):
                    if func.budget_spent_fraction >= 1.0:
                        break
                    strat, neighs = select_strategy(current)
                    if not neighs:
                        continue
                    cand = random.choice(neighs)
                    s = func(cand)
                    if s < curr_score:
                        curr_score, current = s, cand
                        if s < f_opt:
                            f_opt, x_opt = s, tuple(cand)
                            improved = True
                        rewards[strat] += max(0.0, curr_score - s)
                    else:
                        rewards[strat] += 0.0

            # 3) hypermutation restart if stagnated
            if stagnation >= self.stagnation_limit and func.budget_spent_fraction < 1.0:
                k = max(1, self.pop_size // 2)
                new_samples = [list(p) for p in searchspace.get_random_sample(k)]
                for sol in new_samples:
                    sc = func(sol)
                    if sc < f_opt:
                        f_opt, x_opt = sc, tuple(sol)
                # replace worst k
                sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for i in range(k):
                    idx = sorted_idx[i]
                    pop[idx] = new_samples[i]
                    scores[idx] = func(new_samples[i])
                stagnation = 0

        return x_opt, f_opt