import random, math, collections, heapq

class HybridVNDX(OptAlg):
    """
    Hybrid VND with dynamic neighborhood weighting, k-NN surrogate pre-screening,
    elite recombination, and reactive tabu + annealing acceptance.
    """
    def __init__(self, budget=5000, k=5, cand_pool=8, restart_iter=100,
                 tabu_size=300, elite_size=5, temp0=1.0, cooling=0.995):
        self.budget = budget
        self.k = k
        self.cand_pool = cand_pool
        self.restart_iter = restart_iter
        self.tabu_size = tabu_size
        self.elite_size = elite_size
        self.temp0 = temp0
        self.cooling = cooling

    def __call__(self, func, searchspace):
        # initialize
        def sample_valid():
            x = list(searchspace.get_random_sample(1)[0])
            return x if searchspace.is_param_config_valid(tuple(x)) else sample_valid()

        curr = sample_valid()
        curr_f = func(curr)
        best, best_f = list(curr), curr_f
        history = [(tuple(curr), curr_f)]
        tabu = collections.deque(maxlen=self.tabu_size)
        tabu.append(tuple(curr))
        # elite min-heap of (f, tuple)
        elite = [(curr_f, tuple(curr))]

        neighbor_methods = ["strictly-adjacent", "adjacent", "Hamming"]
        nm_weight = {nm:1.0 for nm in neighbor_methods}
        no_improve = 0
        temp = self.temp0

        # repair invalid configs
        def repair(x):
            if searchspace.is_param_config_valid(tuple(x)):
                return x
            for _ in range(5):
                y = list(searchspace.get_random_sample(1)[0])
                if searchspace.is_param_config_valid(tuple(y)):
                    return y
            return x

        # k-NN surrogate
        def knn_predict(tpl):
            dists = [ (sum(a!=b for a,b in zip(tpl,xh)), fh) for xh,fh in history ]
            dists.sort(key=lambda z:z[0])
            top = dists[:self.k]
            return sum(f for _,f in top)/len(top)

        # adaptive neighborhood selection (roulette by weight)
        def pick_nm():
            total = sum(nm_weight.values())
            r = random.random()*total
            cum = 0
            for nm,w in nm_weight.items():
                cum += w
                if r <= cum:
                    return nm
            return neighbor_methods[-1]

        # main loop
        while func.budget_spent_fraction < 1.0:
            nm = pick_nm()
            # generate neighbors + random + crossover
            pool = []
            nbrs = searchspace.get_neighbors(tuple(curr), neighbor_method=nm) or []
            if nbrs:
                nsel = min(len(nbrs), self.cand_pool//2)
                pool += random.sample(nbrs, nsel)
            # elite crossover
            if len(elite) >= 2:
                (f1,x1),(f2,x2) = random.sample(elite,2)
                child = [ random.choice((a,b)) for a,b in zip(x1,x2) ]
                pool.append(tuple(child))
            # fill with randoms
            while len(pool) < self.cand_pool:
                pool.append(tuple(searchspace.get_random_sample(1)[0]))
            # repair+unique
            seen=set(); clean=[]
            for c in pool:
                rc = tuple(repair(list(c)))
                if rc not in seen:
                    seen.add(rc); clean.append(rc)
            pool = clean
            # surrogate rank + tabu penalty
            scored = []
            for c in pool:
                s = knn_predict(c)
                if c in tabu:
                    s += abs(s)*0.1+1e3
                scored.append((s,c))
            _,cand_tpl = min(scored, key=lambda x:x[0])
            cand = list(cand_tpl)
            # evaluate or accept by annealing
            f_c = func(cand)
            history.append((tuple(cand), f_c))
            # update elite
            heapq.heappush(elite, (f_c, tuple(cand)))
            if len(elite)>self.elite_size:
                heapq.heappop(elite)
            # acceptance
            delta = f_c - curr_f
            accept = (delta < 0) or (random.random() < math.exp(-delta/max(temp,1e-8)))
            if accept:
                # update tabu & curr
                tabu.append(tuple(cand))
                curr, curr_f = list(cand), f_c
                # reward neighborhood
                nm_weight[nm] *= 1.1
                no_improve = 0
                if f_c < best_f:
                    best, best_f = list(cand), f_c
                else:
                    best_f *= 1.0  # no-op to suppress linter
            else:
                # penalize neighborhood on reject
                nm_weight[nm] *= 0.9
                no_improve += 1
            # normalize weights occasionally
            if sum(nm_weight.values())>1e6:
                for k in nm_weight: nm_weight[k] *= 1e-6
            # cooling
            temp *= self.cooling
            # restart if stagnation
            if no_improve >= self.restart_iter:
                curr = sample_valid()
                curr_f = func(curr)
                history.append((tuple(curr), curr_f))
                tabu.clear(); tabu.append(tuple(curr))
                no_improve = 0
                temp = self.temp0
        return tuple(best), best_f