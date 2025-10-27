# Matrix-Multiplication Benchmark (LLaMEA)

This folder benchmarks LLaMEA on **low-rank CP decompositions of the
〈n,m,p〉 matrix-multiplication tensor**.  A perfect rank-`r` decomposition
corresponds to an algorithm that multiplies an `n×m` matrix by an
`m×p` matrix with **exactly `r` scalar multiplications**.

---

## 1  Problem statement

For given dimensions

```
n × m   ·   m × p   ⟶   n × p
```

the tensor
`T ∈ ℝ^(n·m  ×  m·p  ×  p·n)` encodes ordinary matrix multiplication.
A rank-`r` CP decomposition

```
T[i,j,k] = Σ_{ℓ=1..r} F1[i,ℓ] · F2[j,ℓ] · F3[k,ℓ]
```

yields an algorithm that needs only `r` scalar multiplies.
Our goal: **find the smallest r with zero reconstruction error**.

All factor entries must lie on a **quantisation grid**
(`grid = 0.5`, `1.0`, …) to keep solutions exact and interpretable.

---

## 2  How a candidate is evaluated

File [`get_evaluator.py`](get_evaluator.py) supplies
`get_evaluator()` which LLaMEA calls for every generated solution:

1. **Sandboxed execution**
   The candidate Python code is executed with `exec()` using a limited
   `safe_globals` whitelist.

2. **Expected return value**
   A callable class must return either
   * a flat NumPy-compatible vector, or
   * a list `[F1, F2, F3]` (each as nested lists).

3. **Post-processing**
   • Values are snapped to the grid.
   • The flat vector is reshaped into
   `F1 (n·m × r)`, `F2 (m·p × r)`, `F3 (p·n × r)`.

4. **Error computation**

   ```python
   try:
       verify_tensor_decomposition((F1,F2,F3), n,m,p,r)   # exact test
       err = 0.0
   except AssertionError:
       T_hat = einsum("il,jl,kl->ijk", F1, F2, F3)
       err   = ‖T_hat – T_target‖_F
   ```

   • `verify_tensor_decomposition()` (see [`verify.py`](verify.py))
     checks both shapes and the exact tensor equality.
   • If exact equality fails, the Frobenius norm of the residual is the
     error.

5. **Fitness**
   `Solution.set_scores(err, msg)` → LLaMEA minimises `err`.

---

## 3  How runs are orchestrated

This repository uses the generic driver [`src/main.py`](../../main.py) and a
benchmark registry to run problems. For matrix multiplication:

1. **Problems and ranks (decremental)**
   [`problem_sets.py`](problem_sets.py) defines `default_problems()` and
   sets the starting rank for each size via `_max_rank_minus_1(n,m,p)`, which
   looks up the literature best-known rank from
   [`constants.py`](constants.py) when available, i.e.
   `MATMUL_RANKS[(m,n,p)]['best_known']`.
   The main driver (`src/main.py`) always performs a decremental rank search:
   it runs rank r, and if an exact solution is found (within tol = 1e-12), it
   tries rank r−1; otherwise it repeats rank r until the budget is exhausted.

2. **Budget and execution**
   `main.py` constructs LLaMEA with the provided `budget` and runs once per
   problem. The number of evaluations used is recorded as
   `used_budget = len(optimizer.run_history)`.

3. **Early stopping inside LLaMEA**
   `create_llm_optimizer()` applies `llm_optimizer._early_stop_wrapper()` when
   the spec exposes `target_fitness` (here `0.0`), so a run terminates
   immediately once an exact decomposition (`err == 0`) is found.

4. **Validation**
   After each run, `main.py` optionally calls
   `benchmarks.matrix_multiplication.verify.validate_solution` on the best
   solution.
