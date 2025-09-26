# Autocorrelation Inequality Optimisation (step function minimisation)

* Tighten Bounds on $C_1$, $C_2$ and $C_3$ using step function on $[-\frac14, \frac14]$.
* Each candidate returns a vector `f`, of length `N`.
* No $L_2$ normalisation is applied, all objectives are scale invariant.

## Discretisation
* Bins `n_bins`: number of bins over $[-\frac14, \frac14]$.
* Step Size: `dx = 0.5/N`.
* Autoconvolution: `g = dx * np.conv(f,f,mode='full')`.

## Riemann-Sum Analogues
*   $$
        I = \Delta x * \sum_i f[i]
    $$
    * `I = dx * np.sum(f)`
*   $$
        L_1 = \Delta x * \sum_j |g[i]|
    $$
    * `L1 = dx * np.sum(abs(g))`
*   $$
        L_2^2 = \Delta x * \sum_j(g[j]^2)
    $$
    * `L2sq = dx * np.sum(g[j] ** 2)`
*   $$
        L_\infty = \max_j |g[j]|
    $$
    * `Linf = np.max(abs(g))`
*   $$
        g_{max} = \max_jg[j]
    $$
    * `max_g  = max_j g[j]`
*   $$
        |g|_{max} = max_j |g[j]|
    $$
    * `max_abs_g = max_j abs(g[j])`

## Fitness
> `Note`:  we minimize the negative of autocorr_ineq_2 to be aligned with the rest of the problems.

* $C_1$ (spec: `autocorr_ineq_1`)
    * Score:
    $$
        AC_1 = \frac{g_{max}}{I^2}
    $$
    * Constraints: `f >= 0`, `I > 0`
* $C_2$ (spec: `autocorr_ineq_2`)
    * Score:
    $$
        AC_2 = -\frac{L_2^2}{L_1 \times L_\infty}
    $$
    * Constraints: `f >= 0`
* $C_3$ (spec: `autocorr_ineq_3`):
    * Score:
    $$
        AC_3 = \frac{|g|_{max}}{I^2}
    $$
    * Constraints: $f\in \mathbb{R}; I \ne 0$.
## Defaults
* $C_1$: `N = 600`
* $C_2$: `N = 50`
* $C_3$: `N = 400`

## Best Known Results.
### Analysis


| Name| Description | Best Known |Alpha Evolve
| :--- | :--- | :--- | :---
| $1^{st}$ Autocorrelation Inequality | Tighten upper bound $C_1$ in \|\|f*f\|\|. Improving this constant sharpens energy bounds in additive combinatorics.| $C_1 \le 1.5098$ | $C_1 \le 1.5053$|
| $2^{nd}$ Autocorrelation Inequality | Improve lower bound $C_2$ in $\|\|f * f\|\|_2^2 \le C_2$. Better $C_2$ narrows the gap in convolution inequalities.| $C_2 \ge 0.8892$ | $C_2 \ge  0.8962$|
| $3^{nd}$ Autocorrelation Inequality | Tighten upper bound $C_3$ for $\max abs(\|\|f*f\|\|)$ . Refines extrenal behaviour of auto‑convolutions.| $C_3 \le 1.4581$ | $C_2 \le 1.4557$|
---
