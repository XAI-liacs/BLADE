# Fourier Uncertainty Inequality
* Goal
  * Minimise score $r_{max}^2/2\pi$.
    * where $r_{max}$ is the largest positive $x$ at which $P(x)/x^2$ changes sign.
* Function Class (AlphaEvolve notebook, section B.4):
  * $f(x) = P(x) \times e^{-\pi x^2}$
  * $P(x) = \sum_{k=0\ldots K}c[k] \times H_{4k}(x)$
    * Where $H_n$ is the (unscaled) physicists' Hermite polynomial.
    * The candidate returns the $K$ (= `n_terms`, default 3) coefficients $c[0..K-1]$ for $H_0, H_4, \ldots$
    * The evaluator appends $c[K] = -\sum_k c[k] H_{4k}(0) / H_{4K}(0)$ so that $P(0) = 0$; for $K = 3$ this is $c[3] = -(c_0 + 12 c_1 + 1680 c_2)/665280$.
    * Evenness holds by construction (degrees $0,4,8,\ldots$).
* Scoring:
  * If the leading coefficient of $P$ is negative, $P$ is replaced by $-P$.
  * $r_{max}$ is the largest positive sign change of $P(x)/x^2$; a candidate without one is invalid.
  * Scale invariant: any nonzero scaling of $c$ leaves the score unchanged.
* Best known: AlphaEvolve $C_4 \le 0.3521$; the stored seed scores $0.3520991$.
---
