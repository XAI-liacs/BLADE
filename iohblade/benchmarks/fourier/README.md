# Fourier Uncertainty Inequality
* Goal
  * Minimise score $r_{max}^2/2\pi$.
    * where $r_{max}$ is the largest positive root after which $P(x)$ stays non-negative.
* Function Class:
  * $f(x) = P(x) \times e^{-\pi x^2}$
  * $P(x) = \sum_{k =0\ldots K - 1}c[k] \times H_{4k}(x)$
    * Where $H_n$ is physicists' Hermite.
    * Evenness holds by construction (degrees $0,4,8,\ldots$).
* Constraints:
  * $P(0) < 0$.
  * Leading coefficient $c[K-1] > 0$
    * Scale-invarieant, any positive scaling leaves score unchanged.
  * Tail nonnegativity: $P(x) \ge 0 \forall x \ge r_{max}$
  * Optional numeric sanity: $P(x_{max}) \ge 0 \forall $ large $x_{max}$.
---
