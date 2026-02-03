# Number Theory
* Benchmark: Sums vs differences via the single-set formulation used in the paper.
    * We search for a finite set $U \doteq \mathbb Z^+ \cup \{0\}$ maximizing:
    $$c(U) = 1 + \frac{\log|U-U| - \log|U+U|}{\log(2\max(U)+1)}$$
    which lower-bounds the exponent (C_6) in $|A-B|\gtrsim |A+B|^{C_6}$.
    * Larger $c(U)$ implies a better lower bound on $C_6$. The evaluator computes $|U+U|$ and $|U-U|$ exactly via FFT convolution/correlation on the indicator of (U) over $[0,\max(U)]$.

`Notes`: the paper reports improved bounds using large (U) (sizes 2,003 and 54,265). Increase max_set_size if you want to attempt those scales.

---