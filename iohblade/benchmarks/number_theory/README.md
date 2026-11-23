# Number Theory
* Benchmark: Sums vs differences via the single-set formulation used in the paper.
    * We search for a finite set $U \doteq \mathbb Z^+ \cup \{0\}$ maximizing:
    $$c(U) = 1 + \frac{\log|U-U| - \log|U+U|}{\log(2\max(U)+1)}$$
    which lower-bounds the exponent (C_6) in $|A-B|\gtrsim |A+B|^{C_6}$.
    * Larger $c(U)$ implies a better lower bound on $C_6$. The evaluator computes $|U+U|$ and $|U-U|$ exactly via FFT convolution/correlation on the indicator of (U) over $[0,\max(U)]$.

* Limits: $|U| \le$ `max_set_size` (default 2048) and $\max(U) \le 2^{21} = 2{,}097{,}152$.

`Notes`: the paper reports improved bounds using large (U) (sizes 2,003 and 54,265). The seed is the 2,003-element set (notebook B.6 `solution_1`, $\max(U) = 1{,}040{,}200$), which scores $c(U) = 1.147989$ and fits the default limits; the 54,265-element set (bound 1.1584) does not. Increase `max_set_size` (and `max_value`) if you want to attempt that scale.

---