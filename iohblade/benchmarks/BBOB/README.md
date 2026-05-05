# Black Box Optimization Benchmarking

* Contains a problem set, of 24 noiseless real-valued test functions supported on $n$ dimensional domain $\mathbb{D} \doteq [-5, 5]^n$.

## Problem Set
### Group 1: Separable Functions
* ***Sphere Function***:
  $$
    f_1(x) = \gamma(n) \times \sum_{i = 1}^nz_i^2 + f_{opt}
  $$
  * Transformations:
    $$z = x - x_{opt}$$

* ***Ellipsoidal Function***
  $$
    f_2(x) = \gamma(n) \times \sum_{i = 1}^n 10^{6\frac{i-1}{n-1}}z_i^2 +f_{opt}
  $$
  * Transformations:
    $$z = T_{osz}(x-x^{opt})$$

* ***Rastrigin Function***
  $$
    f_3(x) = \gamma(n)\times (10n - 10\sum_{i = 1}^n \cos(2\pi z_i) + ||z||^2) + f_{opt}
  $$
  * Transformations:
  $$ z = \Lambda^{10} T_{asy}^{0.2} (T_{osz} (x - x^{opt})) $$

* ***Büche–Rastrigin Function***
  $$
    f_4(x) = \gamma(n)\left(10n - 10\sum_{i=1}^n \cos(2\pi z_i) + ||z||^2 \right) + 100f_{pen}(x) + f_{opt}
  $$

    * Transformations:

        $$
            z = s_iT_{osz}(x_i - x_i^{opt})\quad i = 1,\ldots,n
        $$
        $$
            s_i =
                \begin{cases}
                10^{\frac{i - 1}{2(n - 1)} + 1} & \text{ if } z_i > 0 \text{ and i odd}\\
                10^{\frac{i - 1}{2(n-1)}} &\text{ otherwise}
                \end{cases}
        $$

* ***Linear Slope Function***
    $$
        f_5(x) = \gamma(n)\times\sum_{i=1}^n (5|s_i| - s_iz_i) + f_{opt}
    $$
    * Transformations:
    $$
        z_i = \begin{cases}
            x_i & \text{if } x_i^{opt}x_i \lt 25\\
            x_i^{opt} & \text{ otherwise}
        \end{cases}
        \quad\forall i = 1,\ldots,n
    $$
    $$
        s_i = sign(x_i^{opt}) 10^{\frac{i - 1}{n - 1}} \quad\forall i = 1,\ldots, n
    $$
    $$
        x^{opt} = z^{opt} = 5\times 1^+_-
    $$


### Group 2: Functions with Low or Moderate Conditioning

* ***Attractive Sector Function***
    $$
        f_6(x) = T_{osz} (\gamma(n) \times \sum_{i = 1}^n(s_iz_i)^2)^{0.9} +f_{opt}
    $$

    * Transformations:
        $$
            z = Q \Lambda^{10} R (x - x_{opt})\quad \text{with } R = P_{11}B_1P_{12}, Q =P_{21}B_2P_{22},\\
            s_i = \begin{cases}
                100 & \text{if } z_i \times x_i^{opt} > 0\\
                1 &\text{otherwise}
            \end{cases}
            \quad \forall i = 1, \ldots, n
        $$

* ***Step Ellipsoidal Function***
    $$
        f_7(x) = \gamma(n) \times 0.1 \max(\frac{|\hat z|}{10^4}, \sum_{i = 1}^n100^{\frac{i - 1}{n-1}}z_i^2) + f_{pen}(x) +f_{opt}
    $$
    * Transformations:
        $$
            \hat z= \Lambda^{10}R(x−x^{opt}) \quad \text{ with }  R= P_{11}B_1P_{12}\\
            \tilde z_i = \begin{cases}
                \lfloor 0.5 + \hat{z}_i \rfloor & \text{if } |\hat z| > 0.5\\
                \lfloor 0.5 + 10\hat z_i\rfloor/10 &\text{otherwise}
            \end{cases}
            \quad \forall i = 1, \ldots, n\\
            z = Q\tilde z \text{ with } Q = P_{21}B_2P_{22}
        $$
    
* ***Rosenbrock Function (original)***
    $$
        f_8(x) = \gamma (n) \times \sum_{i = 1}^n(100(z_i^2 - z_{i + 1})^2 + (z_i - 1)^2) +f_{opt}
    $$
    * Transformations:
        $$
            z = \max(1,\frac{\sqrt s}8)(x - x^{opt}) + 1, x^{opt} \in [-3,3]
        $$

* ***Rosenbrock Function (rotated)***
    $$
        f_9(x) = \gamma (n) \times \sum_{i = 1}^n(100(z_i^2 - z_{i + 1})^2 + (z_i - 1)^2) +f_{opt}
    $$
    * Transformations:
        $$
            z = \max(1,\frac{\sqrt s}8)R(x - x^{opt}) + 1,\\
            R = P_1BP_2,\\
             x^{opt} \in [-3,3]
        $$

### Group 3: Functions with High Conditioning and Unimodal
* ***Ellipsoidal Function***
    $$
        f_{10}(x) = \gamma(n) \times \sum_{i = 1} ^ n 10^{6\frac{i - 1}{n-1}}z_i^2 +f_{opt}
    $$
    * Transformations:
        $$
            z = T_{osz} (R(x− xopt)) \quad \text{ with } R= P_1BP_2
        $$

* ***Discus Function***
    $$
        f_{11} = \gamma (n) \times (10^6 \sum_{i = 1}^{\lceil n / 40 \rceil}z_i^2 + \sum_{i = \lceil n / 40 \rceil + 1}^n z_i^2) + f_{opt}
    $$
    * Transformations:
        $$
            z = T_{osz}(R(x - x^{opt})) \quad \text{with } R = P_1BP_2
        $$

* ***Bent Cigar Function***
    $$
        f_{12} = \gamma (n) \times (\sum_{i = 1}^{\lceil n / 40 \rceil}z_i^2 + 10^6\sum_{i = \lceil n / 40 \rceil + 1}^n z_i^2) + f_{opt}
    $$
    * Transformations:
        $$
            z = RT_{asy}^{0.5}(R(x - x^{opt})) \quad \text{with } R = P_1BP_2
        $$
* ***Sharp Ridge Function***
    $$
        f_{13} = \gamma (n) \times (\sum_{i = 1}^{\lceil n / 40 \rceil}z_i^2 + 100 \sqrt{\sum_{i = \lceil n / 40 \rceil + 1}^n z_i^2)} + f_{opt}
    $$
    * Transformations:
        $$
            z = Q\Lambda^{10}R(x− x^{opt}) \quad \text{ with } R= P_{11}B_1P_{12}, Q= P_{21}B_2P_{22}
        $$
* ***Different Powers Function***
    $$
        f_{14}(x) = \gamma(x) \times \sum_{i = 1}^n |z_i|^{(2 + \frac{4(i - 1)}{n-1})} +f_{opt}
    $$
    * Transformations:
        $$
            z = R(x− x^{opt}) \quad \text{ with } R= P_1BP_2
        $$

### Group 4: Multi-modal Functions with Adequate Global Structure
* ***Rastringin Function***
    $$
        f_{15}(x) = \gamma(n) \times (10n-10\sum_{i = 1}^n\cos(2\pi z_i) + ||z||^2) + f_{opt}
    $$

    * Transformations:
        $$
            z = R\Lambda^{10}QT^{0.2}_{asy}(T_{osz}(R(x-x^{opt})))\\
            R= P_{11}B_1P_{12}\\
            Q= P_{21}B_2P_{22}
        $$
* ***Weierstrass Function***
    $$
        f_{16}(x) = 10(\frac1n \sum_{i = 1}^n\sum_{k = 0}^{11}\frac1{2^k}\cos(2\pi 3^k(z_i + \frac12)) - f_0)^3 + \frac{10}nf_{pen}(x) +f_{opt}
    $$
    * Transformations:
        $$
            z = R\Lambda^{\frac1{100}}QT_{osz}(R(x - x^{opt})),\\
                R = P_{11}B_1P_{12}, Q = P_{21}B_2P_{22},\\
                f_0 = \sum_{k = 0}^{11}\frac1{2^k}\cos(\pi3^k)
        $$
* ***Schaffers F7 Function***
    $$
        f_{17}(x) = (\frac1{n-1}\sum_{i = 1}^{n - 1}(\sqrt s_i + \sqrt s_i \sin^2(50(s_i)^{\frac15})))^2 + 10f_{pen}(x) + f_{opt}
    $$
    * Transformations:
        $$
            z = \Lambda^{10}QT^{0.5}_{asy} (R(x− x^{opt})) \\
            \quad Q= P_{21}B_2P_{22}\\
            \quad R= P_{11}B_1P_{12}\\
            s_i = \sqrt{z_i^2 + z_{i + 1}^2}, \quad \forall i = 1, \ldots, n - 1
        $$
* ***Schaffers F7 Function, Moderately Ill-Conditioned***
    $$
        f_{18}(x) = (\frac1{n-1}\sum_{i = 1}^{n - 1}(\sqrt s_i + \sqrt s_i \sin^2(50(s_i)^{\frac15})))^2 + 10f_{pen}(x) + f_{opt}
    $$
    * Transformations:
        $$
            z = \Lambda^{1000}QT^{0.5}_{asy} (R(x− x^{opt})) \\
            \quad Q= P_{21}B_2P_{22}\\
            \quad R= P_{11}B_1P_{12}\\
            s_i = \sqrt{z_i^2 + z_{i + 1}^2}, \quad \forall i = 1, \ldots, n - 1
        $$
* ***Composite Griewank-Rosenbrock Function F8F2***
    $$
        f_{19}(x) = \frac{10}{n - 1}\sum_{i = 1}^{n - 1}(\frac {s_i}{4000} - \cos(s_i) + 10 + f_{opt})
    $$
    * Transformations:
        $$
            z = \max(1, \frac{\sqrt s}8)Rx + \frac12 \quad \text{ with } R = P_1BP_2\\
            s_i = 100(z_i^2 - z_{i + 1})^2 + (z_i - 1)^2, \quad \forall i = 1, \ldots, n - 1\\
            z^{opt} = 1
        $$

### Group 5: Multi-modal Functions with Weak Global Structure
* ***Schwefel Function***
    $$
    f_{20}(x) = - \frac1{100n}\sum_{i = 1}^n z_i\sin(\sqrt{|z_i|}) + 4.189828872724339 + 100f_{pen}(x) + f_{opt}
    $$
    * Transformations:
        $$
            \hat x = 2 \times 1^+_- \otimes x\\
            \hat z_{i + 1} = \hat x_{i + 1} + \frac14(\hat x_i - 2 |x_i^{opt}|), \quad \forall i = 1, \ldots, n - 1\\
            z = 100(\Lambda^{10} (\hat z - 2 |x^{opt}|) + 2|x^{opt}|),\\
            x^{opt} = 4.2096874633/21^+_-
        $$
* ***Gallagher’s Gaussian 101-me Peaks Function***
    $$
        f_{21}(x) = T_{osz}(10 - \max_{i = 1}^{101} w_i e^{-\frac1{2n}(z-y_i)^TB^TC_iB(z-y_i)})^2 + f_{pen}(x) +f_{opt}
    $$
    * Transformations:
        $$
            w_i = \begin{cases}
            1.1 + \frac{8(i - 2)}{99} & 2\le i \le 21\\
            10 &i = 1
            \end{cases}\\
        $$
        * $B$ is a block-diagonal matrix without permutations of the variables.
        $$
            C_i = \frac{\Lambda^{\alpha_i}}{\alpha_i^{0.25}}
        $$
        * where $\Lambda^{\alpha_i}$ is defined as usual, but with randomly permuted diagonal elements. 
        * For $i = 2, \ldots, 21$ $\alpha_i$ is drawn uniformly from the set $\{10^{6j/19}, j = 0, \ldots, 19\}$ without replacement.
        * $\alpha_i = 10^3$ for $i = 1$
        *  The local optima $y_i$ are uniformly drawn from the domain $[−4.9, 4.9]^n$ for $i = 2,\ldots, 21$ and $y_1 \in [−3.92, 3.92]^n$.
        * The global optimum is at x^{opt} = y_1.
* ***Katsura Function***
    $$
        f_{23}(x)=(\frac{10}{n^2}\prod_{i= 1}^n(1 + i\sum_{j = 1}^{32}\frac{|2^jz_i - [2^jz_i]|}{2^j}) - \frac{10}{n^2})
    $$
    * Transformations:
        $$
            z = Q\Lambda1^{100}R(x− x^{opt}),\\R= P_{11}B_1P_{12},\\ Q= P_{21}B_2P_{22}
        $$
* ***Lunacek bi-Rastrigin Function***
    $$
        f_{24}(x) = \gamma(n) \times (\min (\sum_{i = 1}^n(\hat x_i - \mu_0)^2, n + s\sum_{i = 1}^n(\hat x_i - \mu_1)^2) + 10(n - \sum_{i = 1}^n\cos(2\pi z_i))) + 10^4 f_{pen}(x) + f_{opt}
    $$
    * Transformations:
    $$
        \hat x = 2sign(xopt) \otimes x,\\
        x^{opt} = 0.5\mu_01^+_-
    $$
    $$
        z = QΛ^{100}R(\hat x− µ_01),\\
        R = P_{11}B_1P_{12},\\
        Q = P_{21}B_2P_{22}
    $$
    $$
        \mu_0 = 2.5,\\
        \mu_1 = -\sqrt{\frac{\mu_0^2 - 1}{s}},\\
        s= 1 - \frac1{2\sqrt{n + 20} - 8.2}
    $$
----
# SBOX-Const
* SBOX‑COST is a strict box‑constrained variant of the aforementioned BBOB benchmark suite where decision variables outside the domain $[-5,5]^n$ are treated as infeasible and return an invalid evaluation instead of a finite value.  ￼
* It uses the same underlying 24 continuous functions as BBOB but enforces hard boundary constraints and adjusts the distribution of optima to better cover the full domain.
* [Resource](https://inria.hal.science/hal-04403658/file/sboxcost-cmacomparison-authorversion.pdf)

# Many-Affine BBOB
* Many-Affine BBOB (MA-BBOB) generates new continuous optimization problems by taking affine  combinations of the aformentioned 24 BBOB functions, allowing arbitrary weights, shifts, and rotations.
* This method fills gaps between canonical BBOB landscapes in feature space, producing a richer and more diverse set of benchmark instances.
* MA-BBOB preserves meaningful performance patterns while enabling large sets of distinct instances for algorithm evaluation and landscape analysis.
* [Resource](https://dl.acm.org/doi/10.1145/3673908).
---