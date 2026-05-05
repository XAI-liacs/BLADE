# Photonics Optimisation Problem
* Given a continuous design vector $x \in \mathbb{R}^{d}$ representing photonic structure parameters like:
    * geometry
    * material distribution
    * layer thicknesses,
    * $\ldots$\
    optimise the black-box physics simulator f(x) such that:

* The objective is optimised (maximised or minimised depending on task):
    $$O = \min_{x \in \mathbb{R}^d} f(x)$$
    $$O = \max_{x \in \mathbb{R}^d} f(x)$$
    * where f(x) may represent:
        * Transmission efficiency
        * Reflection suppression
        * Mode confinement
        * Bandgap width
        * Field enhancement
* Evaluation budget constraint:
    $$N_{\text{eval}} \le \text{budget}$$
    * The black-box function: $f(x)$ may only be called at most budget times.
    * Black-box setting:
	    * No gradient information available
	    * No analytic form accessible
	    * Potentially noisy evaluations
	    * Expensive simulation (e.g., FDTD / FEM)

* Decision Variables.
    * Continuous vector $x \in \mathbb{R}^d$, where $d$ is `dim`.

* Objective Type.
    * Continuous, simulation-based optimisation.

* Search Space.
    * High-dimensional, bounded continuous domain.

* Constraints.
	* Implicit physical feasibility (handled inside simulator)
	* Strict evaluation budget


* Implementation Interface:

    * The optimisation algorithm must follow:
        * `__init__(self, budget, dim, ...)`
            * `budget`: maximum number of allowed function evaluations
	        * `dim`: dimensionality of design vector
        * `__call__(self, func)`
            * `func(x)`: black-box objective function
	* Total calls to `func` must not exceed `budget`.

* Properties.
	* Continuous
	* Highly non-convex
	* Expensive evaluations
	* No derivative information
	* Strongly multimodal
	* Budget-constrained optimisation
---