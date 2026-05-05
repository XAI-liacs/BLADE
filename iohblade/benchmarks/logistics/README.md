# Travelling Salesman Problem
* Given a set of 2-D points $p \doteq (x, y) \in P$, generate a set of directed edges E such that:
* The total tour length is minimised:
    $$O = \min_{E} \sum_{(i,j)\in E} \|p_i - p_j\|_2$$

* Each point has exactly one incoming and one outgoing edge (no branching paths):
    $$\sum_{j \ne i} x_{ij} = 1 \quad \forall i$$
    $$\sum_{i \ne j} x_{ij} = 1 \quad \forall j$$
    * where:
        $$
            x_{ij} =
            \begin{cases}
            1 & \text{if edge } (i,j) \text{ is selected} \\
            0 & \text{otherwise}
            \end{cases}
        $$
    * The selected edges must form one connected tour visiting all points exactly once.

* Decision Variables.
    *  Binary edge selection variables x_{ij} \in \{0,1\}.

* Objective Type.
    * Combinatorial, permutation-based optimisation.

* Search Space Size.
    $(|P| - 1)! / 2$ possible tours (symmetric case).

* Properties.
    * Discrete
    * NP-hard
    * Highly multimodal
    * Symmetric under cyclic shifts and reversal of tour order.
---
## Capacitated Vehicle Routing Problem (CVRP)
* Given:
    * A depot $p_0$
    * A set of customer nodes $p_i \in P$, each with demand $d_i$.
    * A fleet of $K$ vehicles with identical capacity $Q$
* Generate a set of routes E such that:
    * The total travel distance is minimised:
        $$O = \min_{E} \sum_{(i,j)\in E} \|p_i - p_j\|_2$$
    * where:
        * Each customer is visited exactly once:
        $$\sum_{j \ne i} x_{ij} = 1 \quad \forall i \in P$$
        $$\sum_{i \ne j} x_{ij} = 1 \quad \forall i \in P$$
    * Vehicle capacity constraints are satisfied:
        $$\sum_{i \in R_k} d_i \le Q \quad \forall k = 1,\dots,K$$
        * where $R_k$ is the set of customers served by vehicle $k$.
* Each route is assumed to start and end at the depot, hence depot is not expected in route of any of the vehicle.:
    $$p_0 \notin k, \forall k \in K$$

* Decision Variables.
    * Binary edge variables x_{ij} \in \{0,1\}.

* Objective Type.
    * Combinatorial, multi-route permutation optimisation.

* Search Space Size.
    * Super-exponential in number of customers and vehicles.

* Properties.
    * Discrete
    * NP-hard
    * Constraint-heavy (capacity + routing)
    * Highly multimodal and symmetric within routes
---