Benchmarks
==========

BLADE includes a collection of benchmarks ranging from BBOB to
Google DeepMind–inspired tasks derived from the AlphaEvolve paper.
These instances are available in two complementary forms:

- ``run_benchmarks/`` provides standalone reference scripts for running
  each task directly.
- ``iohblade/benchmarks`` packages the same tasks for programmatic use
  in experiments and pipelines.

The packaged benchmarks are grouped by domain.

------------------------------------------------------------------------------------

.. _list_of_benchmarks:
List of Benchmarks
------------------

Analysis
^^^^^^^^

Perform auto-correlation on a time series using the following configuration:

- The domain size ``N`` defines the discretisation of
  :math:`[-\tfrac{1}{4}, \tfrac{1}{4}]`.
- Step size:
  :math:`dx = \frac{0.5}{N}`
- Auto-convolution:
  :math:`g = dx \cdot \mathrm{conv}(f, f, \text{mode='full'})`
- Riemann sum analogues:

  - :math:`I = dx \sum_i f[i]`
  - :math:`L_1 = dx \sum_j |g[j]|`
  - :math:`L_{2}^{2} = dx \sum_j g[j]^2`
  - :math:`L_{\infty} = \max_j |g[j]|`
  - :math:`\max_g = \max_j g[j]`
  - :math:`\max_{\text{abs}_g} = \max_j |g[j]|`

Auto-Correlation 1
~~~~~~~~~~~~~~~~~~

- Score:
  :math:`\frac{\max_g}{I^2}`
- Constraints:
  :math:`f \ge 0`, :math:`I > 0`
- Optimisation direction: minimisation
- Default:
  :math:`N = 600`

Auto-Correlation 2
~~~~~~~~~~~~~~~~~~

- Score:
  :math:`\frac{L_{2}^{2}}{L_1 \cdot L_{\infty}}`
- Constraints:
  :math:`f \ge 0`
- Optimisation direction: maximisation
- Default:
  :math:`N = 50`

Auto-Correlation 3
~~~~~~~~~~~~~~~~~~

- Score:
  :math:`\frac{\max_{\text{abs}_g}}{I^2}`
- Constraints:
  :math:`f` real-valued, :math:`I \ne 0`
- Optimisation direction: minimisation
- Default:
  :math:`N = 400`

-----------

AutoML
^^^^^^

- This benchmark evaluates **LLM-generated scikit-learn pipelines** on **OpenML tasks**.
- Given an OpenML task ID, BLADE downloads the dataset/splits from OpenML and asks the LLM
to produce a single Python class implementing a pipeline to optimise said task.

-----------

Black Box and Photonics Optimisation (BBO & Photonics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Contains a set of Black Box Optimisation problems from the `ioh benchmarks`_ library.

.. _ioh benchmarks: https://iohprofiler.github.io/IOHexp/

- Also contains a set of **photonics** problems, supported by black bock optimisation.
-----------

Combinatorics
^^^^^^^^^^^^^

Erdős Minimum-Overlap Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The continuous Erdős minimum-overlap problem seeks to find measurable
  functions :math:`f, g: [-1,1] \rightarrow [0,1]` that satisfy:

  1. **Complementarity**: :math:`f(x) + g(x) = 1` for all :math:`x \in [-1,1]`
  2. **Unit mass**: :math:`\int_{-1}^{1} f(x) \, dx = \int_{-1}^{1} g(x) \, dx = 1`
  3. **Bounds**: :math:`f(x), g(x) \in [0,1]` for all :math:`x \in [-1,1]`

- And minimize the **maximum overlap integral**:

  .. math::

     C := \sup_{x \in [-2,2]} \int_{-1}^{1} f(t) g(x+t) \, dt

- where g is extended by zero outside [-1,1].

Euclidean Steiner Tree Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Given an undirected graph, the Euclidean Steiner Tree algorithm optimises the minimum spanning tree by adding extra nodes (Steiner points) to the graph.
- Adding these nodes allows for a shorter MST.
- This benchmark takes a set of points, runs MST on the original points and on the points combined with Steiner points, and returns their ratio.
- Optimisation goal:

  :math:`\min \frac{\text{mst(points + steiner_points)}}{\text{mst(points)}}`

Graph Colouring Problem
~~~~~~~~~~~~~~~~~~~~~~~
- The Graph Colouring benchmark focuses on assigning colours to the vertices of a graph such that no two adjacent vertices share the same colour.
- Input: a graph :math:`G = (V, E)` with vertices :math:`V` and edges :math:`E`.
- Objective: minimise the number of colours used while ensuring a **valid colouring**.
- Constraints:
    1. Each vertex receives exactly one colour.
    2. Adjacent vertices must not share the same colour.

- Fitness:
    .. math::
        \text{fitness} = |C| \quad \text{where } C \text{ is the set of colours used}
-----------

Fourier
^^^^^^^

Fourier Uncertainty Inequality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- This benchmark minimises the Fourier Uncertainty Inequality given by :math:`f(x) = P(x) e^{-\pi x^2}` where :math:`r_{\max}` is the largest positive root after which :math:`P(x)` remains non-negative.

- Function Class:

  .. math::

     f(x) = P(x) \, e^{-\pi x^2}, \quad
     P(x) = \sum_{k=0}^{K-1} c[k] \, H_{4k}(x)

  - :math:`H_n` are physicists' Hermite polynomials.
  - Evenness holds by construction (degrees 0,4,8,…).

- Constraints:

  1. :math:`P(0) < 0`
  2. Leading coefficient :math:`c[K-1] > 0` (scale-invariant: any positive scaling leaves score unchanged)
  3. Tail nonnegativity: :math:`P(x) \ge 0 \quad \forall x \ge r_{\max}`
  4. Optional numeric sanity check: :math:`P(x_{\max}) \ge 0` for large :math:`x_{\max}`
-----------

Geometry
^^^^^^^^

Heilbronn Problem on a Unit-Area Triangle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- This benchmark finds :math:`n = 11` points inside a triangle of area 1 that **maximize the minimum area** of any triangle formed by these points.
- Score:
    .. math::

        \text{Score} = \min_{a,b,c} \Big( \max \sqrt{s (s-a)(s-b)(s-c)} \Big)

where:
    - :math:`a,b,c \in \text{points}`
    - :math:`s = \frac{a+b+c}{2}`
    - :math:`a \ne b \ne c`

Heilbronn Problem on a Unit-Area Convex Region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- This benchmark finds :math:`n \in \{13, 14\}` points inside a convex region of area 1 that **maximize the minimum area** of any triangle formed by these points.
- The convex hull of the points defines the region, which is then rescaled to have area 1.
- Score:
    .. math::

        \text{Score} = \min_{a,b,c} \Big( \max \sqrt{s (s-a)(s-b)(s-c)} \Big)

    where:
        - :math:`a,b,c \in \text{points}`
        - :math:`s = \frac{a + b + c}2`
        - :math:`a \ne b \ne c`

Kissing Number in 11 Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- This benchmark maximises the count of integer vectors
    :math:`C \subset \mathbb{Z}^{11} \setminus \{0\}`

    that satisfies the **kissing configuration constraint**:

    .. math::

            \min_{x \ne y} \|x - y\| \ge \max_{x \in C} \|x\|

- Score:
    .. math::
            \text{Score} = |C|

Min/Max Distance Ratio
~~~~~~~~~~~~~~~~~~~~~~

- This benchmark finds a configuration of points that **minimizes the squared ratio of the maximum to minimum pairwise distances**.

- Fitness function:
    .. math::

        \text{Fitness} = \left( \frac{\max_{i<j} d(i,j)}{\min_{i<j} d(i,j)} \right)^2

- Variants:
    - **2D space** with :math:`n = 16` points
    - **3D space** with :math:`n = 14` points

Spherical Code
~~~~~~~~~~~~~~

- This benchmark is a **maximisation problem** where :math:`n=30` distinct points are arranged on the surface of a **unit sphere**.
- Objective: **maximize the minimum pairwise angle** between any two points.
- Score:

  .. math::

     \text{f} = \min_{i \ne j} \theta(i,j)

  where :math:`\theta(i,j)` is the angle between points :math:`i` and :math:`j`.
-----------

Kernel Tuner
^^^^^^^^^^^^

- This benchmark evaluates **metaheuristic algorithms for hardware kernel tuning** across integer and variable-dimensional search spaces with constraints.
- The algorithm is scored based on performance of the kernel using metrices like:
    - runtime
    - throughput
    - custom matrices.
-----------

Logistics
^^^^^^^^^

Travelling Salesman Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Given a set of 2-D points :math:`(x, y)`, find the sortest path connecting those points.
- Optimisation Direction: Minimisation.
- Fitness:
    .. math::
        \text{Minimise} \quad
        \sum_{i \in V} \sum_{j \in V, j \ne i} d(i,j) \, x_{ij}
- Constraints:
    - Each customer is served exactly once.

Vehicle Routing Problem
~~~~~~~~~~~~~~~~~~~~~~~

- Given a set of 2-D points with their weights as, :math:`(x, y, w)` and a count :math:`n` of similar vehicles, with capacity :math:`c`, and a depot point :math:`(x,y, 0)` find the minimum travel distance.
- Optimisation Direction: Minimisation
- Fitness:
    .. math::

        \text{Minimise} \quad
        \sum_{i \in V} \sum_{j \in V, j \ne i} d(i,j) \, x_{ij}

- Constraints:
    - Each customer is only serviced once.
    - For each vehicle: :math:`sum_{i \in v_j} w_i \le c` where :math:`v_j` is a vehicle, and :math:`j \in \{0,\ldots,n\}`.

-----------

Matrix Multiplication via Tensor Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Given matrix dimensions:
    .. math::

        n \times m \ \cdot \ m \times p \ \longrightarrow \ n \times p

    define a tensor

    .. math::

        T \in \mathbb{R}^{(n \cdot m) \times (m \cdot p) \times (p \cdot n)}

    that encodes ordinary matrix multiplication.

- Factorisation:
    - A rank-:math:`r` CP decomposition of :math:`T`:
        .. math::

            T[i,j,k] = \sum_{\ell=1}^{r} F_1[i,\ell] \cdot F_2[j,\ell] \cdot F_3[k,\ell]

        yields a matrix multiplication algorithm requiring only :math:`r` scalar multiplications.

- Objective:
    - Find the **smallest rank :math:`r`** that allows exact reconstruction (zero error).

- Constraints:
    - All entries of the factor matrices :math:`F_1, F_2, F_3` must lie on a **quantisation grid**:
        .. code-block::

            grid = 0.5, 1.0, 1.5, ...

-----------

Number Theory
^^^^^^^^^^^^^
Sums vs Differences (Single-Set Formulation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- This benchmark searches for a finite set

  :math:`U \subset \mathbb{Z}^+ \cup \{0\}`

  that **maximizes** the following quantity:

  .. math::

     c(U) = 1 + \frac{\log|U-U| - \log|U+U|}{\log(2 \max(U) + 1)}

- Purpose:

  - :math:`c(U)` **lower-bounds the exponent** :math:`C_6` in the inequality

    .. math::

       |A-B| \gtrsim |A+B|^{C_6}

  - A larger :math:`c(U)` corresponds to a **better lower bound** on :math:`C_6`.

- Evaluation:

  - The evaluator computes :math:`|U+U|` and :math:`|U-U|` exactly.
  - Implementation uses **FFT convolution/correlation** on the indicator function of :math:`U` over the domain :math:`[0, \max(U)]`.

-----------

Packing
^^^^^^^

- Contains a set of packing problems, ranging from Hexagonal Packing to Circle Packing.

Circle Packing
~~~~~~~~~~~~~~

- This benchmark selects and places candidate circles inside a circular container
  to **maximize total packed area** under non-overlap constraints.

- Input parameters:

  - A circular container :math:`c = (x, y, r) \in \mathbb{R}^3`
  - Candidate radii :math:`r_i \; \forall i \in \{1, \ldots, n\}`
  - Tolerance :math:`\epsilon > 0` (default :math:`10^{-12}`)

- Optimisation objective:

  .. math::

     O = \max \sum_{i=1}^{n} \alpha_i \, \pi r_i^2

  where:

  - :math:`\alpha_i \in \{0,1\}` indicates whether circle :math:`i` is included.
  - Non-overlap constraint:

    .. math::

       \| (x_i, y_i) - (x_j, y_j) \|
       \ge r_i + r_j + \epsilon
       \quad \forall i \ne j \text{ with } \alpha_i = \alpha_j = 1

  - Containment constraint:

    .. math::

       \| (x_i, y_i) - (x, y) \| + r_i
       \le r + \epsilon
       \quad \forall i \text{ with } \alpha_i = 1

  Only circles satisfying both constraints contribute to the objective.

Hexagonal Packing
~~~~~~~~~~~~~~~~~

- This benchmark generates :math:`n` disjoint regular hexagons parameterised by
    .. math::

        (x_i, y_i, \theta_i)
        \quad \forall i \in \{1, \ldots, n\}

    - where :math:`(x_i, y_i)` is the center and :math:`\theta_i` the orientation.

- Disjointness condition:

- Any two hexagons must be separated by at least tolerance :math:`\epsilon`.

- Optimisation objective:
    - The evaluator computes the smallest enclosing regular hexagon
        :math:`h` that contains all generated hexagons.

    - Goal:
        .. math::

            \text{minimise } \text{Area}(h)


Rectangle Packing
~~~~~~~~~~~~~~~~~

- This benchmark requires the algorithm to generate :math:`n` disjoint circles inside a rectangular container of perimeter :math:`4`, respecting a specified tolerance :math:`\epsilon`.

- Generated circles:
    .. math::

        (x_i, y_i, r_i) \quad \forall i \in \{1, \ldots, n\}

- Constraints:
    - **Non-overlap**:
        .. math::

            \| (x_i, y_i) - (x_j, y_j) \|
            \ge r_i + r_j + \epsilon
            \quad \forall i \ne j

    - **Containment**:
        .. math::

            (x - x_i)^2 + (y - y_i) = r_i^2 \text{ must lie entirely within the rectangle}
            \quad \forall i


Unit Square Packing
~~~~~~~~~~~~~~~~~~~

- This benchmark requires the algorithm to generate :math:`n` disjoint circles inside a unit-square, respecting a specified tolerance :math:`\epsilon`.

- Generated circles:
    .. math::

        (x_i, y_i, r_i) \quad \forall i \in \{1, \ldots, n\}

- Constraints:
    - **Non-overlap**:
        .. math::

            \| (x_i, y_i) - (x_j, y_j) \|
            \ge r_i + r_j + \epsilon
            \quad \forall i \ne j

    - **Containment**:
        .. math::
            (x - x_i)^2 + (y - y_i) = r_i^2 \text{ must lie entirely within the square}
            \quad \forall i

-----------
