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

Methodology Taxonomy
--------------------

The benchmark families in ``iohblade/benchmarks`` can also be grouped by
optimization structure.

.. _taxonomy_black_box_continuous:

Black-box Continuous
^^^^^^^^^^^^^^^^^^^^

- ``BBOB``
- ``mabbob``
- ``photonics``

.. _taxonomy_black_box_discrete_mixed_integer:

Black-box Discrete/Mixed-Integer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``combinatorics``
- ``logistics``
- ``packing``
- ``kerneltuner``

.. _taxonomy_structured_scientific_program_search:

Structured Scientific Program/Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``analysis``
- ``fourier``
- ``number_theory``
- ``matrix_multiplication``
- ``geometry``

.. _taxonomy_ml_pipeline_optimization:

ML Pipeline Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

- ``automl``

------------------------------------------------------------------------------------

.. _default_suites:

Default Suites
--------------

Default suites are named sets of problems with fixed settings, defined in
:mod:`iohblade.benchmarks.suites`. Use them in a YAML experiment with
``suite:``; their ``params`` apply to every problem in the suite.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Suite
     - Problems
     - Settings (defaults)
   * - ``bbob_groups``
     - 5: one per BBOB function group (``BBOB-g<group>-<dim>d``)
     - ``dim=10``, ``budget_factor=2000``; instances 1-5 of each function
       for training, 6-15 for testing (``n_train``, ``n_test``);
       ``groups`` selects groups.
   * - ``bbob``
     - 24: one per BBOB function f1-f24 (``BBOB-f<fid>-<dim>d``)
     - ``dim=10``, ``budget_factor=2000``; instances 1-10 for training,
       11-30 for testing (``n_train``, ``n_test``); ``functions`` selects
       functions.
   * - ``ma_bbob``
     - 1 (``MA_BBOB-<dim>d``)
     - ``dim=10``, ``budget_factor=2000``; instances 0-24 for training,
       25-74 for testing.
   * - ``continuous_bbo``
     - 30: ``bbob_groups``, ``bbob`` and ``ma_bbob``
     - ``dim``, ``budget_factor`` for all 30 problems.
   * - ``alphaevolve_math``
     - 17: auto-correlation inequalities, Erdős minimum overlap, the
       uncertainty inequality, hexagon/rectangle/unit-square packing,
       Heilbronn problems, kissing number 11-D, min/max distance ratios,
       sums versus differences
     - ``use_best=False`` (show the best known solution as the example),
       ``eval_timeout=600``.
   * - ``combinatorial``
     - 6: Euclidean Steiner trees (50, 100, 250 points), graph colouring,
       TSP and CVRP (``A-n53-k7``)
     - ``eval_timeout=300``.
   * - ``default``
     - 28: ``bbob_groups``, ``alphaevolve_math`` and ``combinatorial``
     - ``dim``, ``budget_factor`` (BBOB groups), ``use_best`` (maths); a
       cross-domain collection.
   * - ``quick``
     - 5: BBOB groups 1 and 4, circle packing, Heilbronn triangle, TSP
     - ``dim=5``, ``budget_factor=1000``; for trying out a setup.
   * - ``full``
     - 55: ``continuous_bbo``, ``alphaevolve_math``, ``combinatorial``,
       circle packing and spherical codes
     - ``dim``, ``budget_factor``, ``use_best``;
       ``include_matrix_multiplication`` adds the 54 matrix multiplication
       problems.

**Evaluation timeouts.** Every problem takes ``eval_timeout`` (seconds per
evaluation, at least 120). The defaults are 600 s for the mathematical
problems, 300 s for the combinatorial ones, and for ``bbob_groups``,
``bbob`` and ``ma_bbob`` 1 ms per function evaluation of the training runs
(500 s for a 5-function group or MA-BBOB, 200 s for a single function, in
10-D with budget factor 2000), at least 120 s. Set ``eval_timeout`` in
``params`` to change it.

Suite keys use underscores; ``bbob-groups`` with a hyphen works as well.

The AlphaEvolve suites (``auto_correlation``, ``hexagon_packing``, ...)
all take ``use_best`` (default ``false``) and ``eval_timeout``; the packing
suites still accept the old name ``get_best``.

.. code-block:: yaml

   problems:
     - suite: continuous_bbo
       params: {dim: 10, budget_factor: 2000}
     - suite: alphaevolve_math
       params: {use_best: false}

.. _task_leakage:

What the LLM is told
^^^^^^^^^^^^^^^^^^^^

The prompts describe each task without naming the benchmark suite or the
instance (no "BBOB", function ids, instance names or literature constants),
so that a method cannot rely on what the LLM remembers about the benchmark.

- **Continuous problems** (BBOB, MA-BBOB, photonics) pass the generated
  algorithm a ``BlackBoxFunction`` (``iohblade.benchmarks.blackbox``) instead
  of the IOH problem itself. It offers ``func(x)``, ``func.bounds.lb`` /
  ``func.bounds.ub``, ``func.meta_data.n_variables`` and ``func.state``
  (``evaluations``, ``current_best``), but not ``func.optimum``, the
  function's name and id, or ``reset()``.
- **Best known scores** are not mentioned in the feedback of an evaluation
  (which the LLM sees in the next prompt). To include them, set
  ``problem.show_best_known = True`` on a problem in Python.
- BBOB with ``specific_fid`` still tells the LLM the function's name, on
  purpose; ``use_best=True`` gives the generated code the best known
  solution, also on purpose.

.. _suite_running_time:

Running time
^^^^^^^^^^^^

Approximate wall-clock time of **one run of one method** on each default
suite with ``budget: 100``. With ``budget: 100``, a method such as LLaMEA or
random search makes 100 LLM calls and 100 evaluations per problem, and the
problems of a suite run one after the other. Multiply by the number of
methods and runs (seeds), and divide by ``n_jobs``.

.. list-table::
   :header-rows: 1
   :widths: 18 9 14 13 18 16

   * - Suite
     - Problems
     - LLM calls and evaluations
     - LLM time only
     - Typical total
     - Upper bound (every evaluation times out)
   * - ``bbob_groups`` (10-D)
     - 5
     - 500
     - 3 h
     - 6-10 h
     - 70 h
   * - ``bbob`` (10-D)
     - 24
     - 2,400
     - 14 h
     - 20-28 h
     - 135 h
   * - ``ma_bbob`` (10-D)
     - 1
     - 100
     - 0.6 h
     - 2-4 h
     - 14 h
   * - ``continuous_bbo`` (10-D)
     - 30
     - 3,000
     - 17 h
     - 28-42 h
     - 220 h
   * - ``alphaevolve_math``
     - 17
     - 1,700
     - 9 h
     - 13-39 h
     - 290 h
   * - ``combinatorial``
     - 6
     - 600
     - 3 h
     - 4-9 h
     - 54 h
   * - ``default``
     - 28
     - 2,800
     - 15 h
     - 23-58 h
     - 415 h
   * - ``quick``
     - 5
     - 500
     - 3 h
     - 4-9 h
     - 35 h
   * - ``full``
     - 55
     - 5,500
     - 31 h
     - 45-95 h
     - 605 h

Per problem, one run takes about 1-2 h for a BBOB group in 10-D, about
1 h for a single BBOB function, 2-4 h for MA-BBOB in 10-D, 0.7-2.3 h for a mathematical problem and 0.6-1.4 h
for a combinatorial problem.

How these numbers are estimated:

* **LLM:** 20 s per call, typical for a hosted model writing a full
  algorithm. Local models (for example a 14B model with Ollama) often take
  30-90 s, which roughly doubles the totals.
* **Harness:** about 1.5 s per evaluation for the isolated evaluation
  subprocess (measured), once the evaluation environment is cached. The
  first run builds the environment once, which takes a few minutes.
* **Evaluation of the generated code**, measured with the example solutions
  and scaled for typical LLM-written algorithms:

  * **BBOB group, 10-D, budget factor 2000:** 25 runs of 20,000 function
    evaluations. Random search takes 8-11 s; LLM-written optimisers (for
    example CMA-ES or DE variants in NumPy) typically take 2-5 times longer,
    so 20-50 s.
  * **MA-BBOB, 10-D:** random search takes 26 s, so 50-130 s.
  * **Mathematical and combinatorial problems:** the example constructions
    take milliseconds, but generated code usually runs its own search until
    it is satisfied, so 5-60 s and 1-30 s.

* **Upper bound:** every evaluation runs until its ``eval_timeout``. This
  is useful for planning the worst case on a cluster, not as an
  expectation.

Ways to shorten a run:

* Use ``quick`` (5-D BBOB, budget factor 1000) or a lower ``dim`` and
  ``budget_factor`` for the continuous suites. Evaluation time scales with
  ``dim * budget_factor``.
* Lower ``eval_timeout`` so that slow candidates fail early.
* Run problems, seeds and methods in parallel with ``experiment.n_jobs``.
* Methods that make extra LLM calls per evaluation take longer:
  MCTS-AHD's description calls, and ShinkaEvolve with novelty rejection or
  meta recommendations.

.. _list_of_benchmarks:

List of Benchmarks
------------------

Multi-objective optimisation (ZDT, DTLZ, WFG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LLM designs a **multi-objective optimiser** for continuous problems with
2 or 3 conflicting objectives, all minimised. Problems and their known Pareto
fronts come from `pymoo <https://pymoo.org/problems/test_problems.html>`_;
the indicators from `moocore <https://multi-objective.github.io/moocore/python/>`_.
Both are installed in the evaluation environment.

- Budget: ``budget_factor`` × the number of variables per run (default
  2000, as for BBOB): 60 000 evaluations for the 30-variable ZDT1-3, 20 000
  for 10-variable functions. The default ``eval_timeout`` scales with the
  total number of evaluations.
- Interface (as for BBOB): ``__init__(self, budget, dim)`` and
  ``__call__(self, func)``. ``func(x)`` returns the objective vector of one
  point, or one row per point for a batch of shape ``(k, dim)``; every point
  counts towards the budget and is clipped to ``func.bounds.lb`` /
  ``func.bounds.ub``. ``func.meta_data.n_objectives`` is the number of
  objectives.
- BLADE records every evaluation and scores everything that was evaluated;
  what the algorithm returns is ignored.
- Score (maximised, at most 1): the **area under the normalised hypervolume
  curve** over the budget, the multi-objective counterpart of AOCC.
  Objectives are normalised with the ideal and nadir points of the known
  front, the reference point is 1.1 in every objective, and the hypervolume
  is measured at the end of each of ``checkpoints`` (50) equal parts of the
  budget, divided by the hypervolume of the reference front. While no point
  lies inside the reference box, the value is :math:`-d/(1+d)` for the
  distance :math:`d` of the closest point to the box, as in COCO's
  bi-objective indicator, so weak algorithms are still ranked. The fitness
  is the mean over the functions and runs.
- Feedback per function: the area, the final hypervolume, IGD+ to the
  reference front, the number of non-dominated points and the evaluations
  used.

.. list-table::
   :header-rows: 1

   * - Key
     - Problems
   * - ``multiobjective`` (problem)
     - Configurable: ``training_instances`` and ``test_instances`` are pymoo
       functions, e.g. ``[zdt1, {name: dtlz2, n_obj: 3}]``;
       ``budget_factor`` (evaluations per run = ``budget_factor`` × the
       function's number of variables; 2000, as for BBOB), ``runs`` (3),
       ``checkpoints`` (50).
   * - ``mo_quick`` (suite)
     - Design on ZDT1 and ZDT2, test on ZDT3; 2 runs.
   * - ``mo_zdt`` (suite)
     - 2 objectives: design on ZDT1-3, test on ZDT4 and ZDT6.
   * - ``mo_dtlz`` (suite)
     - 3 objectives: design on DTLZ1-4, test on DTLZ5-7.
   * - ``mo_wfg`` (suite)
     - 2 objectives, 10 variables: design on WFG1-5, test on WFG6-9.

.. code-block:: yaml

    problems:
      - {suite: mo_zdt}
      - type: multiobjective
        name: DTLZ2-3obj
        params:
          training_instances: [{name: dtlz2, n_obj: 3}]
          test_instances: [{name: dtlz4, n_obj: 3}]
          budget_factor: 500

For reference, with the default budget (training functions, 3 runs):

.. list-table::
   :header-rows: 1

   * - Algorithm
     - ``mo_zdt``
     - ``mo_dtlz``
     - ``mo_wfg``
   * - Random search (the example)
     - -0.41
     - -0.31
     - +0.51
   * - pymoo NSGA-II (population 100)
     - +0.92
     - +0.02
     - +0.79

NSGA-II reaches the front on ZDT, DTLZ2 and DTLZ4, and on WFG2-5, but not on
the multimodal DTLZ1 and DTLZ3 (about -0.9) or on WFG1, which leaves room for
better algorithms.

Analysis
^^^^^^^^

Methodology class:
:ref:`Structured Scientific Program/Search <taxonomy_structured_scientific_program_search>`.

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
  :math:`\frac{|\max_g|}{I^2}` (the constant :math:`C_3`; the absolute
  value is taken outside the maximum, as in the AlphaEvolve paper)
- Constraints:
  :math:`f` real-valued, :math:`I \ne 0`
- Optimisation direction: minimisation
- Default:
  :math:`N = 400`

-----------

AutoML
^^^^^^

Methodology class: :ref:`ML Pipeline Optimization
<taxonomy_ml_pipeline_optimization>`.

The AutoML benchmark evaluates **LLM-generated scikit-learn pipelines** on supervised
learning tasks hosted on the `OpenML <https://www.openml.org/>`_ platform.

- **Task source:** OpenML tasks (either individual task IDs or suites such as
  ``amlb-classification-all`` from the AutoML Benchmark).
- **What is optimized:** a single Python class implementing a pipeline with:
  ``__init__(self, X, y, **hyperparameters)`` (fit once) and ``__call__(self, X)``
  (predict), using only standard ML libraries (primarily ``scikit-learn``).
- **Evaluation protocol:** uses the **official OpenML split definition**
  (repeat/fold/sample) and the task’s default evaluation measure (e.g., accuracy, AUC,
  RMSE/MAE). Scores are aggregated across all official splits.
- **Hyperparameter tuning (optional):** with ``hpo=True`` (enabled
  automatically for methods that sample with ``HPO``, e.g. LLaMEA with
  ``HPO=True``), a solution's configuration space is tuned with SMAC on up to
  12 of the search splits before it is scored. smac is only installed in the
  evaluation environment when tuning is enabled; see "Hyperparameter tuning
  (HPO)" in :doc:`yaml_experiments`.
- **Dependencies:** ``openml`` (``uv sync --group extraproblems``) to load
  the tasks.

-----------

Black Box and Photonics Optimisation (BBO & Photonics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methodology class:
:ref:`Black-box Continuous <taxonomy_black_box_continuous>`.

- Contains a set of Black Box Optimisation problems from the `ioh benchmarks`_ library.
- Includes both ``BBOB`` and ``mabbob`` benchmark families.
- **Hyperparameter tuning (optional)** for ``BBOB_SBOX`` (``bbob_sbox``) and
  ``MA_BBOB`` (``ma_bbob``): with ``hpo=True`` (enabled automatically for
  methods that sample with ``HPO``, e.g. LLaMEA with ``HPO=True``), a
  solution's configuration space is tuned with SMAC before it is scored.
  Each SMAC trial runs the algorithm once, with the configuration as
  constructor keyword arguments, on one (function, instance, dimension) of
  a fixed, seeded subset of ``hpo_instances`` (default 12) training
  combinations; its cost is ``1 - AOCC``. The incumbent is then scored like
  any solution (mean AOCC on all training instances), stored in the
  solution's metadata (``incumbent``) and named in the feedback;
  ``test()`` uses it on the test instances without tuning again. The
  budget is ``hpo_trials``: 100 for ``bbob_sbox`` (about 0.4 of a default
  training evaluation, 240 runs) and 50 for ``ma_bbob`` (about 1.25 of a
  default training evaluation, 40 runs), plus SMAC's overhead, so consider
  a larger ``eval_timeout``. See "Hyperparameter tuning (HPO)" in
  :doc:`yaml_experiments`.

.. _ioh benchmarks: https://iohprofiler.github.io/IOHexp/

- Also contains a set of **photonics** problems, supported by black bock optimisation.

-----------

Combinatorics
^^^^^^^^^^^^^

Methodology class: :ref:`Black-box Discrete/Mixed-Integer
<taxonomy_black_box_discrete_mixed_integer>`.

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
- A candidate returns the heights of a step function :math:`f` on :math:`N`
  equal bins of :math:`[-1, 1]` (:math:`dx = 2/N`). Any length
  :math:`1 \le N \le` ``max_bins`` (default 100,000) is accepted;
  ``n_bins`` (default 800) is only the suggested resolution.

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

Methodology class:
:ref:`Structured Scientific Program/Search <taxonomy_structured_scientific_program_search>`.

Fourier Uncertainty Inequality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- This benchmark minimises the Fourier Uncertainty Inequality bound
  :math:`r_{\max}^2 / (2\pi)` for :math:`f(x) = P(x) e^{-\pi x^2}`, where
  :math:`r_{\max}` is the largest positive sign change of :math:`P(x)/x^2`
  (AlphaEvolve notebook, section B.4).

- Function Class:

  .. math::

     f(x) = P(x) \, e^{-\pi x^2}, \quad
     P(x) = \sum_{k=0}^{K} c[k] \, H_{4k}(x)

  - :math:`H_n` are (unscaled) physicists' Hermite polynomials.
  - The candidate returns :math:`K` (default 3) coefficients
    :math:`c[0..K-1]`; the evaluator appends :math:`c[K]` so that
    :math:`P(0) = 0`. For :math:`K = 3`:
    :math:`c[3] = -(c_0 + 12 c_1 + 1680 c_2) / 665280`.
  - Evenness holds by construction (degrees 0,4,8,…).

- Scoring:

  1. If the leading coefficient of :math:`P` is negative, :math:`P` is replaced by :math:`-P`.
  2. :math:`r_{\max}` is the largest positive sign change of :math:`P(x)/x^2`; a candidate without one is invalid.
  3. Scale invariant: any nonzero scaling of :math:`c` leaves the score unchanged.

-----------

Geometry
^^^^^^^^

Methodology class:
:ref:`Structured Scientific Program/Search <taxonomy_structured_scientific_program_search>`.

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

Methodology class: :ref:`Black-box Discrete/Mixed-Integer
<taxonomy_black_box_discrete_mixed_integer>`.

- This benchmark evaluates **metaheuristic algorithms for hardware kernel tuning** across integer and variable-dimensional search spaces with constraints.
- The algorithm is scored based on performance of the kernel using metrices like:
    - runtime
    - throughput
    - custom matrices.
- **Hyperparameter tuning (optional):** with ``hpo=True`` (or a method that
  samples with ``HPO``), a solution's configuration space is tuned with SMAC
  on a fixed subset of ``hpo_instances`` (default 12) training kernel-GPU
  pairs, ``hpo_trials`` (default 50) autotuning-methodology runs of one pair
  each, with cost ``1 - score``. The configuration is passed to the
  strategy as its ``search_method_hyperparameters``; the incumbent is then
  scored on all kernels and GPUs and recorded in the metadata
  (``incumbent``).

-----------

Logistics
^^^^^^^^^

Methodology class: :ref:`Black-box Discrete/Mixed-Integer
<taxonomy_black_box_discrete_mixed_integer>`.

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

Methodology class:
:ref:`Structured Scientific Program/Search <taxonomy_structured_scientific_program_search>`.

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

Methodology class:
:ref:`Structured Scientific Program/Search <taxonomy_structured_scientific_program_search>`.

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
  - Constraints: :math:`0 \in U`, :math:`|U| \le` ``max_set_size`` (default
    2048) and :math:`\max(U) \le 2^{21}`. The seed (AlphaEvolve's
    2,003-element set) scores :math:`c(U) = 1.147989`.

-----------

Packing
^^^^^^^

Methodology class: :ref:`Black-box Discrete/Mixed-Integer
<taxonomy_black_box_discrete_mixed_integer>`.

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
