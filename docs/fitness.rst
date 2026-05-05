Fitness (Multi-Objective)
=========================

BLADE supports multi-objective optimisation through the :class:`~iohblade.fitness.Fitness`
class.  When a problem returns multiple objectives, its :meth:`evaluate` method sets
``solution.fitness`` to a ``Fitness`` object instead of a plain ``float``.

Overview
--------

``Fitness`` wraps a ``dict[str, float]`` mapping objective names to objective values.
It defines Pareto-dominance comparisons so that standard Python operators work
intuitively:

.. list-table::
   :header-rows: 1
   :widths: 15 45

   * - Expression
     - Meaning (minimisation)
   * - ``a < b``
     - ``a`` **strictly dominates** ``b``: better-or-equal on every objective and
       strictly better on at least one.
   * - ``a > b``
     - ``b`` strictly dominates ``a`` (maximisation direction).
   * - ``a <= b``
     - ``a`` dominates ``b`` or ``a`` is better than ``b`` on at least one objective.
   * - ``a == b``
     - Identical objective values on every objective.
   * - ``float(a)``
     - Scalar summary: mean of all objectives (``NaN`` if any objective is ``NaN``).

.. note::
   Do **not** use Python's built-in ``sort`` on ``Fitness`` objects.  Sorting
   requires a total order, which multi-objective dominance cannot guarantee.
   Use dedicated Pareto-sorting utilities instead.

Quick Example
-------------

.. code-block:: python

    from iohblade import Fitness

    a = Fitness({"f1": 1.0, "f2": 3.0})  # good on f1, weak on f2
    b = Fitness({"f1": 3.0, "f2": 1.0})  # weak on f1, good on f2

    # Neither dominates the other (incomparable → both on the Pareto front)
    print(a < b)    # False
    print(b < a)    # False

    # Scalar summary (mean)
    print(float(a)) # 2.0
    print(float(b)) # 2.0

    # Serialise / deserialise (e.g. for JSON logging)
    d = a.to_dict()           # {"f1": 1.0, "f2": 3.0}
    a2 = Fitness.from_dict(d) # reconstructed Fitness

Writing a Multi-Objective Problem
----------------------------------

Return a ``Fitness`` object from ``evaluate()``:

.. code-block:: python

    from iohblade import Fitness, Solution
    from iohblade.problem import Problem


    class MyBiObjectiveProblem(Problem):
        def evaluate(self, solution: Solution) -> Solution:
            # ... run the generated algorithm ...
            best_f1, best_f2 = run_algorithm(solution.code)

            # BLADE maximises: negate so lower raw values → higher fitness
            fitness = Fitness({"f1": -best_f1, "f2": -best_f2})
            return solution.set_scores(fitness=fitness, feedback="...")

Toy Benchmark
-------------

A ready-to-use toy bi-objective problem is provided for testing:

.. code-block:: python

    from iohblade.benchmarks.toy_multiobjective import ToyMultiObjective

    problem = ToyMultiObjective(budget=100)

The problem asks the LLM to generate an algorithm (``BiSphereSearcher``) that
simultaneously minimises:

* **f1** = :math:`x^2`  (distance to 0)
* **f2** = :math:`(x - 2)^2`  (distance to 2)

These objectives conflict: the optimal *x* for *f1* is 0, while for *f2* it is 2.
The Pareto front lies on :math:`x \in [0, 2]`.

Plotting
--------

Use :func:`~iohblade.plots.plot_pareto_front` to visualise the discovered Pareto
front:

.. code-block:: python

    from iohblade.plots import plot_pareto_front

    plot_pareto_front(
        logger=exp_logger,
        objective_x="f1",
        objective_y="f2",
    )

The plot highlights non-dominated solutions (the empirical Pareto front) and
connects them with a step curve for visual clarity.

Convergence plots (via :func:`~iohblade.plots.plot_convergence`) remain available
for multi-objective runs: they use the **mean of all objectives** as the scalar
convergence metric.

Logging
-------

``Fitness`` objects are serialised as plain JSON dicts in ``log.jsonl``:

.. code-block:: json

    {"fitness": {"f1": -0.04, "f2": -1.96}, "name": "BiSphereSearcher", ...}

When :meth:`~iohblade.loggers.ExperimentLogger.get_problem_data` reads this file
it automatically adds a ``fitness_scalar`` column containing the mean objective
value, which is what the convergence and speedup plots consume.

API Reference
-------------

.. autoclass:: iohblade.fitness.Fitness
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iohblade.benchmarks.toy_multiobjective.ToyMultiObjective
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: iohblade.plots.plot_pareto_front
.. autofunction:: iohblade.plots.plot_convergence
.. autofunction:: iohblade.loggers.ExperimentLogger.get_problem_data
