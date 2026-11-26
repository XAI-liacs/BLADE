Analysing results
=================

An experiment's output folder holds everything a run produced: every
generated algorithm with its code, score, feedback and parent, the
conversation with the LLM, logs and the exact configuration. You can browse
it in the web app, or load it in Python for plots and tables.

What is in the output folder
----------------------------

``<output_dir>/`` holds ``experiment.log``, ``progress.json`` (the status of
every run), ``blade_config/`` (the resolved configuration with secrets
redacted, and copies of custom sources) and one folder per run with:

- ``log.jsonl``: every evaluated algorithm (code, fitness, feedback, parent
  ids);
- ``conversationlog.jsonl``: every prompt and answer;
- ``run.log``, and ``error.txt`` if the run failed;
- ``problem.json``, ``method.json``, ``llm.json``: the configuration of
  each component.

The web app
-----------

.. code-block:: bash

   uv run iohblade-webapp      # or: python -m iohblade.webapp

The app lists the experiments in the subfolders of ``./results`` (relative
to the directory it is started in) that contain a ``progress.json`` file, so
use an ``output_dir`` such as ``results/my-experiment``. It works while an
experiment runs and after it has finished.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item::

      **Monitor** a running experiment: overall progress, and per run the
      evaluations, best score so far and failed evaluations. Under
      *Configuration*, the exact setup including custom code, and the
      experiment log.

      .. image:: blade-working.png
         :alt: The web app showing the progress of a running experiment.
         :width: 100%

   .. grid-item::

      **Compare** finished runs: convergence per method (filter methods
      and problems; *Aggregate runs* shows the mean and spread over seeds),
      the code evolution graph of a run, the best algorithms to download,
      and links to the LLM conversation of every run.

      .. image:: blade-analyze.png
         :alt: The web app comparing methods with convergence plots.
         :width: 100%

**Code evolution graph.** Every dot is a generated algorithm, placed in the
order it was written; lines connect it to its parents, the colour is its
score and the height a property of its code (token count, complexity or
number of parameters). Bigger dots had more children. It shows *how* a
method searched, not only how well.

In Python
---------

The same results open in Python with :class:`~iohblade.loggers.ExperimentLogger`
in read mode, and :mod:`iohblade.plots` makes the plots and tables:

.. code-block:: python

   from iohblade.loggers import ExperimentLogger
   from iohblade.plots import (
       fitness_table,
       plot_boxplot_fitness,
       plot_convergence,
       plot_experiment_CEG,
       plot_token_usage,
   )

   logger = ExperimentLogger("results/my-experiment", read=True)

   plot_convergence(logger)       # best-so-far per method, mean over seeds
   plot_boxplot_fitness(logger)   # final scores per method and problem
   plot_experiment_CEG(logger)    # code evolution graphs
   plot_token_usage(logger)       # tokens used per method and problem
   fitness_table(logger)          # mean ± std, best in bold (with p-values)

``logger.get_data()`` returns one row per run as a pandas ``DataFrame``, and
``logger.get_problem_data(name)`` every evaluated algorithm of a problem,
for your own analysis. The :doc:`notebooks/visualization_options` notebook
shows the options of each plot.

For continuous problems, :mod:`iohblade.behaviour_metrics` describes how a
generated optimiser searches, for example its exploration and exploitation,
convergence rate and improvement statistics.

Multi-objective problems have a Pareto front plot; see :doc:`fitness`.
