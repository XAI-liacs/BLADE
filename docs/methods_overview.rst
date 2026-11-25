Methods
=======

A *method* is an LLM-driven algorithm design strategy: it asks an LLM for
algorithms, evaluates them on a problem and uses the results to ask for
better ones. BLADE has nine built-in methods. All of them get the same LLM,
the same problems and the same evaluation budget, so they can be compared
fairly.

.. list-table::
   :header-rows: 1
   :widths: 18 16 46 20

   * - Method
     - YAML ``type``
     - Idea
     - Reference
   * - **LLaMEA**
     - ``llamea``
     - Evolutionary algorithm over code: parents are refined or combined
       by the LLM with configurable mutation and crossover prompts.
     - `code <https://github.com/nikivanstein/LLaMEA>`__,
       `paper <https://arxiv.org/abs/2405.20132>`__
   * - **EoH**
     - ``eoh``
     - Evolution of Heuristics: evolves a "thought" and its code together,
       with exploration and modification prompts.
     - `code <https://github.com/FeiLiu36/EoH>`__,
       `paper <https://arxiv.org/abs/2401.02051>`__
   * - **ReEvo**
     - ``reevo``
     - Reflective evolution: short- and long-term reflections on pairs of
       solutions guide crossover and mutation.
     - `code <https://github.com/ai4co/LLM-as-HH>`__,
       `paper <https://arxiv.org/abs/2402.01145>`__
   * - **FunSearch**
     - ``funsearch``
     - Islands of programs clustered by score; best-shot prompts show
       earlier versions and ask for the next one. Reimplemented in BLADE.
     - `code <https://github.com/google-deepmind/funsearch>`__,
       `paper <https://www.nature.com/articles/s41586-023-06924-6>`__
   * - **ShinkaEvolve**
     - ``shinka_evolve``
     - Sample-efficient program evolution: islands, weighted parent
       selection, diff, full and crossover patches, novelty rejection and
       an LLM bandit. Reimplemented in BLADE.
     - `code <https://github.com/SakanaAI/ShinkaEvolve>`__,
       `paper <https://arxiv.org/abs/2509.19349>`__
   * - **LHNS**
     - ``lhns``
     - LLM-driven heuristic neighbourhood search: destroy and repair parts
       of the best code, accepted by simulated annealing.
     - `code <https://github.com/Acquent0/LHNS>`__,
       `paper <https://ieeexplore.ieee.org/abstract/document/11043025>`__
   * - **MCTS-AHD**
     - ``mcts_ahd``
     - Monte Carlo tree search over algorithms, so that weaker branches
       can still be explored.
     - `code <https://github.com/zz1358m/MCTS-AHD-master/tree/main>`__,
       `paper <https://arxiv.org/abs/2501.08603>`__
   * - **MoEH**
     - ``moeh``
     - Multi-objective evolution of heuristics: a Pareto front of
       algorithms over several objectives.
     - `paper <https://arxiv.org/pdf/2409.16867>`__
   * - **Random search**
     - ``random_search``
     - Asks the LLM for independent algorithms, without feedback. The
       baseline every method should beat.
     -

EoH, ReEvo and MoEH need the optional ``methods`` dependency group
(``uv sync --group methods``); ``uv run iohblade list methods`` shows this
too.

Using a method
--------------

In an experiment file, a method names its ``type``, the LLM(s) it uses and
optionally its own ``params`` and ``budget``:

.. code-block:: yaml

   methods:
     - type: llamea
       name: "LLaMEA-{llm}"          # {llm} is replaced by the LLM name
       llm: [qwen, gemini]           # one method per LLM
       params: {n_parents: 4, n_offspring: 12, elitism: true}
     - type: funsearch
       llm: qwen
       params: {num_islands: 10}
     - type: random_search
       llm: qwen

``params`` are checked against the method's constructor, so a misspelled
parameter is reported by ``iohblade validate``. The parameters of each
method are listed in the :doc:`API reference <methods>`; LLaMEA's operators
are described in :doc:`yaml_experiments`.

Comparing fairly
----------------

- **Budget.** ``budget`` is the number of generated algorithms a run may
  evaluate. Every method uses ``experiment.budget`` unless it sets its own;
  keep it the same for a comparison. Evaluations beyond the budget return a
  failure score.
- **Seeds.** Use several runs (``runs`` or ``seeds``): LLM-driven search is
  noisy, and one run says little.
- **Same LLM.** Compare methods with the same LLM, or compare LLMs with the
  same method, not both at once.
- **Baseline.** Include ``random_search``. A method that does not beat it
  with the same budget is not using the feedback.

Your own method
---------------

Any subclass of :class:`iohblade.method.Method` can be used with
``type: custom``; see :doc:`tutorial_custom_method`.
