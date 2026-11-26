Methods (API)
=============

This page documents the method implementations exposed in
``iohblade.methods``.

.. automodule:: iohblade.methods
   :members:
   :undoc-members:
   :show-inheritance:

FunSearch
---------

:class:`iohblade.methods.FunSearch` implements FunSearch
(Romera-Paredes et al., *Nature* 625, 2024). Its programs database follows
the `reference code <https://github.com/google-deepmind/funsearch>`__ and the
`LLM4AD port <https://github.com/Optima-CityU/llm4ad>`_:

* **Islands and clusters.** ``num_islands`` independent islands; on each
  island programs with the same score form a cluster.
* **Best-shot prompts.** A prompt shows ``programs_per_prompt`` programs of
  one island as versions, worst first. They are drawn by a softmax over the
  cluster scores, with a temperature that starts at ``temperature_init`` and
  decays linearly over ``temperature_period`` programs; within a cluster,
  shorter programs are preferred. The LLM is asked for the next, improved
  version, and ``samples_per_prompt`` programs are sampled per prompt.
* **Island resets.** Every ``reset_period`` evaluations (default
  ``budget // 5``; ``0`` disables resets) the weaker half of the islands is
  restarted from the best program of a stronger island.

Differences with the reference implementation: a program is a complete
solution (for example a class) rather than one function in a skeleton; the
first program is sampled from the problem prompt; resets are counted in
evaluations, not wall-clock time, so runs are reproducible; and samples are
evaluated one at a time. Exactly ``budget`` solutions are evaluated. Failed
evaluations are not added to the database. Only single-objective problems
are supported.

.. code-block:: yaml

   methods:
     - type: funsearch
       llm: gpt
       budget: 200
       params: {num_islands: 10, programs_per_prompt: 2, samples_per_prompt: 4}

ShinkaEvolve
------------

:class:`iohblade.methods.ShinkaEvolve` implements ShinkaEvolve (Lange et
al., `arXiv:2509.19349 <https://arxiv.org/abs/2509.19349>`_), reimplemented
after the `reference code <https://github.com/SakanaAI/ShinkaEvolve>`__
(version 0.0.8). The ``shinka`` package itself cannot be installed next to
BLADE (it requires ``openai>=2`` and ``google-genai>=2``), and its budget
counts generations rather than evaluations. Each generation:

* **chooses an LLM** with an asymmetric UCB bandit when the LLM is a
  ``provider: multi`` LLM (the reward is a child's improvement over its
  parent and the initial program);
* **samples a parent** from an island's archive programs
  (``parent_selection``: ``weighted`` by default, ``power_law`` or
  ``beam_search``) and **inspirations**
  (``num_archive_inspirations`` + ``num_top_k_inspirations``), shown worst
  first;
* **patches** the parent with a SEARCH/REPLACE diff, a full rewrite (one of
  five prompt variants) or a crossover (``patch_type_probs``, default
  0.6/0.3/0.1); islands without a correct program fix a failed program;
* optionally **rejects candidates that are not novel**
  (``novelty_rejection``): a code embedding more similar than
  ``novelty_threshold`` to a program on the island is rejected, unless the
  LLM judges it meaningfully different;
* optionally adds one of the **meta recommendations** that the LLM writes
  every ``meta_interval`` evaluations from summaries of the new programs.

Differences with the reference implementation:

* **Budget.** Exactly ``budget`` solutions are evaluated, one per
  generation. When no reply can be applied, the code of the last reply is
  evaluated; when every candidate is rejected as not novel, the last one
  is evaluated. Rejected candidates are not evaluated, but their LLM calls
  are logged.
* **Embeddings** use a local character n-gram embedding by default (no
  embedding API, no extra cost); pass ``embedding=`` a function from code
  to a vector in Python to use another.
* **Programs** are complete solutions without EVOLVE-BLOCK markers. Fitness
  is shown with the problem's optimisation direction.
* **Bandit:** the linear reward variant without API-cost awareness, and no
  random temperature per query.
* **No asynchronous jobs:** proposals and evaluations run one at a time.
* **No files written:** nothing is written to disk (ShinkaEvolve writes a
  SQLite database and a folder per generation).

Novelty rejection and meta recommendations are off by default, as in
ShinkaEvolve's packaged defaults; turning them on costs extra LLM calls
(a judge call per similar candidate, and ``meta_interval`` + 2 calls per
update). Only single-objective problems are supported.

.. code-block:: yaml

   llms:
     mix:
       provider: multi
       llms: [gpt, gemini]
   methods:
     - type: shinka_evolve
       llm: mix
       budget: 100
       params:
         num_islands: 2
         novelty_rejection: true
         meta_interval: 10
