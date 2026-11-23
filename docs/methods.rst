Search Methods
==============

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
the `reference code <https://github.com/google-deepmind/funsearch>`_ and the
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
