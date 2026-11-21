Problem
==========

Evaluation runs in a separate process inside a virtual environment.
Two attributes control this:

- ``extra_pythonpath`` (list of directories, default empty): added to
  ``PYTHONPATH`` of the evaluation process. Set it when the problem class is
  defined in a module outside the ``iohblade`` package, so the evaluation
  environment can import it.
- ``max_harness_errors`` (default ``3``): an evaluation that fails before
  the problem has read ``solution.code`` is a *harness error*, a fault in
  the problem or its environment that the LLM cannot fix. After this many
  consecutive harness errors the run stops with
  :class:`~iohblade.utils.HarnessError`. ``None`` disables the check.

.. automodule:: iohblade.problem
   :members:
   :undoc-members:
   :show-inheritance:

