Registry
==========

The registry maps short keys to BLADE's built-in components, so they can be
looked up by name (for example from a configuration file) without importing
every optional dependency up front.

.. code-block:: python

    from iohblade.registry import BuiltinMethod, LLMProvider, registry

    LLaMEA = registry.resolve("method", BuiltinMethod.LLAMEA)
    Gemini = registry.resolve("llm", LLMProvider.GEMINI)

    registry.keys("problem")  # all problem keys

Unknown keys raise :class:`~iohblade.utils.ComponentNotFoundError` with close
matches; components whose optional dependencies are missing raise
:class:`~iohblade.utils.ComponentUnavailableError` with the install command.

Register your own components with the decorators:

.. code-block:: python

    from iohblade.problem import Problem
    from iohblade.registry import register_problem

    @register_problem("my_problem")
    class MyProblem(Problem):
        ...

.. automodule:: iohblade.registry
   :members:
   :undoc-members:
   :show-inheritance:
