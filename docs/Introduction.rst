
Introduction
============

**BLADE** is a Python framework for benchmarking the Llm Assisted Design and
Evolution of algorithms.

An experiment combines **LLMs**, LLM-driven **methods** (such as LLaMEA) and
**problems** (benchmarks). The recommended way to define one is a YAML file
that names these components and is validated, smoke-tested and run with the
``iohblade`` command (see :doc:`Quickstart` and :doc:`yaml_experiments`).
The same building blocks are available as a Python API for scripting and
for writing new methods and problems.

**Key Features:**

- **Experiments as configuration**: YAML files and the ``iohblade`` command,
  with clear error messages and smoke tests before any LLM time is spent.
- **Fast and simple**: BLADE is designed to be easy to use and fast to run.
- **Flexible**: BLADE is designed to be flexible and can be easily extended.
- **Scalable**: BLADE is designed to be scalable and can be run on a single machine or a cluster.
- **Open Source**: BLADE is open source and free to use.
- **HLP benchmarks**: Includes high-level property (HLP) benchmark
  generation for structured black-box optimization tasks.
