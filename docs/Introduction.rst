Introduction
============

**LLM-driven automated algorithm design** uses a large language model to
write optimisation algorithms: the LLM proposes an algorithm as code, the
algorithm is evaluated on a problem, and the score and feedback go back
into the next prompt, so that the LLM can write a better one. Methods such
as LLaMEA, EoH, ReEvo, FunSearch and ShinkaEvolve differ in *how* they
prompt, select and combine algorithms.

**BLADE** (Benchmarking LLM-driven Automated Design and Evolution) is a
benchmark suite for these methods: it runs them on the same problems, with
the same LLM and the same evaluation budget, evaluates every generated
algorithm in isolation and records everything, so that methods can be
compared fairly and their results can be inspected.

Concepts
--------

**Problem**
   A benchmark: what the LLM is asked to write (the prompt), and how a
   generated algorithm is scored (``evaluate``). BLADE has continuous
   black-box problems (BBOB, MA-BBOB), mathematical problems from
   AlphaEvolve, combinatorial problems and more; see :doc:`benchmarks`.

**Method**
   An algorithm design strategy that uses an LLM, such as LLaMEA or
   FunSearch; see :doc:`methods_overview`.

**LLM**
   A hosted or local model the method uses; see :doc:`llm`.

**Solution**
   One generated algorithm: its code, name, description, fitness, feedback
   and parents.

**Run, budget and seed**
   One method on one problem with one seed is a *run*. Its *budget* is the
   number of algorithms it may generate and evaluate. Several seeds per
   method and problem show how much results vary.

**Experiment**
   A set of runs: methods × problems × seeds. It is described in a YAML
   file and run with the ``iohblade`` command (see :doc:`yaml_experiments`),
   or built in Python (see :doc:`modules`).

How an evaluation works
-----------------------

Generated code is untrusted: it can crash, hang or use too much memory.
BLADE evaluates every algorithm in a separate process, in a virtual
environment with the problem's dependencies, with a time limit. Errors
become feedback for the LLM instead of stopping the experiment. Failures
caused by the problem itself (*harness errors*) stop a run early, so a
broken benchmark does not use up the budget.

Contributing
------------

Contributions are welcome: bug reports and feature requests on
`GitHub Issues <https://github.com/XAI-liacs/BLADE/issues>`_, new benchmark
problems and methods as pull requests. See ``CONTRIBUTING.md`` in the
repository for the guidelines.

License
-------

BLADE is distributed under the `MIT
<https://choosealicense.com/licenses/mit/>`_ license.

Citing BLADE
------------

If you use BLADE in your research, please cite:

.. code-block:: bibtex

   @inproceedings{vanstein2025blade,
     author    = {Niki van Stein and Anna V. Kononova and Haoran Yin and Thomas B{\"a}ck},
     title     = {BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics},
     booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
     series    = {GECCO '25 Companion},
     year      = {2025},
     pages     = {2336--2344},
     publisher = {Association for Computing Machinery},
     address   = {New York, NY, USA},
     doi       = {10.1145/3712255.3734347},
     url       = {https://doi.org/10.1145/3712255.3734347}
   }
