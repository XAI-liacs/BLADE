Coding agents
=============

Coding agents such as Claude Code can set up BLADE experiments for you:
write the problem class and the experiment file, validate and smoke-test
them, and do a short trial run. BLADE ships a *skill*, a short manual the
agent reads when you ask it about BLADE, so that it follows the same steps
as this documentation.

Install the skill
-----------------

.. code-block:: bash

   uv run iohblade skill                    # writes .claude/skills/blade/ (Claude Code)
   uv run iohblade skill .agents/skills     # or another agent's skills folder
   uv run iohblade skill --force            # update it after upgrading BLADE

The skill (``SKILL.md`` plus ``reference.md``) comes with the package, so it
matches the installed version. In a clone of the BLADE repository it is
already in ``.claude/skills/blade/``.

Ask
---

Describe what you want in plain words, for example:

   *Benchmark my scheduling heuristic problem with LLaMEA and ShinkaEvolve
   on a local Qwen model, 5 runs each. The instances are in instances/.*

The skill tells the agent to:

1. start from ``iohblade init`` with the right template;
2. write the problem (prompts and ``evaluate()``), method or LLM code and
   ``experiment.yml``;
3. run ``iohblade validate`` and ``iohblade check`` and fix what they
   report;
4. do a short trial run in its own output folder
   (``--set experiment.runs=1 --set experiment.budget=4``);
5. hand you the command for the full run instead of starting it, because a
   full run can take hours and many LLM calls.

Review before you run
---------------------

Read what the agent wrote before the full run:

- **The task prompt**: is it the problem as you would explain it to a
  colleague? It is all the LLM knows about the problem.
- **evaluate()**: does it score what you care about? Code that runs but
  measures the wrong thing makes every method optimise the wrong goal.
- **experiment.yml**: methods, model, number of runs and budget.

The agent can also explain errors, resume or extend experiments, and load
results for analysis; see :doc:`troubleshooting` and :doc:`webapp`.
