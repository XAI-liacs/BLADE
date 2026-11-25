# Episode 7: BLADE with a coding agent

**Goal:** the viewer installs the BLADE skill for their coding agent and
lets the agent turn a one-sentence request into a checked experiment, while
staying in control of what runs.

**Length:** about 5 minutes including the standard opening.

**Setup before recording:**

- An empty project folder with BLADE installed, and Claude Code (or another
  agent that reads skills) logged in.
- A short description of a problem of your own to hand to the agent, e.g.
  a scheduling heuristic with a small instance file in the folder.
- Record the agent session once in full first: agent output differs every
  time, so the narration below describes the steps, not exact text. Cut
  long thinking phases.

`iohblade skill` output below was captured on `main` on 25 November 2026.

---

## Scene 0 — Standard opening (0:00–0:15)

Shared opening, then the title card "Episode 7 · BLADE with a coding
agent".

## Scene 1 — Why an agent (0:15–0:40)

**Narration:**

> In the previous episodes we wrote every file ourselves. Coding agents
> such as Claude Code can do a lot of that work, if they know how BLADE
> works. That is what the BLADE skill is for: a short manual the agent
> reads when you ask it about BLADE.

## Scene 2 — Installing the skill (0:40–1:15)

```console
$ uv run iohblade skill
Installed the BLADE skill:
  .claude/skills/blade/SKILL.md
  .claude/skills/blade/reference.md

Ask your coding agent, e.g. 'Set up a BLADE experiment for my problem'.
Other agents: pass their skills folder, e.g. `iohblade skill .agents/skills`.
```

Open `SKILL.md` and scroll briefly.

**Narration:**

> One command. It writes the skill into the folder where Claude Code looks
> for skills; for other agents, pass their skills folder. The skill
> explains the YAML format, the commands, how to write a problem or a
> method, and how to read BLADE's errors. It comes with the package, so it
> always matches the version you installed.

## Scene 3 — The request (1:15–1:45)

**Screen:** the agent's prompt.

> Benchmark my scheduling heuristic problem with LLaMEA and ShinkaEvolve on
> a local Qwen model, 5 runs each. The instances are in `instances/`.

**Narration:**

> Now we ask, in plain words: our problem, two methods, a local model,
> five runs.

## Scene 4 — What the agent does (1:45–3:15)

**Screen:** the agent session (cut down), showing in order:

1. it reads the skill;
2. `iohblade init --template custom-problem`;
3. it writes `problem.py` (prompts, `evaluate()`) and `experiment.yml`;
4. `iohblade validate`, fixing what it reports;
5. `iohblade check`, ending in "ready to run";
6. a short trial run (`--set experiment.runs=1 --set experiment.budget=4`,
   in its own `output_dir`);
7. it gives you the command for the full run.

**Narration:**

> The agent follows the same steps we did in episodes one and four. It
> starts from the template, writes the problem class with a task prompt
> and an `evaluate` method for our instances, and the experiment file with
> the methods and the model we asked for.
>
> Then it validates and smoke-tests, and fixes what fails. That loop is
> where the skill pays off: BLADE's error messages say exactly what is
> wrong and where, so the agent can fix its own mistakes.
>
> Before the real experiment it does a short trial run: one run, a tiny
> budget. And for the full run, which can take hours and many LLM calls,
> it hands you the command instead of starting it.

## Scene 5 — Review before you run (3:15–4:15)

**Screen:** open `problem.py` and `experiment.yml`; point at the task
prompt, `evaluate()` and the methods list.

**Narration:**

> Before starting, read what the agent wrote. Two things deserve a careful
> look. The task prompt: does it describe your problem the way you would
> explain it to a colleague? And `evaluate`: does it score what you care
> about? An agent can write a scoring function that runs but measures the
> wrong thing, and every method in the experiment then optimises for that.
>
> The experiment file is short, so check it too: methods, model, runs and
> budget. Then start the run yourself.

```console
$ uv run iohblade run experiment.yml
```

## Scene 6 — Outro of the series (4:15–4:45)

**Screen:** the banner, the documentation home page, the GitHub page.

**Narration:**

> That concludes this series. We ran a first benchmark, wrote experiments
> in YAML, connected hosted and local LLMs, added our own problem and our
> own method, analysed the results, and let an agent do the setup. Links
> to the code, the documentation and the other episodes are below. We are
> curious what you will build with it, so let us know on GitHub.

---

## Notes for the recording

- `iohblade skill --force` overwrites an existing skill (after upgrading
  BLADE).
- The agent's exact commands and fixes will differ per recording; keep the
  seven steps of scene 4 recognisable in the cut.
- Keep any API keys out of the agent session on screen.
