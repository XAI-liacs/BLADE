# Episode 3: Local and hosted LLMs

**Goal:** the viewer can connect any supported LLM: a hosted API (OpenAI,
Claude, Gemini, DeepSeek), a local model through Ollama or LM Studio, or an
open model served by vLLM on their own GPUs, and spread the load over
several servers with `provider: multi`.

**Length:** about 5.5 minutes including the standard opening.

**Files:** `experiment.yml` in this folder (five LLMs, one LLaMEA per LLM).

**Setup before recording:**

- A `.env` file next to `experiment.yml` with `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. **Never show its contents on
  screen;** blur it if the file tree shows it.
- For scene 6: two vLLM servers (ports 8000 and 8001), or show the commands
  without running them.
- Update the model names to the current ones on the day of recording.

Outputs below were captured on `main` on 25 November 2026.

---

## Scene 0 — Standard opening (0:00–0:15)

Shared opening, then the title card "Episode 3 · Local and hosted LLMs".

## Scene 1 — Why the LLM matters (0:15–0:45)

**Screen:** the LLM row of the platform figure (the provider logos).

**Narration:**

> The LLM is the engine of every method in BLADE. A stronger model writes
> better algorithms, but a run makes hundreds or thousands of calls, so
> cost, speed and reproducibility matter too. BLADE treats every model the
> same way: you name it once in the YAML file, and any method can use it.
> So comparing two LLMs is as easy as comparing two methods.

## Scene 2 — Hosted APIs (0:45–1:40)

```yaml
llms:
  gpt:
    provider: openai
    model: gpt-5-mini
    api_key: "${env:OPENAI_API_KEY}"
  claude:
    provider: claude
    model: claude-sonnet-5-5
    api_key: "${env:ANTHROPIC_API_KEY}"
  gemini:
    provider: gemini
    model: gemini-2.5-flash
    api_key: "${env:GEMINI_API_KEY}"
```

**Narration:**

> For hosted models you need a provider, a model name and an API key.
> OpenAI, Anthropic's Claude, Google's Gemini and DeepSeek are built in.
>
> The keys come from the environment or from a `.env` file next to the
> experiment, never from the YAML file itself. When BLADE records the
> configuration of a run, values read from the environment are replaced by
> "redacted", so you can share the results folder safely. Keep the `.env`
> file out of version control.

## Scene 3 — Local models with Ollama (1:40–2:15)

```console
$ ollama pull qwen2.5-coder:14b
```

```yaml
  qwen-ollama:
    provider: ollama
    model: "qwen2.5-coder:14b"
```

**Narration:**

> For local models, Ollama is the easiest start: pull a model, and use it
> with provider `ollama`. No key, no costs per call, and the model does not
> change under you between experiments, which helps reproducibility. On a
> Mac, LM Studio and MLX work the same way.

## Scene 4 — Testing the connection (2:15–2:45)

```console
$ uv run iohblade check experiment.yml --llm
```

Expected: one line per LLM, `[OK  ] llm <name>: answered 'OK'`. Record this
with working keys and servers; capture the real output.

**Narration:**

> `check` normally does not contact any LLM. With `--llm`, it sends each
> model one short prompt, so a wrong key, a typo in a model name or a
> server that is not running shows up now, not two hours into a run.

## Scene 5 — vLLM: open models on your own GPUs (2:45–3:50)

**Screen:** two terminals, then the YAML.

```console
$ CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-32B --port 8000
$ CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-32B --port 8001
```

```yaml
  gpu0:
    provider: openai
    model: Qwen/Qwen3-32B
    params: {base_url: "http://localhost:8000/v1"}
  gpu1:
    provider: openai
    model: Qwen/Qwen3-32B
    params: {base_url: "http://localhost:8001/v1"}
```

**Narration:**

> For larger experiments, vLLM serves open models much faster than Ollama.
> vLLM speaks the OpenAI protocol, so we use provider `openai` with a
> `base_url` pointing at our own server. No key is needed. The same works
> for SGLang, llama.cpp's server, or any other OpenAI-compatible endpoint.
>
> Here we start one server per GPU.

## Scene 6 — Spreading the load with provider: multi (3:50–4:35)

```yaml
  qwen-vllm:
    provider: multi
    llms: [gpu0, gpu1]

experiment:
  n_jobs: 8
```

**Narration:**

> `provider: multi` combines several LLMs into one: each call goes to one
> of the servers, picked at random, so the load spreads over them. With `n_jobs: 8`, eight runs share the two GPUs, and
> both stay busy. With four GPUs, add four servers; nothing else changes.

## Scene 7 — Comparing LLMs (4:35–5:15)

```yaml
methods:
  - type: llamea
    name: "LLaMEA-{llm}"
    llm: [gpt, claude, gemini, qwen-ollama, qwen-vllm]
```

```console
$ uv run iohblade validate experiment.yml
...
Methods (5):
  - LLaMEA-gpt (LLM: gpt, budget: 50)
  - LLaMEA-claude (LLM: claude, budget: 50)
  - LLaMEA-gemini (LLM: gemini, budget: 50)
  - LLaMEA-qwen-ollama (LLM: qwen-ollama, budget: 50)
  - LLaMEA-qwen-vllm (LLM: qwen-vllm, budget: 50)
Problems (1):
  - circle_packing_n40
Seeds: [0, 1, 2]
Runs: 5 methods x 1 problems x 3 seeds = 15

Configuration is valid.
```

**Narration:**

> And this is how you compare LLMs: one method, a list of models. BLADE
> creates one LLaMEA per model, with the same budget and the same
> problems. Each run's conversation log also records the tokens used, and
> `plot_token_usage` turns that into a chart, so you can weigh quality
> against cost.

## Scene 8 — Outro (5:15–5:30)

**Narration:**

> Next episode: bringing your own problem to BLADE, so these methods and
> models work on the problems you care about.

---

## Notes for the recording

- `check --llm` with invalid keys retries for several minutes before it
  fails; record it with working keys.
- Scene 7's output was captured with dummy key values in `.env`; it does
  not contact the APIs.
- Token usage: check that the conversation logs of the recorded runs hold
  token counts (some local servers do not report them), or drop that
  sentence.
