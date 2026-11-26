LLMs
====

Every method needs an LLM. In an experiment file, each LLM gets a name of
your choice, a ``provider`` and a ``model``; methods refer to it by name.
API keys are read from the environment or a ``.env`` file next to the
experiment, never from the YAML file itself:

.. code-block:: yaml

   llms:
     claude:
       provider: claude
       model: claude-sonnet-5-5
       api_key: "${env:ANTHROPIC_API_KEY}"
     qwen:
       provider: ollama
       model: "qwen2.5-coder:14b"

Providers
---------

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - ``provider``
     - Class
     - Notes
   * - ``openai``
     - :class:`~iohblade.llm.OpenAI_LLM`
     - OpenAI models, and any OpenAI-compatible server (vLLM, SGLang,
       llama.cpp) through ``params: {base_url: ...}``; see below.
   * - ``claude``
     - :class:`~iohblade.llm.Claude_LLM`
     - Anthropic's Claude models; see below.
   * - ``gemini``
     - :class:`~iohblade.llm.Gemini_LLM`
     - Google's Gemini models.
   * - ``deepseek``
     - :class:`~iohblade.llm.DeepSeek_LLM`
     - DeepSeek models through their OpenAI-compatible API.
   * - ``ollama``
     - :class:`~iohblade.llm.Ollama_LLM`
     - Local models served by `Ollama <https://ollama.com>`_; no key.
   * - ``lmstudio``
     - :class:`~iohblade.llm.LMStudio_LLM`
     - Local models in LM Studio (``uv sync --group apple-silicon``).
   * - ``mlx_lm``
     - :class:`~iohblade.llm.MLX_LM_LLM`
     - Local MLX models on Apple silicon (``uv sync --group apple-silicon``).
   * - ``multi``
     - :class:`~iohblade.llm.Multi_LLM`
     - Picks one of several named LLMs (``llms: [a, b]``) at random for
       each call, e.g. to spread the load over one server per GPU.
   * - ``dummy``
     - :class:`~iohblade.llm.Dummy_LLM`
     - Offline stand-in for tests; returns a fixed algorithm.
   * - ``custom``
     - your class
     - Any :class:`~iohblade.llm.LLM` subclass; see
       :doc:`tutorial_custom_llm`.

``uv run iohblade check experiment.yml --llm`` sends one short prompt to
every LLM, so a wrong key or model name shows up before a run starts. In
Python, every provider is a class with the same interface (see
:doc:`llm_api`).

Claude (Anthropic)
------------------

:class:`iohblade.llm.Claude_LLM` calls Claude through the official
``anthropic`` Python SDK (version 2.x, API version ``2026-10-21``).

Credentials are read by the SDK: pass ``api_key`` explicitly, or pass
``None`` to use the ``ANTHROPIC_API_KEY`` environment variable (or a profile
created with ``ant auth login``).

.. code-block:: python

   import os

   from iohblade.llm import Claude_LLM
   from iohblade.methods import LLaMEA

   llm = Claude_LLM(
       os.getenv("ANTHROPIC_API_KEY"),
       "claude-opus-5-5",
       effort="high",
       fallbacks="default",
   )
   method = LLaMEA(llm, budget=100, n_parents=4, n_offspring=12, elitism=False)

Arguments:

- ``model``: a Claude model ID, default ``"claude-opus-5-5"``.
- ``max_tokens``: output cap per response, default ``16000``. Thinking counts
  towards it, so keep it generous when generating whole algorithms.
- ``effort``: ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` or ``"max"``.
  It controls how much the model thinks and so how many tokens it spends.
  ``None`` uses the model's default (``"medium"`` on Claude Opus 5.5).
- ``fallbacks``: server-side refusal fallbacks. With ``"default"``, a request
  that Claude's safety filters decline is re-run by the API on Anthropic's
  recommended fallback model, so the candidate is not lost.
- ``base_url``: an alternative API endpoint. The key is sent as
  ``Authorization: Bearer``.
- ``temperature``: deprecated and ignored. API version ``2026-10-21`` no
  longer accepts sampling parameters; passing a value raises a
  ``DeprecationWarning``.

Behaviour worth knowing:

- Only the text of the response is returned; thinking is not included.
- A refused request (``stop_reason == "refusal"``) returns an empty string,
  so the search method treats it as a response without code.
- Rate limits, timeouts, connection and server errors are retried with
  backoff. Errors that a retry cannot fix (400, 401, 403, 404) are raised
  immediately.
- ``get_config()`` (written to the experiment logs) never contains the API
  key.
- The tokens BLADE logs per call are the ones the API reports, thinking
  tokens and prompt-cache reads and writes included (see
  :ref:`llm-token-usage`).

Local models with vLLM (OpenAI-compatible servers)
--------------------------------------------------

:class:`iohblade.llm.OpenAI_LLM` works with any server that implements the
OpenAI chat completions API, such as `vLLM <https://docs.vllm.ai>`_, SGLang
or llama.cpp. Set ``base_url`` to the server and ``model`` to the name the
server uses. Without an ``api_key``, ``"EMPTY"`` is sent, because local
servers usually accept any key.

vLLM batches concurrent requests, so with ``experiment.n_jobs`` above 1 one
server answers many runs at the same time. For a model on several GPUs,
start one server per GPU and combine them with ``provider: multi``, which
spreads the calls over the servers:

.. code-block:: bash

   # One server per GPU (ports 8000-8003).
   for i in 0 1 2 3; do
     CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-32B --port 800$i \
       --max-model-len 32768 &
   done

.. code-block:: yaml

   experiment:
     n_jobs: 32            # about 8 concurrent requests per server
   llms:
     gpu0: {provider: openai, model: Qwen/Qwen3-32B,
            params: {base_url: "http://localhost:8000/v1"}}
     gpu1: {provider: openai, model: Qwen/Qwen3-32B,
            params: {base_url: "http://localhost:8001/v1"}}
     gpu2: {provider: openai, model: Qwen/Qwen3-32B,
            params: {base_url: "http://localhost:8002/v1"}}
     gpu3: {provider: openai, model: Qwen/Qwen3-32B,
            params: {base_url: "http://localhost:8003/v1"}}
     qwen:
       provider: multi
       llms: [gpu0, gpu1, gpu2, gpu3]

The same in Python:

.. code-block:: python

   from iohblade.llm import Multi_LLM, OpenAI_LLM

   llm = Multi_LLM(
       [
           OpenAI_LLM(model="Qwen/Qwen3-32B", base_url=f"http://localhost:800{i}/v1")
           for i in range(4)
       ]
   )

Other arguments:

- ``extra_body``: extra fields sent with every request, for server-specific
  options. For example, ``{"chat_template_kwargs": {"enable_thinking":
  false}}`` turns off the thinking mode of Qwen3 models, which saves many
  tokens when generating code; ``{"max_tokens": 8192}`` caps the length of
  a reply.
- ``timeout``: seconds before a request times out. Raise it for long
  replies from a busy server.

When a method's name contains ``{llm}``, it shows the name of the ``multi``
LLM (``qwen`` above). With ``provider: multi``, ShinkaEvolve's bandit
chooses between the servers; because all four serve the same model, this
only balances the load.

.. _llm-token-usage:

Token usage and cost
--------------------

For every call, BLADE writes two entries to ``conversationlog.jsonl``: the
prompt (``role: "client"``) and the reply (``role``: the model), each with a
``tokens`` and a ``cost`` field (USD). :func:`iohblade.plots.plot_token_usage`
adds up these tokens.

**Tokens** are the counts the provider's API returns with the response, so
no extra request is made:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Provider
     - Prompt entry
     - Reply entry
   * - Claude
     - ``input_tokens`` + cache read + cache write tokens
     - ``output_tokens`` (thinking included)
   * - OpenAI, DeepSeek, OpenAI-compatible servers
     - ``prompt_tokens`` (cached tokens included)
     - ``completion_tokens`` (reasoning included)
   * - Gemini
     - ``prompt_token_count``
     - ``candidates_token_count`` + ``thoughts_token_count``
   * - Ollama
     - ``prompt_eval_count``
     - ``eval_count``

The usage of the last call is also available as ``llm.last_usage`` (a
:class:`iohblade.llm_costs.TokenUsage`). Providers that report no usage
(LM Studio, MLX, ``Dummy_LLM`` and your own ``LLM`` subclasses, unless their
``_query`` sets ``self.last_usage``) are logged with a rough estimate of
four characters per token. A call that fails is logged with 0 tokens.

**Cost** is computed by BLADE from these tokens and the list prices in
:data:`iohblade.llm_costs.MODEL_PRICES` (USD per million tokens for input,
output and cache reads, checked on the date in
:data:`iohblade.llm_costs.PRICES_CHECKED`). The model the API says answered
is priced, so a dated snapshot such as ``gpt-4o-2024-08-06`` or a Claude
refusal fallback model gets its own price. The table covers the current
first-party models of Anthropic, OpenAI, Google (Gemini) and DeepSeek.

- Local models (Ollama, vLLM, LM Studio, MLX) and models that are not in the
  table have cost 0; their tokens are still logged.
- Prices are standard-tier list prices for the shortest context tier.
  Long-context, batch, priority, fast-mode and regional prices, and discounts,
  are not modelled. DeepSeek is priced at its peak rate (off-peak requests
  cost half).
- Prices change: for billing, use the provider's console. To price another
  model, add it to the table before the run:

  .. code-block:: python

     from iohblade.llm_costs import MODEL_PRICES, ModelPrice

     MODEL_PRICES["my-model"] = ModelPrice(input=1.0, output=4.0, cache_read=0.1)

API reference
-------------

All providers and the usage and cost helpers are documented in
:doc:`llm_api`.
