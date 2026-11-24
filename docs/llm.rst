LLM
==========

Every LLM provider in BLADE subclasses :class:`iohblade.llm.LLM` and is passed
to a search method (for example ``LLaMEA(llm, budget=...)``) in the same way.
The sections below cover provider-specific setup; the full API reference
follows.

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
- The token counts and costs BLADE logs per call are estimates from
  ``tokencost``. They do not include thinking tokens and are 0 for models
  ``tokencost`` does not know; check the Claude Console for actual usage.

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

API reference
-------------

.. automodule:: iohblade.llm
   :members:
   :undoc-members:
   :show-inheritance:
