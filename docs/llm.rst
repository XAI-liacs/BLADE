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

API reference
-------------

.. automodule:: iohblade.llm
   :members:
   :undoc-members:
   :show-inheritance:
