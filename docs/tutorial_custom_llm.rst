Tutorial: your own LLM
======================

The built-in providers cover hosted APIs, Ollama, LM Studio and any
OpenAI-compatible server (see :doc:`llm`). For anything else, such as an
in-house endpoint or a model loaded in Python, write a subclass of
:class:`iohblade.llm.LLM` and use it with ``provider: custom``.

1. Write the class
------------------

An LLM needs two methods: ``_query`` sends a conversation and returns the
answer as text, and ``get_config`` describes the model for the records of
each run. Everything else (extracting code from the answer, retries in the
methods, logging the conversation) is done by the base
class.

.. code-block:: python
   :caption: my_llm.py

   import requests

   from iohblade.llm import LLM


   class MyLLM(LLM):
       """An LLM behind a simple HTTP endpoint that takes and returns text."""

       def __init__(self, model, url="http://localhost:9000/generate", api_key=None, **kwargs):
           super().__init__(api_key, model, **kwargs)
           self.url = url

       def _query(self, session, **kwargs):
           # session: [{"role": "user" | "assistant" | "system", "content": "..."}, ...]
           prompt = "\n\n".join(m["content"] for m in session)
           response = requests.post(
               self.url, json={"model": self.model, "prompt": prompt}, timeout=600
           )
           response.raise_for_status()
           return response.json()["text"]

       def get_config(self):
           # Recorded with every run (llm.json), for reproducibility.
           return [{"model": self.model, "config": {"url": self.url}, "hardware": {}}]

2. Use it in an experiment
--------------------------

.. code-block:: yaml
   :caption: experiment.yml

   llms:
     mine:
       provider: custom
       source: my_llm.py        # relative to this file
       class: MyLLM
       model: my-model
       params: {url: "http://localhost:9000/generate"}
   methods:
     - {type: llamea, name: "LLaMEA-{llm}", llm: mine}

``model`` and ``api_key`` are passed to the constructor like for built-in
providers, and ``params`` are checked against its signature. The class runs
in BLADE's own environment, so install what it imports there.

3. Check it
-----------

.. code-block:: bash

   uv run iohblade validate experiment.yml     # the class loads and its parameters fit
   uv run iohblade check experiment.yml --llm  # one real prompt to your endpoint

``validate`` reports a missing method (for example ``get_config``) or an
unknown parameter; ``check --llm`` sends one short prompt and shows the
answer. In Python, the same class works directly:
``LLaMEA(MyLLM("my-model"), budget=50)``.
