Installation
------------

It is easiest to use BLADE from the PyPI package:

.. code-block:: bash

   pip install iohblade

.. important::
   The Python version **must** be >= 3.11.
   An OpenAI/Gemini/Ollama/Claude/DeepSeek API key is needed for using LLM models.

You can also install the package from source using **uv** (0.7.9).

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/XAI-liacs/BLADE.git
      cd BLADE

2. Install the required dependencies via *uv*:

   .. code-block:: bash

      uv sync

3. (Optional) Install extra dependencies for developer tools and documentation:

   .. code-block:: bash

      uv sync --group dev --group docs

4. (Optional) Install support for running MLX optimised LLMs locally on apple silicon machine:

   .. code-block:: bash

      uv sync --group dev --group docs --group apple_silicon --prerelease=allow
