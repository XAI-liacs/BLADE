Installation
------------

It is easiest to use BLADE from the PyPI package:

.. code-block:: bash

   pip install blade

.. important::
   The Python version **must** be >= 3.10.
   An OpenAI/Gemini/Ollama API key is needed for using LLM models.

You can also install the package from source using Poetry (1.8.5).

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/xai-liacs/BLADE.git
      cd BLADE

2. Install the required dependencies via Poetry:

   .. code-block:: bash

      poetry install