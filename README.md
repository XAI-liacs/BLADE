<h1 align="center">LLaMEA-SAGE: Guiding Automated Algorithm Design with
Structural Feedback from Explainable AI</h1>


## Introduction
This is the reproducability repository for the paper "LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI".



## 🎁 Installation

> [!Important]
> The Python version **must** be larger or equal to Python 3.11.
> You need an OpenAI/Gemini/Ollama/Claude/DeepSeek API key for using LLM models.

You can install the code from source using <a href="https://docs.astral.sh/uv/" target="_blank">uv</a> (0.7.19).
make sure you have `uv` installed.

1. Clone the repository:
   ```bash
   git clone <this_repo>
   cd BLADE
   ```

2. Install the required dependencies via uv:
   ```bash
   uv sync
   ```


## 💻 Quick Start

1. Set up an API key for your preferred provider:
   - Obtain an API key from [OpenAI](https://openai.com/), Claude, Gemini, or another LLM provider.
   - Set the API key in your environment variables:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

2. Running an Experiment

    You can (re)run the experiments of the paper as follows:  

    ```bash
    uv run python run_archive_guided.py
    ```
    This will run Experiment 1 (LLaMEA and LLaMEA-SAGE on SBOX-COST).

    ```bash
    uv run python run_archive_guided-2.py
    uv run python run_archive_guided-2-1.py
    uv run python run_archive_guided-2-2.py
    ```
    This will run Experiment 2 (LLaMEA and LLaMEA-SAGE on MA-BBOB), baselines and the abblation using Gemini respectively.

3. Produce paper artifacts.

    Use the Python Notebooks `visualize_1.ipynb`, `visualize_2.ipynb`, `visualize_3.ipynb` and `final_validation.ipynb` to create the plots from the paper.  
    These notebooks use the raw results from the folder `paper_results`.

