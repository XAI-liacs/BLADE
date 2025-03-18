<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img alt="Shows the BLADE logo." src="logo.png" width="200px">
  </picture>
</p>

<h1 align="center">IOH-BLADE: Benchmarking LLM-driven Automated Design and Evolution of Iterative Optimization Heuristics</h1>

<p align="center">
  <a href="https://pypi.org/project/blade/">
    <img src="https://badge.fury.io/py/blade.svg" alt="PyPI version" height="18">
  </a>
  <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg" alt="Maintenance" height="18">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python 3.10+" height="18">
</p>

## Table of Contents
- [Introduction](#introduction)
- [News](#-news)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)


## Introduction


## 🔥 News 



## 🎁 Installation

It is the easiest to use BLADE from the pypi package (`iohblade`).

```bash
  pip install iohblade
```
> [!Important]
> The Python version **must** be larger or equal to Python 3.10.
> You need an OpenAI/Gemini/Ollama API key for using LLM models.

You can also install the package from source using Poetry (1.8.5).

1. Clone the repository:
   ```bash
   git clone https://github.com/XAI-liacs/BLADE.git
   cd BLADE
   ```
2. Install the required dependencies via Poetry:
   ```bash
   poetry install
   ```

## 💻 Quick Start

1. Set up an OpenAI API key:
   - Obtain an API key from [OpenAI](https://openai.com/) or Gemini or another LLM provider.
   - Set the API key in your environment variables:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

2. Running an Experiment

    To run a benchmarking experiment using BLADE:

    ```python
    from iohblade import Experiment

    from iohblade.experiment import Experiment
    from iohblade.llm import Ollama_LLM
    from iohblade.methods import LLaMEA, RandomSearch
    from iohblade.problems import BBOB_SBOX
    import os

    llm = Ollama_LLM("qwen2.5-coder:14b") #qwen2.5-coder:14b, deepseek-coder-v2:16b
    budget = 50 #short budget for testing

    RS = RandomSearch(llm, budget=budget) #Random Search baseline
    LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", n_parents=4, n_offspring=12, elitism=False) #LLamEA with 4,12 strategy

    methods = [RS, LLaMEA_method]

    # List containing function IDs per group
    group_functions = [
        [], #starting at 1
        [1, 2, 3, 4, 5],      # Separable Functions
        [6, 7, 8, 9],         # Functions with low or moderate conditioning
        [10, 11, 12, 13, 14], # Functions with high conditioning and unimodal
        [15, 16, 17, 18, 19], # Multi-modal functions with adequate global structure
        [20, 21, 22, 23, 24]  # Multi-modal functions with weak global structure
    ]
    
    problems = []
    # include all SBOX_COST functions with 5 instances for training and 10 for final validation as the benchmark problem.
    training_instances = [(f, i) for f in range(1,25) for i in range(1, 6)]
    test_instances = [(f, i) for f in range(1,25) for i in range(5, 16)]
    problems.append(BBOB_SBOX(training_instances=training_instances, test_instances=test_instances, dims=[5], budget_factor=2000, name=f"SBOX_COST"))
    # Set up the experiment object with 5 independent runs per method/problem. (in this case 1 problem)
    experiment = Experiment(methods=methods, problems=problems, llm=llm, runs=5, show_stdout=True, log_dir="results/SBOX") #normal run
    experiment() #run the experiment, all data is logged in the folder results/SBOX/
    ```

---

## 💻 Examples

---

## 🤖 Contributing

Contributions to BLADE are welcome! Here are a few ways you can help:

- **Report Bugs**: Use [GitHub Issues](https://github.com/nikivanstein/BLADE/issues) to report bugs.
- **Feature Requests**: Suggest new features or improvements.
- **Pull Requests**: Submit PRs for bug fixes or feature additions.

Please refer to CONTRIBUTING.md for more details on contributing guidelines.

## 🪪 License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See `LICENSE` for more information.


## ✨ Citation


