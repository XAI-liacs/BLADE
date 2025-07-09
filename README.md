<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img alt="Shows the BLADE logo." src="logo.png" width="200px">
  </picture>
</p>

<h1 align="center">IOH-BLADE: Benchmarking LLM-driven Automated Design and Evolution of Iterative Optimization Heuristics</h1>

<p align="center">
  <a href="https://pypi.org/project/iohblade/">
    <img src="https://badge.fury.io/py/iohblade.svg" alt="PyPI version" height="18">
  </a>
  <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg" alt="Maintenance" height="18">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python 3.11+" height="18">
  <a href="https://codecov.io/gh/XAI-liacs/BLADE" > 
    <img src="https://codecov.io/gh/XAI-liacs/BLADE/graph/badge.svg?token=ZOT67R1TP7" alt="CodeCov" height="18"/> 
  </a>
</p>

> [!TIP]
> See also the [Documentation](https://xai-liacs.github.io/BLADE/).

## Table of Contents
- [News](#-news)
- [Introduction](#introduction)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Webapp](#-webapp)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

## 🔥 News 

- 2025.03 ✨✨ **BLADE v0.0.1 released**!


## Introduction
**BLADE** (Benchmark suite for LLM-driven Automated Design and Evolution) provides a standardized benchmark suite for evaluating automatic algorithm design algorithms, particularly those generating metaheuristics by large language models (LLMs). It focuses on **continuous black-box optimization** and integrates a diverse set of **problems** and **methods**, facilitating fair and comprehensive benchmarking.


### Features

- **Comprehensive Benchmark Suite:** Covers various classes of black-box optimization problems.
- **LLM-Driven Evaluation:** Supports algorithm evolution and design using large language models.
- **Built-In Baselines:** Includes state-of-the-art metaheuristics for comparison.
- **Automatic Logging & Visualization:** Integrated with **IOHprofiler** for performance tracking.

#### Included Benchmark Function Sets

BLADE incorporates several benchmark function sets to provide a comprehensive evaluation environment:

| Name                               | Short Description                                                                                                                         | Number of Functions | Multiple Instances |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|---------------------|--------------------|
| **BBOB** (Black-Box Optimization Benchmarking) | A suite of 24 noiseless functions designed for benchmarking continuous optimization algorithms. [Reference](https://arxiv.org/pdf/1903.06396) | 24                  | Yes                |
| **SBOX-COST**                      | A set of 24 boundary-constrained functions focusing on strict box-constraint optimization scenarios. [Reference](https://inria.hal.science/hal-04403658/file/sboxcost-cmacomparison-authorversion.pdf) | 24                  | Yes                |
| **MA-BBOB** (Many-Affine BBOB)     | An extension of the BBOB suite, generating functions through affine combinations and shifts. [Reference](https://dl.acm.org/doi/10.1145/3673908) | Generator-Based     | Yes                |
| **GECCO MA-BBOB Competition Instances** | A collection of 1,000 pre-defined instances from the GECCO MA-BBOB competition, evaluating algorithm performance on diverse affine-combined functions. [Reference](https://iohprofiler.github.io/competitions) | 1,000               | Yes                |

In addition, several real-world applications are included such as several photonics problems.

### Included Search Methods

The suite contains the state-of-the-art LLM-assisted search algorithms:

| Algorithm           | Description                                        | Link
|--------------------------|-------------------------------------------------|--------------|
| **LLaMEA** | Large Langugage Model Evolutionary Algorithm                 | [code](https://github.com/nikivanstein/LLaMEA) [paper](https://arxiv.org/abs/2405.20132) |
| **EoH** | Evolution of Heuristics         | [code](https://github.com/FeiLiu36/EoH) [paper](https://arxiv.org/abs/2401.02051) |
| **FunSearch**   | Google's GA-like algorithm | [code](https://github.com/google-deepmind/funsearch) [paper](https://www.nature.com/articles/s41586-023-06924-6) |
| **ReEvo**    | Large Language Models as Hyper-Heuristics with Reflective Evolution | [code](https://github.com/ai4co/LLM-as-HH) [paper](https://arxiv.org/abs/2402.01145) |

> Note, FunSearch is currently not yet integrated.

### Supported LLM APIs

BLADE supports integration with various LLM APIs to facilitate automated design of algorithms:

| LLM Provider | Description                                                                                                         | Integration Notes                                                                                             |
|--------------|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Gemini**   | Google's multimodal LLM designed to process text, images, audio, and more. [Reference](https://en.wikipedia.org/wiki/Gemini_%28language_model%29) | Accessible via the Gemini API, compatible with OpenAI libraries. [Reference](https://ai.google.dev/gemini-api/docs/openai) |
| **OpenAI**   | Developer of GPT series models, including GPT-4, widely used for natural language understanding and generation. [Reference](https://openai.com/) | Integration through OpenAI's REST API and client libraries.                                                    |
| **Ollama**   | A platform offering access to various LLMs, enabling local and cloud-based model deployment. [Reference](https://www.ollama.ai/) | Integration details can be found in their official documentation.                                             |


### Evaluating against Human Designed baselines

An important part of BLADE is the final evaluation of generated algorithms against state-of-the-art human designed algorithms.
In the `iohblade.baselines` part of the package, several well known SOTA black-box optimizers are imolemented to compare against.
Including but not limited to CMA-ES and DE variants.

For the final validation **BLADE** uses [**IOHprofiler**](https://iohprofiler.github.io/), providing detailed tracking and visualization of performance metrics.


## 🎁 Installation

It is the easiest to use BLADE from the pypi package (`iohblade`).

```bash
  pip install iohblade
```
> [!Important]
> The Python version **must** be larger or equal to Python 3.11.
> You need an OpenAI/Gemini/Ollama API key for using LLM models.

You can also install the package from source using <a href="https://docs.astral.sh/uv/" target="_blank">uv</a> (0.7.19).
make sure you have `uv` installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/XAI-liacs/BLADE.git
   cd BLADE
   ```

2. Install the required dependencies via uv:
   ```bash
   uv sync
   ```

3. *(Optional)* Install additional packages:
   ```bash
   uv sync --group kerneltuner --group dev --group docs
   ```
   This will install additional dependencies for development and building documentation.
   The (experimental) auto-kernel application is also under a separate group for now. 

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
    from iohblade import Ollama_LLM
    from iohblade.methods import LLaMEA, RandomSearch
    from iohblade.problems import BBOB_SBOX
    from iohblade.loggers import ExperimentLogger
    import os

    llm = Ollama_LLM("qwen2.5-coder:14b") #qwen2.5-coder:14b, deepseek-coder-v2:16b
    budget = 50 #short budget for testing

    RS = RandomSearch(llm, budget=budget) #Random Search baseline
    LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", n_parents=4, n_offspring=12, elitism=False) #LLamEA with 4,12 strategy
    methods = [RS, LLaMEA_method]

    problems = []
    # include all SBOX_COST functions with 5 instances for training and 10 for final validation as the benchmark problem.
    training_instances = [(f, i) for f in range(1,25) for i in range(1, 6)]
    test_instances = [(f, i) for f in range(1,25) for i in range(5, 16)]
    problems.append(BBOB_SBOX(training_instances=training_instances, test_instances=test_instances, dims=[5], budget_factor=2000, name=f"SBOX_COST"))
    # Set up the experiment object with 5 independent runs per method/problem. (in this case 1 problem)
    logger = ExperimentLogger("results/SBOX")
    experiment = Experiment(methods=methods, problems=problems, runs=5, show_stdout=True, exp_logger=logger) #normal run
    experiment() #run the experiment, all data is logged in the folder results/SBOX/
    ```

## 🌐 Webapp

After running experiments you can browse them using the built-in Streamlit app:

```bash
uv run iohblade-webapp
```

The app lists available experiments from the `results` directory, displays their progress, and shows convergence plots.

---

## 💻 Examples


See the files in the `examples` folder for examples on experiments and visualisations.

---

## 🤖 Contributing

Contributions to BLADE are welcome! Here are a few ways you can help:

- **Report Bugs**: Use [GitHub Issues](https://github.com/XAI-Liacs/BLADE/issues) to report bugs.
- **Feature Requests**: Suggest new features or improvements.
- **Pull Requests**: Submit PRs for bug fixes or feature additions.

Please refer to CONTRIBUTING.md for more details on contributing guidelines.

## 🪪 License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See `LICENSE` for more information.


## ✨ Citation


TBA

-----
Happy Benchmarking with IOH-BLADE! 🚀
