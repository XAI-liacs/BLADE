<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img alt="Shows the BLADE logo." src="logo.png" width="200px">
  </picture>
</p>

<h1 align="center">IOH-BLADE: Benchmarking LLM-driven Automated Design and Evolution of Iterative Optimization Heuristics</h1>

<h3>Auto-generated Optimization Algorithms for Auto-Tuning</h3>


> [!TIP]
> This repository contains the experimental setup script using BLADE and the main results on the auto-tuning problem.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [License](#-license)


## Introduction
**BLADE** (Benchmark suite for LLM-driven Automated Design and Evolution) provides a standardized benchmark suite for evaluating automatic algorithm design algorithms, particularly those generating metaheuristics by large language models (LLMs). It focuses on **continuous black-box optimization** and integrates a diverse set of **problems** and **methods**, facilitating fair and comprehensive benchmarking.



## 🎁 Installation

It is the easiest to use BLADE from the pypi package (`iohblade`).

```bash
  pip install iohblade
```
> [!Important]
> The Python version **must** be larger or equal to Python 3.11.
> You need an OpenAI/Gemini/Ollama API key for using LLM models.

## 💻 Quick Start

1. Set up an OpenAI API key:
   - Obtain an API key from [OpenAI](https://openai.com/) or Gemini or another LLM provider.
   - Set the API key in your environment variables:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

2. Run the Kernel-tuner experiment

    `python run-kerneltuner.py`

3. Results can be viewed in the `kerneltuner-results` directory.

    
## 🪪 License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See `LICENSE` for more information.

