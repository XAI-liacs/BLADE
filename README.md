<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img alt="Shows the BLADE logo." src="logo.png" width="200px">
  </picture>
</p>

<h1 align="center">BLADE: Benchmarking LLM-driven Automated Design and Evolution</h1>

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

It is the easiest to use BLADE from the pypi package.

```bash
  pip install blade
```
> [!Important]
> The Python version **must** be larger or equal to Python 3.10.
> You need an OpenAI/Gemini/Ollama API key for using LLM models.

You can also install the package from source using Poetry (1.8.5).

1. Clone the repository:
   ```bash
   git clone https://github.com/nikivanstein/BLADE.git
   cd BLADE
   ```
2. Install the required dependencies via Poetry:
   ```bash
   poetry install
   ```

## 💻 Quick Start

1. Set up an OpenAI API key:
   - Obtain an API key from [OpenAI](https://openai.com/).
   - Set the API key in your environment variables:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

2. Running an Experiment

    To run a benchmarking experiment using BLADE:

    ```python
    from blade import Experiment
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


