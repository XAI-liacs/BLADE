# AGENTS

Guidance for AI contributors, including Codex, on working with this repository.

## Setup

- Requires Python 3.11+.
- Install [uv](https://docs.astral.sh/uv/) (v0.7.19 or newer).
- Install dependencies:
  ```bash
  uv sync
  ```
- For development and documentation tools:
  ```bash
  uv sync --group dev --group docs
  ```

## Development tools

- Use `uv run` to execute commands in the environment.
- Format code before committing:
  ```bash
  uv run isort iohblade/
  uv run black iohblade/
  ```
- Search the codebase with `rg`; avoid `grep -R` and `ls -R`.
- Style guidelines:
  - Use 4 spaces for indentation.
  - Keep lines under roughly 80 characters.
  - Class names use `CamelCase` and functions use `snake_case`.

## Testing

- Run tests for any code change:
  ```bash
  uv run pytest tests/
  ```
- Documentation or comment only changes do not require tests or linters.

