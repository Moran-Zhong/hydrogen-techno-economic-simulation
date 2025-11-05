# Repository Guidelines

## Project Structure & Module Organization
`hopp/` holds the Hybrid Optimization and Performance Platform sources; multi-wind enhancements live in `hopp/simulation/technologies/wind/` and optimizers in `hopp/tools/optimization/`. Tests and fixtures sit under `tests/hopp/`, with `api_responses/` and `inputs/` supplying cached data. Use `examples/` for runnable notebooks and YAML scenarios, `docs/` for the Jupyter Book, and keep generated results in `output/` or `log/` out of version control.

## Build, Test & Development Commands
Create a development install with `python -m pip install -e ".[develop]"` (Conda users may activate the provided `.venv` or follow `conda_build.sh`). Run the full regression suite via `pytest tests/hopp`; focus runs can target modules such as `pytest tests/hopp/test_layout.py -k multi_wind`. Rebuild documentation with `jupyter-book build docs/` and lint packaging through `python -m build` if you touch `conda.recipe/` or release assets.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, descriptive snake_case module and function names, and CamelCase classes (`MultiWindPlant`, `SystemOptimizer`). Add Google-style docstrings and type hints when the surrounding module uses them. Keep line length near 100 characters and favor explicit imports; shared utilities belong in `hopp/tools/` rather than ad hoc helpers.

## Testing Guidelines
New features require pytest coverage alongside examples demonstrating multi-wind behavior. Place unit tests beside related modules under `tests/hopp/` and name files `test_<feature>.py`. Mock external services with fixtures in `api_responses/`; tests that need live API keys must check for `NREL_API_KEY` and skip gracefully. Coverage is configured in `pyproject.toml` for branch data on `hopp/*`, so avoid excluding new modules without discussion.

## Commit & Pull Request Guidelines
Write short, imperative commit subjects and reference issues when applicable (`git commit -m "Add multi-wind layout fixture (GH123)"`). Group unrelated changes into separate commits and include command output (e.g., `pytest` summaries) in PR descriptions. Every PR should link to a GitHub issue or clearly state its motivation, note any required environment variables, and attach screenshots for documentation or GUI updates.

## Environment & Secrets
Resource downloaders expect `NREL_API_KEY` and `NREL_API_EMAIL` in your shell or a local `.env` (never commit this file). Cache-heavy artefacts belong in `output/` or external storage; prefer lightweight CSV or YAML fixtures for repeatable results. Update `README.md` or `docs/` when schema or configuration options change so downstream projects stay synchronized.
