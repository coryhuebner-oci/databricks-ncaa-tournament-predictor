# NCAA Tournament Predictor
An attempt at using Databricks to build a model that can predict the outcome of the NCAA men's basketball tournament
to see if performs better than human-Cory did for the 2025 tournament.

## Prerequisites
Before beginning, you'll need the following on your machine
- Python (3.11.1 or greater)
- The [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)

## Local environment setup
To setup your local environment, run the following:
1. Initialize a virtual environment: `python -m venv .venv --copies`, The `--copies` value is used to ensure the Python binary is accessible
even when run as a container.
2. Activate that environment: `. .venv/bin/activate`
3. Install this project and its dependencies: `pip install -e .`
4. Install development dependencies: `pip install '.[dev]'`