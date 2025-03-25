# NCAA Tournament Predictor
An attempt at using Databricks to build a model that can predict the outcome of the NCAA men's basketball tournament
to see if performs better than human-Cory did for the 2025 tournament.

## Prerequisites
Before beginning, you'll need the following on your machine
- Python (3.11.1 or greater)
- The [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
- Optional - [The Databricks Connect VS Code extension](https://docs.databricks.com/aws/en/dev-tools/vscode-ext/install). This helps with establishing connectivity, profile setup, etc. with Databricks if you are using VS Code as your IDE.

## Local environment setup
To setup your local environment, run the following:
1. Initialize a virtual environment: `python -m venv .venv --copies`, The `--copies` value is used to ensure the Python binary is accessible
even when run as a container.
2. Activate that environment: `. .venv/bin/activate`
3. Install this project and its dependencies: `pip install -e .`
4. Install development dependencies: `pip install '.[dev]'`
5. Ensure you've [setup a Databricks profile for the Databricks connection](https://docs.databricks.com/aws/en/dev-tools/cli/profiles).
The app leverages Databricks profile configuration to connect to Databricks
5. Lastly, setup a .env file or environment variables for configuration values; e.g. you'll need a `DATABRICKS_PROFILE` value holding the name of the Databricks profile you want to use. You can use the [.env.template](./.env.template) file as a reference of configuration
values required.

## Datasets
The datasets used to build this predictor are as-follows:

1. [Kaggle NCAA Stats](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset) - Team stats for each season from 2013-2025. 2013-2024 stats are used for model training. 2025 (current season) stats are used as inputs for inference
2. []