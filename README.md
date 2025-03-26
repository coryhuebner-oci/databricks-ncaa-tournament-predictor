# NCAA Tournament Predictor
An attempt at using Databricks to build a model that can predict the outcome of the NCAA men's basketball tournament
to see if performs better than human-Cory did for the 2025 tournament.

## Prerequisites
Before beginning, you'll need the following on your machine
- Python (3.11.1 or greater)
- Java - Used if you run Spark-based tests locally. I'm using Temurin 21.0.6, but any supported Java version for
Spark is fine
- [PDM](https://pdm-project.org/en/latest/#installation) - This project used PDM for project/dependency management.
This is especially important for easily switching dependency groups; E.g. Mutually exclusive dependencies like Databricks Connect for remote runs and PySpark for local testing
- The [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
- Optional - [The Databricks Connect VS Code extension](https://docs.databricks.com/aws/en/dev-tools/vscode-ext/install). This helps with establishing connectivity, profile setup, etc. with Databricks if you are using VS Code as your IDE.

## Local environment setup
To setup your local environment:
1. Run the `initialize.sh` script at the root of this project: `./initialize.sh`
1. [Activate the desired virtual environment](https://pdm-project.org/en/latest/usage/venv/#activate-a-virtualenv):
    - `eval $(pdm venv activate local-environment)`: Use this to run in the local environment. This is used primarily for automated tests
    - `eval $(pdm venv activate remote-databricks-cluster)`: Use this to run against a real Databricks cluster
1. Ensure you've [setup a Databricks profile for the Databricks connection](https://docs.databricks.com/aws/en/dev-tools/cli/profiles) if you want to run this application with a real Databricks cluster.
The app leverages Databricks profile configuration to connect to Databricks
1. Lastly, setup a .env file or environment variables for configuration values; e.g. you'll need a `DATABRICKS_PROFILE` value holding the name of the Databricks profile you want to use. You can use the [.env.template](./.env.template) file as a reference of configuration
values required.

### The intialize.sh script
This script takes care of:
1. Creating a Python virtual environment for both local testing and remote Databricks access
2. Installing necessary dependencies for each environment

Note that the script creates two virtual environments. This is necessary to support conflicting dependencies:
- `PySpark` - Needed to run local automated tests
- `Databricks Connect` - Needed to run against a real Databricks cluster. Databricks Connect currently overwrites
PySpark, so the two can't live side-by-side. Databricks Connect does not appear to support isolated local testing
as it always attempts to connect to a remote Databricks cluster

## Datasets
The datasets used to build this predictor are as-follows:

1. [Kaggle NCAA Stats](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset) - Team stats for each season from 2013-2025. 2013-2024 stats are used for model training. 2025 (current season) stats are used as inputs for inference
2. Kenpom Head to Head stats - Head-to-head game results are used for training/testing to validate the picks
made by the model. These were pulled by manually scraping the free Kenpom text endpoint for each season. E.g. [https://kenpom.com/cbbga13.txt](https://kenpom.com/cbbga13.txt), [https://kenpom.com/cbbga14.txt](https://kenpom.com/cbbga14.txt), etc.

## Switching between VS Code workspaces

The setup is a bit complicated, but to use VS Code, you'll need to switch between workspaces to get the correct
dependencies. Open the workspace relevant for the task at-hand:
- [local-environment.code-workspace](./local-environment.code-workspace) - Used to run tests locally
- [remote-databricks-cluster.code-workspace] - Used to run actual ETL, analysis, etc. using a live Databricks cluster