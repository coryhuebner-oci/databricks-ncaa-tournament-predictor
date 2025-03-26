#!/bin/sh

# Create the local testing environment
local_env_name='local-environment'
pdm venv create --name "$local_env_name"
eval $(pdm venv activate "$local_env_name")
pdm install -G "$local_env_name" --lockfile "$local_env_name.pdm.lock"
deactivate

# Create the remote cluster access environment
remote_env_name='remote-databricks-cluster'
pdm venv create --name "$remote_env_name"
eval $(pdm venv activate "$remote_env_name")
pdm install -G "$remote_env_name" --lockfile "$remote_env_name.pdm.lock"
deactivate