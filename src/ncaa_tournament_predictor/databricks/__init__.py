from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

import mlflow.tracking._model_registry.utils


def get_databricks_spark_session(
    profile_name: str | None = None, serverless: bool = False
) -> SparkSession:
    """Get a Databricks Spark session using the provided profile or the default profile if no profile given"""
    session_builder = DatabricksSession.builder.serverless(serverless)
    if profile_name:
        session_builder.profile(profile_name)
    if serverless:
        mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
            lambda: "databricks-uc"
        )

    return session_builder.getOrCreate()
