from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession


def get_databricks_spark_session(profile_name: str | None = None) -> SparkSession:
    """Get a Databricks Spark session using the provided profile or the default profile if no profile given"""
    return (
        DatabricksSession.builder.profile(profile_name).getOrCreate()
        if profile_name
        else DatabricksSession.builder.getOrCreate()
    )
