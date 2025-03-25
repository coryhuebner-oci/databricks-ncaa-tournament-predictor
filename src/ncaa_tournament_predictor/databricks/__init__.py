from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

from ncaa_tournament_predictor.config import Config


def get_databricks_spark_session() -> DatabricksSession:
    """Get a Databricks Spark session using the configured profile"""
    profile_name = Config.databricks_profile()
    return DatabricksSession.builder.profile(profile_name).getOrCreate()


# A type-hint for local environments for the globally available spark variable
# that exists in Databricks environments
spark: SparkSession
