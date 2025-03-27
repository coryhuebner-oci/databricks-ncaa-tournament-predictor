from pyspark.sql import DataFrame, SparkSession
from databricks.feature_engineering import FeatureEngineeringClient

from ncaa_tournament_predictor import volumes, tables, transformation, databricks


def write_cleaned_kaggle_stats(spark: SparkSession) -> DataFrame:
    """Convert raw Kaggle stat CSV data into a cleaned up table for easier downstream consumption"""
    raw_kaggle_stats = (
        spark.read.format("csv")
        .options(header=True, inferSchema=True, mergeSchema=True)
        .load(volumes.raw_kaggle_stats)
    )
    cleaned_stats = transformation.get_cleaned_kaggle_stats(raw_kaggle_stats)
    cleaned_stats.write.format("delta").option("overwriteSchema", "true").saveAsTable(
        tables.cleaned_kaggle_stats, mode="overwrite"
    )


def convert_cleaned_kaggle_stats_to_feature_store(spark: SparkSession):
    fe_client = FeatureEngineeringClient()
    fe_client.create_table(
        name=tables.game_prediction_feature_set,
        df=spark.read.table(tables.cleaned_kaggle_stats),
        primary_keys=["team", "college_season"],
    )


def run_job():
    spark = databricks.get_databricks_spark_session()
    write_cleaned_kaggle_stats(spark)
    convert_cleaned_kaggle_stats_to_feature_store(spark)
