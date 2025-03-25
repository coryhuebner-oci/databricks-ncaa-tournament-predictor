from pyspark.sql import DataFrame

from ncaa_tournament_predictor.databricks import spark
from ncaa_tournament_predictor import volumes, tables, transformation


def write_cleaned_kaggle_stats() -> DataFrame:
    """Convert raw Kaggle stat CSV data into a cleaned up table for easier downstream consumption"""
    raw_kaggle_stats = (
        spark.read.format("csv")
        .options(header=True, inferSchema=True, mergeSchema=True)
        .load(volumes.raw_kaggle_stats)
    )
    cleaned_stats = transformation.get_cleaned_kaggle_stats(raw_kaggle_stats)
    cleaned_stats.write.format("delta").saveAsTable(
        tables.cleaned_kaggle_stats, mode="overwrite"
    )


def run_job():
    write_cleaned_kaggle_stats()
