from pyspark.sql import DataFrame
import dlt

from ncaa_tournament_predictor.databricks import spark
from ncaa_tournament_predictor import volumes, transformation


@dlt.table(name="cleaned_kaggle_stats")
def get_cleaned_kaggle_stats() -> DataFrame:
    """Convert raw Kaggle stat CSV data into a cleaned up table for easier downstream consumption"""
    raw_kaggle_stats = (
        spark.read.format("csv")
        .options(header=True, inferSchema=True, mergeSchema=True)
        .load(volumes.raw_kaggle_stats)
    )
    return transformation.get_cleaned_kaggle_stats(raw_kaggle_stats)
