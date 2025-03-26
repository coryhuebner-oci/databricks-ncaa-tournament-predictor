from pyspark.sql import DataFrame

from ncaa_tournament_predictor import tables, databricks


def get_training_dataset() -> DataFrame:
    """Combine and transform data sources to create a training dataset for an ML model"""
    spark = databricks.get_databricks_spark_session()
    kaggle_stats = spark.read.table(tables.cleaned_kaggle_stats)
    head_to_head_results = spark.read.table(tables.cleaned_head_to_head_results)
    return kaggle_stats.join(head_to_head_results, on="college_season")
