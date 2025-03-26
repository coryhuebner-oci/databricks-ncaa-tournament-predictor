from pyspark.sql import DataFrame

from ncaa_tournament_predictor import volumes, tables, transformation, databricks


def write_cleaned_head_to_head_data() -> DataFrame:
    """Convert raw Kaggle stat text data into a cleaned up table for easier downstream consumption"""
    spark = databricks.get_databricks_spark_session()
    raw_head_to_head_data = spark.read.text(volumes.raw_head_to_head)
    cleaned_head_to_head_results = transformation.get_cleaned_head_to_head_data(
        raw_head_to_head_data
    )
    cleaned_head_to_head_results.write.format("delta").option(
        "overwriteSchema", "true"
    ).saveAsTable(
        tables.cleaned_head_to_head_results,
        mode="overwrite",
    )


def run_job():
    write_cleaned_head_to_head_data()
