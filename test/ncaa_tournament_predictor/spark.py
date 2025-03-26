import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Fixture to create a SparkSession in local mode"""
    spark = SparkSession.builder.master("local[1]").appName("Tests").getOrCreate()
    yield spark
    spark.stop()
