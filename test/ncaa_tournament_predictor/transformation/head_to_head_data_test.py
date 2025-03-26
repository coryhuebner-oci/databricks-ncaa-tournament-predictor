from pyspark.sql import DataFrame

from ..test_data import get_file_path

# VS Code claims this import is unused, but it is definitely used when passed to the test
from ..spark import spark
from ncaa_tournament_predictor.transformation import get_cleaned_head_to_head_data


def test_get_cleaned_head_to_head_data_splits_text_into_typed_columns(spark):
    input_file = get_file_path("head_to_head_data.txt")
    input_df: DataFrame = spark.read.text(input_file)

    result = get_cleaned_head_to_head_data(input_df)

    assert result.count() == 4
    assert "game_date" in result.columns
    assert "team_1" in result.columns
    assert "team_1_score" in result.columns
    assert "team_2" in result.columns
    assert "team_2_score" in result.columns
