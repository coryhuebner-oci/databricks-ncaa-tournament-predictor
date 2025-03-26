from datetime import date

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
    assert "team_1_won" in result.columns
    assert "winning_team" in result.columns
    assert "college_season" in result.columns

    # Spot-check a few of the rows with interesting data
    rows = result.collect()
    first_row = rows[0]
    assert first_row["game_date"] == date(2012, 11, 9)
    assert first_row["team_1"] == "Utah Valley"
    assert first_row["team_1_score"] == 54
    assert first_row["team_2"] == "IUPUI"
    assert first_row["team_2_score"] == 67
    assert first_row["team_1_won"] == False
    assert first_row["winning_team"] == "IUPUI"
    assert first_row["college_season"] == 2013

    second_row = rows[1]
    assert second_row["game_date"] == date(2024, 3, 26)
    assert second_row["team_1"] == "Fairfield"
    assert second_row["team_1_score"] == 58
    assert second_row["team_2"] == "Seattle"
    assert second_row["team_2_score"] == 75
    assert second_row["team_1_won"] == False
    assert second_row["winning_team"] == "Seattle"
    assert second_row["college_season"] == 2024

    last_row = rows[-1]
    assert last_row["game_date"] == date(2023, 12, 19)
    assert last_row["team_1"] == "Cal St.Dominguez Hills"
    assert last_row["team_1_score"] == 107
    assert last_row["team_2"] == "Long Beach St."
    assert last_row["team_2_score"] == 98
    assert last_row["team_1_won"] == True
    assert last_row["winning_team"] == "Cal St.Dominguez Hills"
    assert last_row["college_season"] == 2024
