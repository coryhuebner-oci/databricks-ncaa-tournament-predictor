from datetime import date, datetime
from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import substring, col, trim, to_date, when, year, month, udf
from pyspark.sql.types import IntegerType


def _get_fixed_width_column(start_position: int, length: str) -> Column:
    return trim(substring(col("value"), start_position, length))


def get_fixed_width_text_column(name: str, start_position: int, length: str) -> Column:
    return _get_fixed_width_column(start_position, length).alias(name)


def get_fixed_width_int_column(name: str, start_position: int, length: str) -> Column:
    return (
        _get_fixed_width_column(start_position, length).cast(IntegerType()).alias(name)
    )


def get_fixed_width_date_column(name: str, start_position: int, length: str) -> Column:
    return to_date(_get_fixed_width_column(start_position, length), "MM/dd/yyyy").alias(
        name
    )


def team_one_won() -> Column:
    return when(col("team_1_score") > "team_2_score", True)


@udf(returnType=IntegerType())
def get_college_season(value: date):
    """Get the college season for a given file; this handles games played late in the previous year being
    tied to the next year's college season. E.g. a game played on 11/23/2023 is actually part of the 2024 season.
    A game played on 03/10/2024 is also part of the 2024 season. A game played on 10/1/2024 is part of the 2025 season
    """
    if isinstance(value, datetime):  # If it's a timestamp, extract date
        value = value.date()
    if isinstance(value, date):  # Ensure it's a date
        return value.year + 1 if value.month > 5 else value.year
    return None  # Handle unexpected types


def get_cleaned_head_to_head_data(raw_head_to_head_data: DataFrame) -> DataFrame:
    """Convert raw Kaggle stat text data into a cleaned up DataFrame for easier downstream consumption"""
    # Split columns from fixed-width text files
    split_into_columns = raw_head_to_head_data.select(
        # Game date (positions 1-10)
        get_fixed_width_date_column("game_date", 1, 10),
        # Extract team 1 (positions 12-31)
        get_fixed_width_text_column("team_1", 12, 23),
        # Extract team 1 score (positions 33-35)
        get_fixed_width_int_column("team_1_score", 35, 3),
        # Extract team 2 (positions 39-58)
        get_fixed_width_text_column("team_2", 39, 23),
        # Extract team 2 score (positions 60-62)
        get_fixed_width_int_column("team_2_score", 62, 3),
    )
    with_winner_fields = split_into_columns.withColumn(
        "team_1_won", col("team_1_score") > col("team_2_score")
    ).withColumn(
        "winning_team", when(col("team_1_won"), col("team_1")).otherwise(col("team_2"))
    )
    with_year_field = with_winner_fields.withColumn(
        "college_season", get_college_season(col("game_date"))
    )
    return with_year_field
