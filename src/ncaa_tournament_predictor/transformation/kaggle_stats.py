from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_extract, Column, when, equal_null
from pyspark.sql.types import IntegerType

_not_applicable = "N/A"


def normalize_na(column: Column) -> Column:
    """Normalize the NA and N/A values to all be consistently N/A"""
    return when(column == "NA", _not_applicable).otherwise(column)


def na_as_null(column: Column) -> Column:
    return when(normalize_na(column) == _not_applicable, None).otherwise(column)


def parse_seed(column: Column) -> Column:
    """The tournament seed can either be N/A or an integer
    Convert N/A to null and stringified integers into actual integers
    """
    return na_as_null(column).cast(IntegerType())


def get_cleaned_kaggle_stats(df: DataFrame) -> DataFrame:
    """Cleanup Kaggle stats data. This includes:
    1. Renaming columns to be more desriptive & snake cased
    2. Drop all data for teams that didn't make the tournament
    2. Add lineage columns including the NCAA season for the data
    3. Standardize N/A fields
    4. Drop unneeded columns"""
    cleaned_data = (
        df.withColumnsRenamed(
            {
                "Team": "team",
                "CONF": "conference",
                "G": "games",
                "W": "wins",
                "ADJOE": "adjusted_offensive_efficiency",
                "ADJDE": "adjusted_defensive_efficiency",
                "BARTHAG": "power_rating",
                "EFG_O": "effective_field_goal_percentage",
                "EFG_D": "effective_field_goal_percentage_allowed",
                "TOR": "turnover_rate",
                "TORD": "caused_turnover_rate",
                "ORB": "offensive_rebound_rate",
                "DRB": "defensive_rebound_rate",
                "FTR": "freethrows_attempted_rate",
                "FTRD": "freethrows_allowed_rate",
                "2P_O": "two_point_shooting_percentage",
                "2P_D": "two_point_shooting_percentage_allowed",
                "3P_O": "three_point_shooting_percentage",
                "3P_D": "three_point_shooting_percentage_allowed",
                "ADJ_T": "adjusted_tempo",
                "WAB": "wins_above_bubble",
                "SEED": "tournament_seed",
                "POSTSEASON": "postseason_result",
            }
        )
        # Add source file information (including year)
        .withColumn("source_filename", col("_metadata.file_path"))
        .withColumn(
            "college_season",
            regexp_extract(col("_metadata.file_path"), r"cbb(\d{4})\.csv", 1).cast(
                IntegerType()
            ),
        )
        # Standardize N/A fields
        .withColumn("postseason_result", na_as_null(col("postseason_result")))
        .withColumn("tournament_seed", parse_seed(normalize_na(col("tournament_seed"))))
        # Drop RK column as it is only present in 2020 and 2025 datasets
        # Drop undocumented 3PR fields; assumed to be absolute 3-point numbers. Using percentage fields instead of absolute numbers
        # Drop undocumented EFG; assumed to be some version of field goal %, but dropping due to not being in all datasets
        .drop("RK", "3PR", "3PRD", "EFGD_D", "EFG%", "EFGD%")
    )
    return cleaned_data
