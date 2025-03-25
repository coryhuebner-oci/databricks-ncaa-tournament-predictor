from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_extract, Column, when


def normalize_na(column: Column) -> Column:
    """Normalize the NA and N/A values to all be consistently N/A"""
    return when(column == "NA", "N/A").otherwise(column)


def get_cleaned_kaggle_stats(df: DataFrame) -> DataFrame:
    return (
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
            "year", regexp_extract(col("_metadata.file_path"), r"cbb(\d{4})\.csv", 1)
        )
        # Standardize N/A fields
        .withColumn("postseason_result", normalize_na(col("postseason_result")))
        .withColumn("tournament_seed", normalize_na(col("tournament_seed")))
        # Drop RK column as it is only present in 2020 and 2025 datasets
        .drop("RK")
        # Drop undocumented fields; assumed to be absolute 3-point numbers. Using percentage fields instead of absolute numbers
        .drop("3PR")
        .drop("3PRD")
        # Drop undocumented fields; assumed to be some version of field goal %, but dropping due to not being in all datasets
        .drop("EFGD_D")
        .drop("EFG%")
        .drop("EFGD%")
    )
