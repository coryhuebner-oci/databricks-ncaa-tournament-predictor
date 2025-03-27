from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def get_training_dataset(
    team_stats: DataFrame, head_to_head_results: DataFrame
) -> DataFrame:
    """Combine and transform data sources to create a training dataset for an ML model"""

    # h2h = head-to-head
    h2h_alias = "h2h"
    head_to_head_columns = [
        col(f"{h2h_alias}.{column}") for column in head_to_head_results.columns
    ]

    # t1 = team 1
    t1_stats_alias = "t1_stats"
    t1_stats_columns = [
        col(f"{t1_stats_alias}.{column}").alias(f"t1_{column}")
        for column in team_stats.columns
        if column != "college_season"
    ]

    # t2 = team 2
    t2_stats_alias = "t2_stats"
    t2_stats_columns = [
        col(f"{t2_stats_alias}.{column}").alias(f"t2_{column}")
        for column in team_stats.columns
        if column != "college_season"
    ]

    joined_data = (
        head_to_head_results.alias(h2h_alias)
        .join(
            team_stats.alias(t1_stats_alias),
            on=(col(f"{h2h_alias}.team_1") == col(f"{t1_stats_alias}.team"))
            & (
                col(f"{h2h_alias}.college_season")
                == col(f"{t1_stats_alias}.college_season")
            ),
        )
        .join(
            team_stats.alias(t2_stats_alias),
            on=(col(f"{h2h_alias}.team_2") == col(f"{t2_stats_alias}.team"))
            & (
                col(f"{h2h_alias}.college_season")
                == col(f"{t2_stats_alias}.college_season")
            ),
        )
        .select(head_to_head_columns + t1_stats_columns + t2_stats_columns)
    )
    return joined_data
