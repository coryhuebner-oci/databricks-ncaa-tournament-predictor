from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg


def get_comparable_features_training_dataset(
    team_stats: DataFrame, head_to_head_results: DataFrame
) -> DataFrame:
    """Combine and transform data sources to create a training dataset for an ML model; this dataset
    provides calculated columns to compare different features between the two teams in each game
    """

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
    with_unneeded_columns_removed = joined_data.drop(
        "team_1_score",
        "team_2_score",
        "winning_team",
        "college_season",
        "t1_team",
        "t1_postseason_result",
        "t1_source_filename",
        "t2_team",
        "t2_postseason_result",
        "t2_source_filename",
    )

    avg_field_goal_percentage = team_stats.select(
        avg(col("effective_field_goal_percentage")).alias(
            "effective_field_goal_percentage"
        )
    ).first()["effective_field_goal_percentage"]
    with_filled_tournament_seeds = with_unneeded_columns_removed.fillna(
        0, ["t1_tournament_seed", "t2_tournament_seed"]
    )
    with_filled_field_goal_percentages = with_filled_tournament_seeds.fillna(
        avg_field_goal_percentage,
        [
            "t1_effective_field_goal_percentage",
            "t1_effective_field_goal_percentage_allowed",
            "t2_effective_field_goal_percentage",
            "t2_effective_field_goal_percentage_allowed",
        ],
    )
    return with_filled_field_goal_percentages
