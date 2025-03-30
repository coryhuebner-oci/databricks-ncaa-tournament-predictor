# Get compared-features dataset

from pyspark.sql import SparkSession, DataFrame

from ncaa_tournament_predictor import tables

_non_tournament_team_seed = 20

_t1_t2_stats_selection: str = f"""
    t1_fs.team as team_1,
    t2_fs.team as team_2,
    t1_fs.conference as t1_conference,
    t2_fs.conference as t2_conference,
    t1_fs.games - t2_fs.games as games_difference,
    t1_fs.wins - t2_fs.wins as wins_difference,
    t1_fs.losses - t2_fs.losses as losses_difference,
    t1_fs.adjusted_offensive_efficiency - t2_fs.adjusted_offensive_efficiency as offensive_effeciency_difference,
    t1_fs.adjusted_defensive_efficiency - t2_fs.adjusted_defensive_efficiency as defensive_effeciency_difference,
    t1_fs.adjusted_offensive_efficiency - t2_fs.adjusted_defensive_efficiency as t1_efficiency_spread,
    t2_fs.adjusted_offensive_efficiency - t1_fs.adjusted_defensive_efficiency as t2_efficiency_spread,
    t1_fs.power_rating - t2_fs.power_rating as power_rating_difference,
    t1_fs.effective_field_goal_percentage - t2_fs.effective_field_goal_percentage as effective_field_goal_percentage_difference,
    t1_fs.effective_field_goal_percentage_allowed - t2_fs.effective_field_goal_percentage_allowed as effective_field_goal_percentage_allowed_difference,
    t1_fs.effective_field_goal_percentage - t2_fs.effective_field_goal_percentage_allowed as t1_effective_field_goal_percentage_spread,
    t2_fs.effective_field_goal_percentage - t1_fs.effective_field_goal_percentage_allowed as t2_effective_field_goal_percentage_spread,
    t1_fs.turnover_rate - t2_fs.turnover_rate as turnover_rate_difference,
    t1_fs.turnover_rate - t2_fs.caused_turnover_rate as t1_turnover_rate_spread,
    t2_fs.turnover_rate - t1_fs.caused_turnover_rate as t2_turnover_rate_spread,
    t1_fs.offensive_rebound_rate - t2_fs.offensive_rebound_rate as offensive_rebound_rate_difference,
    t1_fs.defensive_rebound_rate - t2_fs.defensive_rebound_rate as defensive_rebound_rate_difference,
    t1_fs.offensive_rebound_rate - t2_fs.defensive_rebound_rate as t1_offensive_rebound_rate_spread,
    t2_fs.offensive_rebound_rate - t1_fs.defensive_rebound_rate as t2_offensive_rebound_rate_spread,
    t1_fs.freethrows_attempted_rate - t2_fs.freethrows_attempted_rate as freethrows_attempted_rate_difference,
    t1_fs.freethrows_allowed_rate - t2_fs.freethrows_allowed_rate as freethrows_allowed_rate_difference,
    t1_fs.two_point_shooting_percentage - t2_fs.two_point_shooting_percentage as two_point_shooting_percentage_difference,
    t1_fs.two_point_shooting_percentage_allowed - t2_fs.two_point_shooting_percentage_allowed as two_point_shooting_percentage_allowed_difference,
    t1_fs.three_point_shooting_percentage - t2_fs.three_point_shooting_percentage as three_point_shooting_percentage_difference,
    t1_fs.three_point_shooting_percentage_allowed - t2_fs.three_point_shooting_percentage_allowed as three_point_shooting_percentage_allowed_difference,
    t1_fs.adjusted_tempo - t2_fs.adjusted_tempo as adjusted_tempo_difference,
    t1_fs.wins_above_bubble - t2_fs.wins_above_bubble as wins_above_bubble_difference,
    coalesce(t1_fs.tournament_seed, {_non_tournament_team_seed}) - coalesce(t2_fs.tournament_seed, {_non_tournament_team_seed}) as tournament_seed_difference
"""


def _fill_unpopulated_field_goal_percentages(df: DataFrame) -> DataFrame:
    """Some teams don't have field goal percentage allowed stats; use this function
    just default differences to 0 for those match-ups"""
    return df.fillna(
        0.0,
        [
            "effective_field_goal_percentage_difference",
            "effective_field_goal_percentage_allowed_difference",
            "t1_effective_field_goal_percentage_spread",
            "t2_effective_field_goal_percentage_spread",
        ],
    )


def get_stats_differences_training_dataset(spark: SparkSession):
    sql = f"""
        select
            h2h.team_1_won,
            {_t1_t2_stats_selection}
        from {tables.cleaned_head_to_head_results} h2h
        join {tables.cleaned_kaggle_stats} t1_fs
            on h2h.team_1 = t1_fs.team
            and h2h.college_season = t1_fs.college_season
        join {tables.cleaned_kaggle_stats} t2_fs
            on h2h.team_2 = t2_fs.team
            and h2h.college_season = t2_fs.college_season
    """
    differences_df = spark.sql(sql)
    with_filled_field_goal_percentages = _fill_unpopulated_field_goal_percentages(
        differences_df
    )
    return with_filled_field_goal_percentages


def _escape_string(value: str) -> str:
    return value.replace("'", "\\'")


def get_stats_differences(
    spark: SparkSession, team_1: str, team_2: str, college_season: int
):
    sql = f"""
        select
            {_t1_t2_stats_selection}
        from {tables.cleaned_kaggle_stats} t1_fs
        cross join {tables.cleaned_kaggle_stats} t2_fs
        where t1_fs.team = '{_escape_string(team_1)}'
        and t1_fs.college_season = {college_season}
        and t2_fs.team = '{_escape_string(team_2)}'
        and t2_fs.college_season = {college_season}
    """
    differences_df = spark.sql(sql)
    with_filled_field_goal_percentages = _fill_unpopulated_field_goal_percentages(
        differences_df
    )
    return with_filled_field_goal_percentages
