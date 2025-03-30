from pyspark.sql import DataFrame
from keras import layers

from ncaa_tournament_predictor.tensorflow_models.text_embedding import (
    create_vectorization_layer,
)

supported_words_per_team = 4
supported_words_per_conference = 1


def get_team_vectorization_layer(team_stats_df: DataFrame) -> layers.TextVectorization:
    """Get the text vectorization pre-processing layer for teams"""
    return create_vectorization_layer(
        team_stats_df, "team", supported_word_count_per_row=supported_words_per_team
    )


def get_conference_vectorization_layer(
    team_stats_df: DataFrame,
) -> layers.TextVectorization:
    """Get the text vectorization pre-processing layer for conferences"""
    return create_vectorization_layer(
        team_stats_df,
        "conference",
        supported_word_count_per_row=supported_words_per_conference,
    )
