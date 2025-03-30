# A helper module to create text embedding layers

from dataclasses import dataclass
import math
from typing import Literal

import keras
from pyspark.sql import DataFrame


def _get_unique(team_stats_df: DataFrame, field_name: str) -> list[str]:
    """Get a list of unique values from a Dataframe; only use this when your dataset is
    of a reasonable size to loop through to find distinct values"""
    return [
        row[field_name]
        for row in team_stats_df.select(field_name).distinct().toLocalIterator()
    ]


def get_embedding_output_size(distinct_data_size: int) -> int:
    """Use the 4 * sqrt(distinct data size) rule-of-thumb coupled with
    the optimization to round to a power of 2 in order to get the optimal
    size of an embedding output dimension for text embedding"""
    raw_output_size = 4 * math.sqrt(distinct_data_size)
    return 2 ** round(math.log2(raw_output_size))


StandardizationOptions = Literal[
    "lower_and_strip_punctuation", "lower", "strip_punctuation"
]


def create_vectorization_layer(
    df: DataFrame,
    field_name: str,
    supported_word_count_per_row: int,
    standardize: StandardizationOptions | None = None,
) -> keras.layers.TextVectorization:
    """Given a Dataframe, get all distinct values from the given field to create a
    vectorization layer for values in that field
    Args:
        df (DataFrame): The Dataframe containing the entire dataset for analysis
        field_name (str): The name of the field to build text embedding layers for
        supported_word_count_per_row (int): How many words should be supported for each entry in the field.
        E.g. if your dataset typically consists of 1-3 words per row in the field, use a number somewhere
        between 1-3. Note: In that example, if you chose a number less than the number of fields it might result in
        lossiness in your model interpretation of the field, but it won't cause any errors. It's a trade-off
        between model intelligence and speed to choose a higher vs. lower number. However, setting this higher
        than the max words-per-row will not be beneficial and just slow things down
        standardize (StandardizationOptions | None): Standardization options for transforming the field's text
        before tokenizing it

    Returns:
        TextVectorization: The vectorization layers responsible for converting text fields into numeric representation
    """
    unique_values = _get_unique(df, field_name)
    return keras.layers.TextVectorization(
        output_mode="int",
        vocabulary=unique_values,
        output_sequence_length=supported_word_count_per_row,
        standardize=standardize,
    )


def create_embedding_layer(
    vectorizer: keras.layers.TextVectorization, layer_name: str | None = None
) -> keras.layers.Embedding:
    """Get an embedding layer corresponding to the given text vectorization layer.
    This method uses the text vectorization layer to determine how many output dimensions
    to include in the embedding layer for minimal loss"""
    embedding_input_size = vectorizer.vocabulary_size()
    embedding_output_size = get_embedding_output_size(embedding_input_size)
    return keras.layers.Embedding(
        input_dim=embedding_input_size,
        output_dim=embedding_output_size,
        name=layer_name,
    )
