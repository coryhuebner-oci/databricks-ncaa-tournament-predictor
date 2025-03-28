from pyspark.sql import DataFrame
import tensorflow as tf
import keras

from ncaa_tournament_predictor.tensorflow_models.text_embedding import (
    create_text_embedding_layers,
)

_text_feature_columns = ["team_1", "t1_conference", "team_2", "t2_conference"]

_numeric_feature_columns = [
    "t1_games",
    "t1_wins",
    "t1_adjusted_offensive_efficiency",
    "t1_adjusted_defensive_efficiency",
    "t1_power_rating",
    "t1_effective_field_goal_percentage",
    "t1_effective_field_goal_percentage_allowed",
    "t1_turnover_rate",
    "t1_caused_turnover_rate",
    "t1_offensive_rebound_rate",
    "t1_defensive_rebound_rate",
    "t1_freethrows_attempted_rate",
    "t1_freethrows_allowed_rate",
    "t1_two_point_shooting_percentage",
    "t1_two_point_shooting_percentage_allowed",
    "t1_three_point_shooting_percentage",
    "t1_three_point_shooting_percentage_allowed",
    "t1_adjusted_tempo",
    "t1_wins_above_bubble",
    "t1_tournament_seed",
    "t2_games",
    "t2_wins",
    "t2_adjusted_offensive_efficiency",
    "t2_adjusted_defensive_efficiency",
    "t2_power_rating",
    "t2_effective_field_goal_percentage",
    "t2_effective_field_goal_percentage_allowed",
    "t2_turnover_rate",
    "t2_caused_turnover_rate",
    "t2_offensive_rebound_rate",
    "t2_defensive_rebound_rate",
    "t2_freethrows_attempted_rate",
    "t2_freethrows_allowed_rate",
    "t2_two_point_shooting_percentage",
    "t2_two_point_shooting_percentage_allowed",
    "t2_three_point_shooting_percentage",
    "t2_three_point_shooting_percentage_allowed",
    "t2_adjusted_tempo",
    "t2_wins_above_bubble",
    "t2_tournament_seed",
]
label_column = "team_1_won"
_all_columns = _numeric_feature_columns + _text_feature_columns + [label_column]


def _spark_tensor_generator(df: DataFrame):
    """Read the incoming Spark DataFrame and convert each row into a tensor. Use
    an iterator to faciliate streaming a large dataset rather than loading it all
    into memory at once"""

    def _underlying_generator():
        for row in df.toLocalIterator():  # Stream rows one by one
            yield {column: row[column] for column in _all_columns}

    return _underlying_generator


def _get_input_tensor_schema():
    """Get the schema for the incoming input tensors; these are the heterogeneous (mix of strings, numbers, etc)
    from the incoming Spark Dataframe. This will later be pre-processed to become an all-numeric tensor for
    model usage"""
    text_tensor_inputs = {
        text_feature_column: tf.TensorSpec(shape=(), dtype=tf.string)
        for text_feature_column in _text_feature_columns
    }
    numeric_tensor_inputs = {
        numeric_feature_column: tf.TensorSpec(shape=(), dtype=tf.float32)
        for numeric_feature_column in _numeric_feature_columns
    }
    return {
        **text_tensor_inputs,
        **numeric_tensor_inputs,
        "team_1_won": tf.TensorSpec(shape=(), dtype=tf.bool),
    }


def _get_preprocessor(
    team_vectorizer: keras.layers.TextVectorization,
    conference_vectorizer: keras.layers.TextVectorization,
    stats_normalizer: keras.layers.Normalization,
):
    """Get the preprocessor that can be used for converting raw data into data suitable for feeding the model. This includes:
    1. Text-to-number conversion
    2. Normalizing data to avoid large-number bias
    """

    def preprocessor(training_data):
        # Vectorize text columns
        team_1_index = team_vectorizer(training_data["team_1"])
        team_2_index = team_vectorizer(training_data["team_2"])
        team_1_conference_index = conference_vectorizer(training_data["t1_conference"])
        team_2_conference_index = conference_vectorizer(training_data["t2_conference"])

        # Normalize numeric columns
        numeric_columns = tf.stack(
            [training_data[column] for column in _numeric_feature_columns]
        )
        normalized_stat_fields = stats_normalizer(numeric_columns)

        # Convert the boolean team_1_won field into a bit (1 for True, 0 for False)
        team1_won_numeric = tf.cast(training_data["team_1_won"], tf.float32)

        return {
            "team_1_index": team_1_index,
            "team_2_index": team_2_index,
            "team_1_conference_index": team_1_conference_index,
            "team_2_conference_index": team_2_conference_index,
            "normalized_stats": normalized_stat_fields,
            "team_1_won": team1_won_numeric,
        }

    return preprocessor


def get_game_prediction_dataset(training_data_df: DataFrame, team_stats_df: DataFrame):
    """Convert a Spark dataframe of training data into a Tensorflow dataset suitable for use in an ML model
    training_data_df: A dataset suitable for training (features and labels); fields in this dataset will
    be converted into numeric representation and normalized to feed a model
    team_stats_df: This dataset is used to understand what kind of vectorization/embeddings need to be
    created for text fields in the training dataset (e.g. distinct teams, distinct conferences, etc.)
    """

    # First, convert the incoming Spark DataFrame into a tensor
    input_tensor_schema = _get_input_tensor_schema()
    dataset = tf.data.Dataset.from_generator(
        _spark_tensor_generator(training_data_df),
        output_signature=input_tensor_schema,
    )

    # Next, get layers required for preprocessing based on the input data
    team_embedding_layers = create_text_embedding_layers(team_stats_df, "team", 3)
    conference_embedding_layers = create_text_embedding_layers(
        team_stats_df, "conference", 1
    )
    # Just use a sample of the dataset to get a numeric field normalizer
    sample_seed = 105  # Arbitrary, but jconsistent
    numeric_fields_sample = (
        training_data_df.sample(fraction=0.1, seed=sample_seed)
        .select(*_numeric_feature_columns)
        .collect()
    )
    numeric_fields_sample_tensor = tf.convert_to_tensor(
        numeric_fields_sample, dtype=tf.float32
    )
    stats_normalizer = keras.layers.Normalization()
    stats_normalizer.adapt(numeric_fields_sample_tensor)

    # Then, convert that input tensor into data suitable for a model (e.g. text to numbers, normalization, etc)
    preprocessor = _get_preprocessor(
        team_embedding_layers.vectorizer,
        conference_embedding_layers.vectorizer,
        stats_normalizer,
    )
    return dataset.map(preprocessor).batch(8)
