from dataclasses import dataclass
from pyspark.sql import DataFrame
import tensorflow as tf
import keras

from ncaa_tournament_predictor.tensorflow_models.game_prediction.text_layers import (
    get_team_vectorization_layer,
    get_conference_vectorization_layer,
)
from ncaa_tournament_predictor.tensorflow_models.game_prediction import columns


def _spark_tensor_generator(df: DataFrame, numeric_feature_columns: list[str]):
    """Read the incoming Spark DataFrame and convert each row into a tensor. Use
    an iterator to faciliate streaming a large dataset rather than loading it all
    into memory at once"""

    all_columns = columns.text_feature_columns + numeric_feature_columns

    def _underlying_generator():
        for row in df.toLocalIterator():  # Stream rows one by one
            yield {column: row[column] for column in all_columns}, row[
                columns.label_column
            ]

    return _underlying_generator


def _get_input_tensor_schema(numeric_feature_columns: list[str]):
    """Get the schema for the incoming input tensors; these are the heterogeneous (mix of strings, numbers, etc)
    from the incoming Spark Dataframe. This will later be pre-processed to become an all-numeric tensor for
    model usage"""
    text_tensor_inputs = {
        text_feature_column: tf.TensorSpec(shape=(), dtype=tf.string)
        for text_feature_column in columns.text_feature_columns
    }
    numeric_tensor_inputs = {
        numeric_feature_column: tf.TensorSpec(shape=(), dtype=tf.float32)
        for numeric_feature_column in numeric_feature_columns
    }
    return (
        {
            **text_tensor_inputs,
            **numeric_tensor_inputs,
        },
        tf.TensorSpec(shape=(), dtype=tf.bool),
    )


def get_features_preprocessor(
    numeric_feature_columns: list[str],
    team_vectorizer: keras.layers.TextVectorization,
    conference_vectorizer: keras.layers.TextVectorization,
    stats_normalizer: keras.layers.Normalization,
):
    """Get the preprocessor that can be used for converting raw features into model-ready data. This includes:
    1. Text-to-number conversion
    2. Normalizing data to avoid large-number bias
    """

    def preprocessor(features):
        # Vectorize text columns
        team_1_vector = team_vectorizer(features["team_1"])
        team_2_vector = team_vectorizer(features["team_2"])
        team_1_conference_vector = conference_vectorizer(features["t1_conference"])
        team_2_conference_vector = conference_vectorizer(features["t2_conference"])

        # Normalize numeric columns
        numeric_columns = tf.stack(
            [
                tf.cast(features[column], tf.float32)
                for column in numeric_feature_columns
            ],
            axis=-1,
        )
        normalized_stat_fields = stats_normalizer(numeric_columns)

        return {
            "team_1_vector": team_1_vector,
            "team_2_vector": team_2_vector,
            "team_1_conference_vector": team_1_conference_vector,
            "team_2_conference_vector": team_2_conference_vector,
            "normalized_stats": normalized_stat_fields,
        }

    return preprocessor


def get_training_data_preprocessor(
    features_preprocessor,
):
    """Get the preprocessor that can be used for converting raw data into data suitable for feeding the model. This includes:
    1. Text-to-number conversion
    2. Normalizing data to avoid large-number bias
    3. Convert the label field ("team_1_won") from a boolean into a number
    """

    def preprocessor(features, label):
        # Convert the boolean team_1_won field into a bit (1 for True, 0 for False)
        team1_won_numeric = tf.cast(label, tf.float32)
        return (features_preprocessor(features), team1_won_numeric)

    return preprocessor


@dataclass
class GamePredictionPreprocessingLayers:
    team_vectorizer: keras.layers.TextVectorization
    conference_vectorizer: keras.layers.TextVectorization
    stats_normalizer: keras.layers.Normalization


def get_data_preprocessing_layers(
    training_data_df: DataFrame,
    numeric_feature_columns: list[str],
    team_stats_df: DataFrame,
) -> GamePredictionPreprocessingLayers:
    """Get the Keras layers needed for pre-processing data from Spark. These are built in their
    own method to allow re-use across model train, test, etc.
    These include:
    1. Text vectorization layers for text fields
    2. Normalization layers to put numeric statistics on normalized scales to avoid bias
    """
    # Next, get layers required for preprocessing based on the input data
    team_vectorizer = get_team_vectorization_layer(team_stats_df)
    conference_vectorizer = get_conference_vectorization_layer(team_stats_df)
    # Just use a sample of the dataset to get a numeric field normalizer
    sample_seed = 105  # Arbitrary, but consistent across runs
    numeric_fields_sample = (
        training_data_df.sample(fraction=0.1, seed=sample_seed)
        .select(*numeric_feature_columns)
        .collect()
    )
    numeric_fields_sample_tensor = tf.convert_to_tensor(
        numeric_fields_sample, dtype=tf.float32
    )
    stats_normalizer = keras.layers.Normalization()
    stats_normalizer.adapt(numeric_fields_sample_tensor)
    return GamePredictionPreprocessingLayers(
        team_vectorizer, conference_vectorizer, stats_normalizer
    )


def get_preprocessed_game_prediction_training_dataset(
    training_data_df: DataFrame,
    numeric_feature_columns: list[str],
    preprocessor,
):
    """Convert a Spark dataframe of training data into a Tensorflow dataset suitable for use in an ML model
    training_data_df: A dataset suitable for training (features and labels); fields in this dataset will
    be converted into numeric representation and normalized to feed a model
    Args:
        training_data_df (DataFrame): The Dataframe containing the dataset to feed the model
        numeric_feature_columns (list[str]): The fields representing numeric features
        preprocessor: The function that preprocesses raw data to format for machine learning

    Returns:
        DatasetV2: The Tensorflow dataset with numeric representation of all input data
    """

    # First, convert the incoming Spark DataFrame into a tensor
    input_tensor_schema = _get_input_tensor_schema(numeric_feature_columns)
    dataset = tf.data.Dataset.from_generator(
        _spark_tensor_generator(training_data_df, numeric_feature_columns),
        output_signature=input_tensor_schema,
    )

    # Then, convert that input tensor into data suitable for a model (e.g. text to numbers, normalization, etc)
    return dataset.map(preprocessor).batch(32)
