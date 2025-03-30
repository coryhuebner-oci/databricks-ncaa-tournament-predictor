import tensorflow as tf
from keras import layers

from ncaa_tournament_predictor.tensorflow_models.game_prediction.text_layers import (
    supported_words_per_team,
    supported_words_per_conference,
)
from ncaa_tournament_predictor.tensorflow_models.text_embedding import (
    create_embedding_layer,
)


def create_model(
    numeric_feature_columns: list[str],
    teams_vectorizer: layers.TextVectorization,
    conference_vectorizer: layers.TextVectorization,
):
    """Create a model to predict game winners.
    This method uses the text vectorizers from pre-processing as input to determine the proper dimensions for text embedding layers
    """

    # Define input layers
    normalized_stats_input = tf.keras.layers.Input(
        shape=(
            1,
            len(numeric_feature_columns),
        ),
        name="normalized_stats",
    )
    team_1_input = tf.keras.layers.Input(
        shape=(supported_words_per_team,), dtype=tf.int32, name="team_1_vector"
    )
    team_2_input = tf.keras.layers.Input(
        shape=(supported_words_per_team,), dtype=tf.int32, name="team_2_vector"
    )
    team_1_conference_input = tf.keras.layers.Input(
        shape=(supported_words_per_conference,),
        dtype=tf.int32,
        name="team_1_conference_vector",
    )
    team_2_conference_input = tf.keras.layers.Input(
        shape=(supported_words_per_conference,),
        dtype=tf.int32,
        name="team_2_conference_vector",
    )

    # Team and conference embedding layers for dense text representation
    team_1_embedding = create_embedding_layer(teams_vectorizer, "team_1_embedding")(
        team_1_input
    )
    team_2_embedding = create_embedding_layer(teams_vectorizer, "team_2_embedding")(
        team_2_input
    )
    team_1_conference_embedding = create_embedding_layer(
        conference_vectorizer, "team_1_conference_embedding"
    )(team_1_conference_input)
    team_2_conference_embedding = create_embedding_layer(
        conference_vectorizer, "team_2_conference_embedding"
    )(team_2_conference_input)

    # Flatten team and conference embeddings
    flattened_team_1_embedding = tf.keras.layers.Flatten()(team_1_embedding)
    flattened_team_2_embedding = tf.keras.layers.Flatten()(team_2_embedding)
    flattened_team_1_conference_embedding = tf.keras.layers.Flatten()(
        team_1_conference_embedding
    )
    flattened_team_2_conference_embedding = tf.keras.layers.Flatten()(
        team_2_conference_embedding
    )

    # Normalize the stats input (the shape is (None, 1, <feature_columns>) after normalization)
    # Reshape to (None, <feature_columns>) before flattening
    flattened_stats_input = tf.keras.layers.Reshape((len(numeric_feature_columns),))(
        normalized_stats_input
    )

    # Concatenate all features
    combined_layers = tf.keras.layers.Concatenate(axis=1)(
        [
            flattened_stats_input,
            flattened_team_1_embedding,
            flattened_team_2_embedding,
            flattened_team_1_conference_embedding,
            flattened_team_2_conference_embedding,
        ]
    )

    # Define the brains part of the model
    brains = tf.keras.layers.Dense(128, activation="relu")(combined_layers)
    brains = tf.keras.layers.Dropout(0.2)(brains)
    brains = tf.keras.layers.Dense(64, activation="relu")(brains)
    brains = tf.keras.layers.Dropout(0.2)(brains)
    brains = tf.keras.layers.Dense(32, activation="relu")(brains)
    # Binary classification
    output = tf.keras.layers.Dense(1, activation="sigmoid")(brains)

    # Build the model
    model = tf.keras.Model(
        inputs=[
            normalized_stats_input,
            team_1_input,
            team_2_input,
            team_1_conference_input,
            team_2_conference_input,
        ],
        outputs=output,
    )
    return model
