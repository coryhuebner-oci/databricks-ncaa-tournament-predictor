catalog = "object_computing"
schema = "ncaa_mens_basketball"


def _with_default_schema(object_name: str) -> str:
    return f"{catalog}.{schema}.{object_name}"


cleaned_kaggle_stats = _with_default_schema("cleaned_kaggle_stats")
cleaned_head_to_head_results = _with_default_schema("cleaned_head_to_head_results")
game_prediction_feature_set = _with_default_schema("game_prediction_features")
