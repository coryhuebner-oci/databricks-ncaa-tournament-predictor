catalog = "object_computing"
schema = "ncaa_mens_basketball"


def _with_default_schema(object_name: str) -> str:
    return f"{catalog}.{schema}.{object_name}"


_cleaned_kaggle_stats = "cleaned_kaggle_stats"
_cleaned_head_to_head_results = "cleaned_head_to_head_results"
_game_prediction_features = "game_prediction_features"

cleaned_kaggle_stats = _with_default_schema(_cleaned_kaggle_stats)
cleaned_head_to_head_results = _with_default_schema(_cleaned_head_to_head_results)
game_prediction_feature_set = _with_default_schema(_game_prediction_features)
