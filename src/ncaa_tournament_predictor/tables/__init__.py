catalog = "object_computing"
schema = "ncaa_mens_basketball"


def _with_default_schema(object_name: str) -> str:
    return f"{catalog}.{schema}.{object_name}"


cleaned_kaggle_stats = _with_default_schema("cleaned_kaggle_stats")
