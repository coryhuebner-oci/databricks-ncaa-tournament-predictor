# References to volumes held in the NCAA mens basketball schema

raw_kaggle_stats = (
    "dbfs:/Volumes/object_computing/ncaa_mens_basketball/raw_kaggle_stats/"
)
raw_head_to_head = "dbfs:/Volumes/object_computing/ncaa_mens_basketball/head_to_head/"


def without_dbfs_protocol(volume: str) -> str:
    """Get a volume without the DBFS protocol prefix; this is useful for certain
    helper methods in Databricks that are explicitly tied to volumes and don't allow
    the protocol prefix"""
    return volume.removeprefix("dbfs:")


def as_sql_object(volume: str) -> str:
    """Convert a volume path into a SQL object name"""
    return (
        without_dbfs_protocol(volume)
        .replace("/", ".")
        .removeprefix(".Volumes.")
        .removesuffix(".")
    )
