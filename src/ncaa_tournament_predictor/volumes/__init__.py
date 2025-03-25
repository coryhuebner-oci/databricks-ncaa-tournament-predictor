# References to volumes held in the NCAA mens basketball schema

raw_kaggle_stats = (
    "dbfs:/Volumes/object_computing/ncaa_mens_basketball/raw_kaggle_stats/"
)


def without_dbfs_protocol(volume: str) -> str:
    """Get a volume without the DBFS protocol prefix; this is useful for certain
    helper methods in Databricks that are explicitly tied to volumes and don't allow
    the protocol prefix"""
    return volume.lstrip("dbfs:")
