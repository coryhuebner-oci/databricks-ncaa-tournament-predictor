from os import getenv


def get_required(environment_variable: str) -> str:
    """Get the required environment variable and throw an error if unset"""
    config_value = getenv(environment_variable)
    if config_value is None or config_value == "":
        raise KeyError(f"Required environment variable not set: {environment_variable}")
    return config_value


class Config:
    """Configuration for the application"""

    @staticmethod
    def databricks_profile() -> str:
        return get_required("DATABRICKS_PROFILE")

    @staticmethod
    def databricks_serverless() -> bool:
        config_value = getenv("DATABRICKS_SERVERLESS")
        return config_value is not None and config_value.lower() in ["true", 1]
