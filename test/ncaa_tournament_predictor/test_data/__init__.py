import os


def get_file_path(test_data_filename: str) -> str:
    """Get the absolute file path for a test file from this directory"""
    return os.path.join(os.path.dirname(__file__), test_data_filename)


def get_test_data_contents(test_data_filename: str, encoding: str = "utf-8") -> str:
    """A helper to make it easier to load test data files from this directory, regardless
    of the caller's directory"""
    full_test_filepath = get_file_path(test_data_filename)
    with open(full_test_filepath, "r", encoding=encoding) as file:
        return file.read()
