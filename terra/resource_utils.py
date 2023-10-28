import pkg_resources
from functools import partial


def get_pkg_resource_file_path(filename: str, prefix: str):
    return pkg_resources.resource_filename("terra", f"{prefix}/{filename}")


get_data_file_path = partial(get_pkg_resource_file_path, prefix="data")
get_help_file_path = partial(get_pkg_resource_file_path, prefix="help")
get_assets_file_path = partial(get_pkg_resource_file_path, prefix="assets")
