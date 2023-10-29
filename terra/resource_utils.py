import pkg_resources
from functools import partial


def get_pkg_resource_file_path(filename: str, prefix: str):
    return pkg_resources.resource_filename("terra", f"{prefix}/{filename}")


get_data_file_path = partial(get_pkg_resource_file_path, prefix="data")
get_help_file_path = partial(get_pkg_resource_file_path, prefix="help")
get_assets_file_path = partial(get_pkg_resource_file_path, prefix="assets")


def get_assets_image_data(filename: str):
    with open(get_assets_file_path(filename), "rb") as f:
        return f.read()
