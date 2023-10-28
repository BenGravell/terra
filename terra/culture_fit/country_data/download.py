import json
import pkg_resources

from terra.culture_fit.country_data import types


def load_country_data() -> types.JSONType:
    path = pkg_resources.resource_filename("terra", "data/culture_dimensions.json")
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import pprint

    pprint.pprint(load_country_data())
