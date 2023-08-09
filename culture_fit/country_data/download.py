import json
from culture_fit.country_data import types


def load_country_data() -> types.JSONType:
    with open("./data/culture_dimensions.json", "r") as f:
        return json.load(f)
