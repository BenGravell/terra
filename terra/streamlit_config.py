"""Load and expose the .streamlit/config.toml file as a Python dict."""

import tomllib


STREAMLIT_CONFIG_PATH = "./.streamlit/config.toml"

with open(STREAMLIT_CONFIG_PATH, "rb") as f:
    STREAMLIT_CONFIG = tomllib.load(f)


if __name__ == "__main__":
    from pprint import pprint

    pprint(STREAMLIT_CONFIG)
