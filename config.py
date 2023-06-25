"""Load and expose the .streamlit/config.toml file as a Python dict."""

import tomllib

with open("./.streamlit/config.toml", "rb") as f:
    STREAMLIT_CONFIG = tomllib.load(f)
