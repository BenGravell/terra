"""List of supported countries in alphabetic order"""

# Developer note: generate supported_countries.json with maintenance/get_supported_countries.py

import pandas as pd

SUPPORTED_COUNTRIES = pd.read_csv("./data/supported_countries.csv", header=None)[0].to_list()

if __name__ == "__main__":
    print(SUPPORTED_COUNTRIES)
