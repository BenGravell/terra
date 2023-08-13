import pandas as pd

from culture_fit import country_data, visualisation


def diff_col(obj1, obj2, col1=None, col2=None, name1=None, name2=None):
    if isinstance(obj1, str) and obj1.endswith(".csv"):
        path1 = obj1
        df1 = pd.read_csv(path1)
    else:
        df1 = obj1

    if isinstance(obj2, str) and obj2.endswith(".csv"):
        path2 = obj2
        df2 = pd.read_csv(path2)
    else:
        df2 = obj2

    if col1 is None:
        col1 = "country"
    if col2 is None:
        col2 = "country"

    if name1 is None:
        name1 = path1
    if name2 is None:
        name2 = path2

    print(f"only in {name1}, not in {name2}")
    s1 = set(df1[col1])
    s2 = set(df2[col2])
    diff = sorted(list(s1.difference(s2)))
    for val in diff:
        print(val)
    print("")


# Culture Fit
culture_fit_data_dict = country_data.get_country_dict()
culture_fit_df = pd.DataFrame.from_dict(culture_fit_data_dict, orient="index").rename_axis("country").reset_index()

# Country Flag URLS
country_urls_df = pd.DataFrame.from_dict(visualisation.COUNTRY_URLS, orient="index")
country_urls_df = country_urls_df.rename_axis("country").reset_index()


# Checks against country_codes_alpha_3, which is the most complete list.
# This is to harmonize the names that are present across all datasets.

diff_col("./data/country_coords.csv", "./data/country_codes_alpha_3.csv")
# known_missing: TODO

diff_col(country_urls_df, "./data/country_codes_alpha_3.csv", name1="country_urls_df")
# known missing: South Ossetia, Vatican City

diff_col(culture_fit_df, "./data/country_codes_alpha_3.csv", name1="culture_fit_df")
# known missing: None

diff_col("./data/happy_planet_index_2019.csv", "./data/country_codes_alpha_3.csv")
# known missing: None

diff_col("./data/social_progress_index_2022.csv", "./data/country_codes_alpha_3.csv")
# known missing: West Bank and Gaza

diff_col("./data/human-freedom-index-2022.csv", "./data/country_codes_alpha_3.csv")
# known missing: None

diff_col("./data/english_speaking.csv", "./data/country_codes_alpha_3.csv")
# known missing: None

diff_col("./data/country_temperature.csv", "./data/country_codes_alpha_3.csv")
# known missing: None

# Checks from source-to-source (sequential)

diff_col(culture_fit_df, "./data/happy_planet_index_2019.csv", name1="culture_fit_df")
# known missing: Angola, Cape Verde, Fiji, Puerto Rico, Sao Tome and Principe, Suriname, Syria

diff_col("./data/happy_planet_index_2019.csv", "./data/social_progress_index_2022.csv")
# known missing: Hong Kong, Palestine, Taiwan, Vanuatu

diff_col("./data/social_progress_index_2022.csv", "./data/human-freedom-index-2022.csv")
# known missing: Afghanistan, Equatorial Guinea, Eritrea, Kyrgyzstan, Laos, Maldives, Sao Tome and Principe,
# Slovakia, Solomon Islands, South Sudan, Turkmenistan, Uzbekistan, West Bank and Gaza

diff_col("./data/human-freedom-index-2022.csv", "./data/english_speaking.csv")
# known missing: a lot...
