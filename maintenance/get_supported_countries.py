import pandas as pd

from culture_fit import country_data


culture_fit_data_dict = country_data.get_country_dict()
culture_fit_df = pd.DataFrame.from_dict(culture_fit_data_dict, orient="index").rename_axis("country").reset_index()

culture_fit_countries_set = set(culture_fit_df["country"])

score_data_source_paths = [
    "./data/happy_planet_index_2019.csv",
    "./data/social_progress_index_2022.csv",
    "./data/human-freedom-index-2022.csv",
]

score_coutries_set = set.union(*[set(pd.read_csv(path)["country"]) for path in score_data_source_paths])

supported_countries_set = set.union(culture_fit_countries_set, score_coutries_set)

supported_countries_list = sorted(list(supported_countries_set))
print(supported_countries_list)
