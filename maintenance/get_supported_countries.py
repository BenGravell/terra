import pandas as pd

from culture_fit import country_data


culture_fit_data_dict = country_data.get_country_dict()
culture_fit_df = pd.DataFrame.from_dict(culture_fit_data_dict, orient="index").rename_axis("country").reset_index()

culture_fit_countries_set = set(culture_fit_df["country"])

other_data_source_paths = [
    "./data/happy_planet_index_2019.csv",
    "./data/social_progress_index_2022.csv",
    "./data/human-freedom-index-2022.csv",
    "./data/country_temperature.csv",
    "./data/country_sunshine_hours_per_day.csv",
]

other_coutries_set = set.intersection(*[set(pd.read_csv(path)["country"]) for path in other_data_source_paths])

supported_countries_set = set.intersection(culture_fit_countries_set, other_coutries_set)

supported_countries_list = sorted(list(supported_countries_set))
print(f"Number of Supported Countries: {len(supported_countries_list)}")
for country in supported_countries_list:
    print(country)


supported_countries_all_set = set.union(
    culture_fit_countries_set, set.union(*[set(pd.read_csv(path)["country"]) for path in other_data_source_paths])
)
supported_countries_all_list = sorted(list(supported_countries_all_set))

pd.DataFrame(supported_countries_all_list).to_csv("data/supported_countries.csv", index=False, header=False)
