WORLD_FACTBOOK_URL = "https://www.cia.gov/the-world-factbook"

# Ad-hoc adjustments based on looking at https://www.cia.gov/the-world-factbook/countries/
SPECIAL_CASES = {
    "Bahamas": "bahamas-the",
    "Cape Verde": "cabo-verde",
    "Congo, Republic of": "congo-republic-of-the",
    "Congo, Democratic Republic of": "congo-democratic-republic-of-the",
    "Cote d'Ivoire": "cote-divoire",
    "Czech Republic": "czechia",
    "Gambia": "gambia-the",
    "Myanmar": "burma",
    "Palestine": "west-bank",
    "South Korea": "korea-south",
    "North Korea": "korea-north",
    "Turkey": "turkey-turkiye",
}


def default_slugify(country: str) -> str:
    return country.lower().replace(" ", "-").replace("'", "")


def slugify(country: str) -> str:
    # Standard pattern
    return SPECIAL_CASES[country] if country in SPECIAL_CASES else default_slugify(country)


def get_world_factbook_url(country: str) -> str:
    return f"{WORLD_FACTBOOK_URL}/countries/{slugify(country)}/"


if __name__ == "__main__":
    from terra.data_handling.loading import DATA

    countries = sorted(list(DATA.merged_df["country"]))
    for country in countries:
        print(get_world_factbook_url(country))
