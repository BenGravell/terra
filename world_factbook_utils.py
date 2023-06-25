import streamlit as st

from supported_countries import SUPPORTED_COUNTRIES


def default_slugify(country):
    return country.lower().replace(" ", "-").replace("'", "")


# Initialize with standard pattern
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG = {country: default_slugify(country) for country in SUPPORTED_COUNTRIES}

# Ad-hoc adjustments based on looking at https://www.cia.gov/the-world-factbook/countries/
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Bahamas"] = "bahamas-the"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Cape Verde"] = "cabo-verde"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Congo, Republic of"] = "congo-republic-of-the"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Congo, Democratic Republic of"] = "congo-democratic-republic-of-the"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Cote d'Ivoire"] = "cote-divoire"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Czech Republic"] = "czechia"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Gambia"] = "gambia-the"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Myanmar"] = "burma"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Palestine"] = "west-bank"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["South Korea"] = "korea-south"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["North Korea"] = "korea-north"
WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG["Turkey"] = "turkey-turkiye"


WORLD_FACTBOOK_URL_BASE = "https://www.cia.gov/the-world-factbook"


@st.cache_data
def get_world_factbook_url(country: str) -> str:
    country_slug = WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG[country]
    url = f"{WORLD_FACTBOOK_URL_BASE}/countries/{country_slug}/"
    return url


if __name__ == "__main__":
    countries = sorted(list(WORLD_FACTBOOK_COUNTRY_NAME_TO_SLUG))
    for country in countries:
        print(get_world_factbook_url(country))
