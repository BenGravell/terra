# Terra
*Want to jump right in? Go to the deployed app on Streamlit Community Cloud:*
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://terra-country-recommender.streamlit.app/)

## What is this?
This app is designed to answer the question "which country should I live in?"

Use data to decide which country is right for you. Terra will take your personal preferences regarding Culture Fit, Human Freedom, and Language into account and recommend one or more countries that match.

## Provenance
This app is largely based off of Michał Nowotka's tutorial ([Blog](https://blog.streamlit.io/how-to-make-a-culture-map/)) ([GitHub](https://github.com/streamlit/demo-culture-map)), which is partially used (with modifications) as a subpackage.

## Data Sources
The data included in the `data` directory was collected from the following sources:
- `human-freedom-index-2022.csv`: https://www.cato.org/human-freedom-index/2022
- `english_speaking.csv`: https://en.wikipedia.org/wiki/List_of_countries_by_English-speaking_population
- `country_coords.csv`: https://developers.google.com/public-data/docs/canonical/countries_csv