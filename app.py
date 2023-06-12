import dataclasses

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from culture_map import country_data
from culture_map import distance_calculations
from culture_map import visualisation
from culture_map import dimensions_info

from options import AppOptions, NONE_COUNTRY, PLOTLY_MAP_PROJECTION_TYPES


# Streamlit setup
st.set_page_config(page_title="Terra", page_icon="🌎", layout="wide")
state = st.session_state

# Pyplot setup
plt.style.use(["dark_background", "./terra.mplstyle"])


@st.cache_data
def load_data():
    # Load all data sources
    # Human Freedom
    human_freedom_df = pd.read_csv("data/human-freedom-index-2022.csv")
    # Culture Fit
    countries_dict = country_data.get_country_dict()
    # English speaking
    df_english = pd.read_csv("data/english_speaking.csv")

    return human_freedom_df, countries_dict, df_english


@st.cache_data
def load_country_to_emoji():
    # Country codes alpha3
    df_codes_alpha_3 = pd.read_csv("data/country_codes_alpha_3.csv")
    # Flag emoji
    df_flag_emoji = pd.read_csv("data/country_flag_emoji.csv")

    df_country_to_emoji = df_codes_alpha_3.merge(df_flag_emoji, on="country_code_alpha_3")
    return df_country_to_emoji[["country", "emoji"]].set_index("country").to_dict()["emoji"]


# Load data
human_freedom_df, countries_dict, df_english = load_data()
COUNTRY_TO_EMOJI = load_country_to_emoji()

# Basic derived data
all_countries = list(countries_dict.values())


def culture_fit_reference_callback():
    if state.culture_fit_reference_country == NONE_COUNTRY:
        return

    country_info = countries_dict[state.culture_fit_reference_country]

    for dimension in dimensions_info.DIMENSIONS:
        state[dimension] = getattr(country_info, dimension)


def options_callback():
    return


personal_freedom_score_help = "Personal Freedom measures the degree to which members of a country are free to exercise civil liberties. This includes freedom of movement, freedom of religion, freedom of assembly and political action, freedom of the press and information, and freedom to engage in various interpersonal relationships. This also includes the rule of law, security, and safety, which are necessary for meaningful exercise of personal freedoms. The Personal Freedom Index (score) is provided by the Cato Institue and Fraser Institute. See https://www.cato.org/human-freedom-index/2022 for more details."
economic_freedom_score_help = "Economic Freedom measures the degree to which members of a country are free to exercise financial liberties. This includes the freedom to trade, the freedom to use sound money. This also includes the size of government, legal system and property rights, and market regulation, which are necessary for meaningful exercise of economic freedoms. The Economic Freedom Index (score) is provided by the Cato Institue and Fraser Institute. See https://www.cato.org/human-freedom-index/2022 for more details."


def get_options_from_query_params(app_options=None):
    if app_options is None:
        app_options = AppOptions()

    query_params = st.experimental_get_query_params()

    for field in dataclasses.fields(app_options):
        if field.name in query_params:
            query_param_val = query_params[field.name]

            # This takes care of extracting singleton lists (required due to experimental_get_query_params implementation)
            if len(query_param_val) == 1:
                query_param_val = query_param_val[0]

            # This takes care of converting from string to the proper data type for the field
            query_param_val = field.type(query_param_val)
            setattr(app_options, field.name, query_param_val)

    return app_options


def get_options_from_ui(app_options=None):
    if app_options is None:
        app_options = AppOptions()

    with st.expander("Culture Fit preferences"):
        st.caption(
            "",
            help="Preferences regarding your desired national culture. See https://geerthofstede.com/culture-geert-hofstede-gert-jan-hofstede/6d-model-of-national-culture/ for more information.",
        )

        for dimension in dimensions_info.DIMENSIONS:
            dimension_info = dimensions_info.DIMENSIONS_INFO[dimension]

            slider_str = f'{dimension_info["name"]} ({dimension_info["abbreviation"]})'
            help_str = f'{dimension_info["name"]} ({dimension_info["abbreviation"]}): {dimension_info["question"]} \n\n {dimension_info["description"]}'
            default_value = getattr(app_options, f"culture_fit_preference_{dimension}")

            preference_val = st.slider(
                slider_str, min_value=0, max_value=100, value=default_value, help=help_str, key=dimension
            )

            setattr(app_options, f"culture_fit_preference_{dimension}", preference_val)

    with st.expander("Filters"):
        app_options.year_min, app_options.year_max = st.slider(
            "Year Range",
            min_value=2000,
            max_value=2020,
            value=(app_options.year_min, app_options.year_max),
            help="Years over which to aggregate statistics. Only affects Human Freedom scores.",
        )
        app_options.cf_score_min = st.slider(
            "Culture Fit Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.cf_score_min,
            key="cf_score_min",
        )
        app_options.pf_score_min = st.slider(
            "Personal Freedom Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.pf_score_min,
            help=personal_freedom_score_help,
            key="pf_score_min",
        )
        app_options.ef_score_min = st.slider(
            "Economic Freedom Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.ef_score_min,
            help=economic_freedom_score_help,
            key="ef_score_min",
        )
        app_options.english_ratio_min = st.slider(
            "English Speaking Ratio Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.english_ratio_min,
            key="english_ratio_min",
        )

    with st.expander("Weights"):
        app_options.cf_score_weight = st.slider(
            "Culture Fit Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.cf_score_weight,
        )
        app_options.pf_score_weight = st.slider(
            "Personal Freedom Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.pf_score_weight,
            help=personal_freedom_score_help,
        )
        app_options.ef_score_weight = st.slider(
            "Economic Freedom Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.ef_score_weight,
            help=economic_freedom_score_help,
        )

    with st.expander("Results options"):
        app_options.N = st.number_input(
            "Number of Top Matching countries to show",
            min_value=1,
            max_value=20,
            value=app_options.N,
        )

        app_options.show_radar = st.checkbox(
            label="Show radar plots",
            value=app_options.show_radar,
            help="Not showing radar plots can significantly improve rendering time.",
        )

        field_for_world_map_options = ["overall_score", "cf_score", "pf_score", "ef_score", "english_ratio"]
        field_for_world_map_default_index = field_for_world_map_options.index(app_options.field_for_world_map)
        app_options.field_for_world_map = st.selectbox(
            label="Field to plot in the World Map",
            options=field_for_world_map_options,
            index=field_for_world_map_default_index,
        )

        app_options.world_map_projection_type = st.selectbox(
            label="World Map projection type",
            options=PLOTLY_MAP_PROJECTION_TYPES,
            index=PLOTLY_MAP_PROJECTION_TYPES.index(app_options.world_map_projection_type),
        )

    return app_options


# Options

# Only pull the query_params on the first run e.g. to support deeplinking.
# Otherwise, only use the options that have been set in the current session.
# This helps avoid a race condition between getting options via query_params and getting options via the UI.
if not "query_params_pulled_down" in state:
    state.query_params_pulled_down = True
    app_options = get_options_from_query_params()
else:
    app_options = AppOptions()

with st.sidebar:
    with st.form(key="reference_country_form"):
        culture_fit_reference_country_options = [NONE_COUNTRY] + sorted(list(countries_dict))
        culture_fit_reference_country = st.selectbox(
            "Reference Country",
            options=culture_fit_reference_country_options,
            key="culture_fit_reference_country",
        )
        culture_fit_reference_country_submit_button = st.form_submit_button(
            label="Set Culture Fit preferences to selected reference country", on_click=culture_fit_reference_callback
        )

    with st.form(key="options_form"):
        app_options = get_options_from_ui(app_options)
        submit_button = st.form_submit_button(label="Update Options", on_click=options_callback)

# Set the query params with all the app_options
st.experimental_set_query_params(**dataclasses.asdict(app_options))


def render_ui_section_title():
    st.title("🌎 :blue[Terra]", anchor=False)
    terra_question = 'This app is designed to answer the question "which country should I live in?"'
    terra_explanation = "Use data to decide which country is right for you. Terra will take your personal preferences regarding Culture Fit, Human Freedom, and Language into account and recommend one or more countries that match."
    terra_caveats = "Caveats and limitations: This app integrates several data sources, which do not have complete information for every country. Therefore, some countries will be excluded from the analysis. Please contact the app author if you have more complete data to share."
    terra_help = f"{terra_question}\n\n{terra_explanation}\n\n{terra_caveats}"
    st.caption("Find the right country for you!", help=terra_help)


def main():
    render_ui_section_title()

    # Options validation
    if not app_options.year_min <= app_options.year_max:
        st.error(
            f"Invalid time range selected ({app_options.year_min} > {app_options.year_max}), please correct and resubmit options."
        )
        return

    # Human Freedom
    human_freedom_df_year_filtered = human_freedom_df.query(f"{app_options.year_min} <= year <= {app_options.year_max}")

    df = human_freedom_df_year_filtered.groupby(["country"])[["pf_score", "ef_score"]].mean()
    df[["pf_score", "ef_score"]] = df[["pf_score", "ef_score"]] * 0.1  # undo 10X scaling

    df = df.reset_index()

    # Cultural Fit
    # Derived from https://culture-map.streamlit.app/Country_Match?ref=blog.streamlit.io
    user_ideal = country_data.types.CountryInfo(
        id=999,
        title="user",
        slug="user",
        pdi=app_options.culture_fit_preference_pdi,
        idv=app_options.culture_fit_preference_idv,
        mas=app_options.culture_fit_preference_mas,
        uai=app_options.culture_fit_preference_uai,
        lto=app_options.culture_fit_preference_lto,
        ind=app_options.culture_fit_preference_ind,
        ivr=app_options.culture_fit_preference_ind,  # not a typo
        adjective="user",
    )

    distances, max_distance = distance_calculations.compute_distances(
        countries_from=[user_ideal], countries_to=all_countries, distance_metric="Cosine"
    )
    distances = distances.sort_values("user")
    culture_fit_score = (1 - distances).reset_index()
    culture_fit_score = culture_fit_score.rename(columns={"index": "country", "user": "cf_score"})

    df = df.merge(culture_fit_score, on="country")

    # Language
    # From https://en.wikipedia.org/wiki/List_of_countries_by_English-speaking_population
    # Percentage of people who speak English as a mother tongue or foreign language

    if app_options.do_filter_english:
        df = df.merge(df_english[["country", "english_ratio"]], on="country")

    # Overall Score
    weight_sum = app_options.pf_score_weight + app_options.ef_score_weight + app_options.cf_score_weight
    app_options.pf_score_weight /= weight_sum
    app_options.ef_score_weight /= weight_sum
    app_options.cf_score_weight /= weight_sum

    df["pf_score_weighted"] = df["pf_score"] * app_options.pf_score_weight
    df["ef_score_weighted"] = df["ef_score"] * app_options.ef_score_weight
    df["cf_score_weighted"] = df["cf_score"] * app_options.cf_score_weight

    df["overall_score"] = df["pf_score_weighted"] + df["ef_score_weighted"] + df["cf_score_weighted"]
    df = df.sort_values("overall_score", ascending=False)

    # Filters
    df["acceptable"] = True
    if app_options.do_filter_culture_fit:
        df["acceptable"] = df["acceptable"] & (df["cf_score"] > app_options.cf_score_min)
    if app_options.do_filter_freedom:
        df["acceptable"] = (
            df["acceptable"] & (df["pf_score"] > app_options.pf_score_min) & (df["ef_score"] > app_options.ef_score_min)
        )
    if app_options.do_filter_english:
        df["acceptable"] = df["acceptable"] & (df["english_ratio"] > app_options.english_ratio_min)

    df = df[df["acceptable"]]

    if df.shape[0] == 0:
        st.warning("No matches found! Try adjusting the filters to be less strict.")
    else:
        # Results

        # Best match
        best_match_country = str(df.iloc[0].country)
        best_match_country_emoji = COUNTRY_TO_EMOJI[best_match_country]

        st.header(f"Your Best Match Country: :blue[{best_match_country}] ({best_match_country_emoji})", anchor=False)
        with st.expander("Flag", expanded=True):
            st.image(visualisation.country_urls.COUNTRY_URLS[best_match_country], width=100)

        best_match_country_slug = best_match_country.lower().replace(" ", "-")
        cia_world_factbook_url = f"https://www.cia.gov/the-world-factbook/countries/{best_match_country_slug}/"
        with st.expander(f"CIA World Factbook", expanded=True):
            st.markdown(f"([open in new tab]({cia_world_factbook_url}))")
            st.components.v1.iframe(cia_world_factbook_url, height=600, scrolling=True)

        # Top N best matches
        st.header(f"Top Matching Countries ({app_options.N})", anchor=False)
        column_remap = {
            "overall_score": "Overall Score",
            "pf_score": "Personal Freedom Score",
            "ef_score": "Economic Freedom Score",
            "cf_score": "Cultural Fit Score",
            "pf_score_weighted": "Personal Freedom Score (weighted)",
            "ef_score_weighted": "Economic Freedom Score (weighted)",
            "cf_score_weighted": "Cultural Fit Score (weighted)",
        }
        df_top_N = df.head(app_options.N).rename(columns=column_remap)

        def pct_fmt(x):
            return f"{round(100*x, 2):.0f}%"

        with st.expander("Score Contributions", expanded=True):
            fig = px.bar(
                df_top_N,
                x="country",
                y=[
                    "Personal Freedom Score (weighted)",
                    "Economic Freedom Score (weighted)",
                    "Cultural Fit Score (weighted)",
                ],
            )
            for idx, row in df_top_N.iterrows():
                fig.add_annotation(
                    x=row.country,
                    y=row["Overall Score"],
                    yanchor="bottom",
                    showarrow=False,
                    align="left",
                    text=f"{pct_fmt(row['Overall Score'])}",
                    font={"size": 12},
                )
            fig.update_layout(legend=dict(orientation="v", yanchor="top", y=-0.2, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True)

        # TODO replace pyplot radar plots with plotly radar plots
        # See https://plotly.com/python/radar-chart/
        def get_radar(country_names, user_ideal):
            dimensions = distance_calculations.compute_dimensions(
                [countries_dict[country_name] for country_name in country_names]
            )
            reference = distance_calculations.compute_dimensions([user_ideal])
            radar = visualisation.generate_radar_plot(dimensions, reference)
            return radar

        if app_options.show_radar:
            with st.expander("Culture Fit"):
                radar = get_radar(country_names=df_top_N.country, user_ideal=user_ideal)
                st.caption("", help="The dashed :red[red shape] depicts your preferences.")
                st.pyplot(radar)

        # All matches
        st.header(f"All Matching Countries ({df.shape[0]})", anchor=False)

        def generate_choropleth(df, name):
            df = df.reset_index()
            fig = px.choropleth(
                df,
                locationmode="country names",
                locations="country",
                color=name,
                hover_name="country",
                color_continuous_scale=px.colors.sequential.deep_r,
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            return fig

        with st.expander("World Map", expanded=True):
            fig = generate_choropleth(df, app_options.field_for_world_map)
            fig.update_geos(projection_type=app_options.world_map_projection_type)
            fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
            fig.update_layout(geo_bgcolor="#0E1117")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw Results Data"):
            st.dataframe(df.set_index("country"), use_container_width=True)


main()
