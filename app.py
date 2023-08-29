import dataclasses
from copy import deepcopy
from math import floor
from functools import partial
from urllib.parse import urlencode

import numpy as np
from sklearn.decomposition import (
    PCA,
    FastICA,
    NMF,
    MiniBatchSparsePCA,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
from scipy.spatial import distance
from scipy.cluster import hierarchy
import pandas as pd
import streamlit as st
from streamlit_globe import streamlit_globe
from streamlit_extras.metric_cards import style_metric_cards

import plotly.express as px
import plotly.figure_factory as ff
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from supported_countries import SUPPORTED_COUNTRIES
import config
from app_options import AppOptions, NONE_COUNTRY, TTL, CONTINENT_OPTIONS
import clustering_options
import map_options
import color_options
import utils
import world_factbook_utils
from culture_fit import country_data, distance_calculations, dimensions_info

# Streamlit setup
st.set_page_config(page_title="Terra", page_icon="🌎", layout="wide")

# Custom CSS ("css hack") to set the background color of all iframe components to white.
# This is needed because most websites assume a white background and will be illegible
# when iframed without this when a dark background is used.
st.markdown(
    """
<style>iframe {background-color: white;}</style>
""",
    unsafe_allow_html=True,
)

# Style metrics
style_metric_cards(
    border_left_color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
    border_color=config.STREAMLIT_CONFIG["theme"]["secondaryBackgroundColor"],
    background_color=config.STREAMLIT_CONFIG["theme"]["backgroundColor"],
    border_size_px=2,
    border_radius_px=20,
    box_shadow=False,
)


# Pyplot setup
plt.style.use(["dark_background", "./terra.mplstyle"])

# Short convenience alias
state = st.session_state


def flexecute(
    func,
    label="",
    expanded=False,
    func_args=None,
    func_kwargs=None,
    checkbox_value=False,
    conditional=True,
    header=False,
    subheader=False,
):
    """Execute func in streamlit expander.

    Includes ability to conditionally execute func if an st.checkbox is checked.
    """

    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    if conditional:
        checkbox_label = f"Show {label}"
        spinner_label = f"Executing {label}"
        not_shown_str = f'Enable "{checkbox_label}" to populate this section. Note this may increase render time.'

    with st.expander(label, expanded=expanded):
        if header:
            st.header(label, anchor=False)
        if subheader:
            st.subheader(label, anchor=False)

        if conditional:
            if st.checkbox(checkbox_label, value=checkbox_value):
                with st.spinner(spinner_label):
                    func(*func_args, **func_kwargs)
            else:
                # TODO make slow verbiage optional in args to this function
                st.info(not_shown_str)
        else:
            func(*func_args, **func_kwargs)


@st.cache_resource(ttl=TTL)
def load_data():
    """Load all data sources."""

    # Culture Fit
    culture_fit_data_dict = country_data.get_country_dict()
    # Happy Planet
    happy_planet_df = pd.read_csv("./data/happy_planet_index_2019.csv")
    # Social Progress
    social_progress_df = pd.read_csv("./data/social_progress_index_2022.csv")
    # Human Freedom
    human_freedom_df = pd.read_csv("./data/human-freedom-index-2022.csv")
    # English speaking
    df_english = pd.read_csv("./data/english_speaking.csv")
    # Temperature
    df_temperature = pd.read_csv("./data/country_temperature.csv")
    # Sunshine
    df_sunshine = pd.read_csv("./data/country_sunshine_hours_per_day.csv")
    df_sunshine = df_sunshine.rename(columns={"year": "average_sunshine_hours_per_day"})
    df_sunshine = df_sunshine[["country", "average_sunshine_hours_per_day"]]
    # Coordinates
    df_coords = pd.read_csv("./data/country_coords.csv")
    df_coords = df_coords.set_index("country")
    # Continents
    df_continents = pd.read_csv("./data/country_continents.csv")
    # Country codes alpha3
    df_codes_alpha_3 = pd.read_csv("./data/country_codes_alpha_3.csv")
    # Flag emoji
    df_flag_emoji = pd.read_csv("./data/country_flag_emoji.csv")

    # PREPROCESSING

    # Culture Fit
    # Remove countries that do not have all dimensions populated
    country_names_to_remove = set()
    for country_name, country_info in culture_fit_data_dict.items():
        for dimension in dimensions_info.DIMENSIONS:
            val = getattr(country_info, dimension)
            if val is None or val < 0:
                country_names_to_remove.add(country_name)

    for country_name in country_names_to_remove:
        culture_fit_data_dict.pop(country_name)

    culture_fit_df = pd.DataFrame.from_dict(culture_fit_data_dict, orient="index")[dimensions_info.DIMENSIONS]
    culture_fit_df *= 0.01  # Undo 100X scaling
    culture_fit_df = culture_fit_df.rename_axis("country").reset_index()  # Move country to column

    # Happy Planet
    happy_planet_df = happy_planet_df[["country", "HPI"]]
    happy_planet_df = happy_planet_df.rename(columns={"HPI": "hp_score"})
    happy_planet_df["hp_score"] *= 0.01  # Undo 100X scaling
    happy_planet_df["hp_score"] /= happy_planet_df["hp_score"].max()  # Normalize by max value achieved in the dataset

    # Social Progress
    # Pick out just the columns we need and rename country column
    social_progress_cols_keep = ["country", "Social Progress Score"]
    social_progress_df = social_progress_df[social_progress_cols_keep]
    social_progress_df = social_progress_df.set_index("country")
    social_progress_df *= 0.01  # Undo 100X scaling
    social_progress_df = social_progress_df.reset_index()
    social_progress_df = social_progress_df.rename(columns={"Social Progress Score": "sp_score"})
    social_progress_df["sp_score"] /= social_progress_df[
        "sp_score"
    ].max()  # Normalize by max value achieved in the dataset

    # Human Freedom
    human_freedom_df["hf_score"] /= human_freedom_df["hf_score"].max()  # Normalize by max value achieved in the dataset

    year_min, year_max = 2015, 2020
    human_freedom_df_year_filtered = human_freedom_df.query(f"{year_min} <= year <= {year_max}")
    freedom_score_cols = ["hf_score"]
    human_freedom_df = human_freedom_df_year_filtered.groupby(["country"])[freedom_score_cols].mean()
    human_freedom_df = human_freedom_df.reset_index()

    # Country emoji
    df_country_to_emoji = df_codes_alpha_3.merge(df_flag_emoji, on="country_code_alpha_3")
    country_to_emoji = df_country_to_emoji[["country", "emoji"]].set_index("country").to_dict()["emoji"]

    return (
        culture_fit_data_dict,
        culture_fit_df,
        happy_planet_df,
        social_progress_df,
        human_freedom_df,
        df_english,
        df_temperature,
        df_sunshine,
        df_coords,
        df_continents,
        df_codes_alpha_3,
        df_flag_emoji,
        country_to_emoji,
    )


# Load data and expose in the outermost scope of this script
(
    culture_fit_data_dict,
    culture_fit_df,
    happy_planet_df,
    social_progress_df,
    human_freedom_df,
    df_english,
    df_temperature,
    df_sunshine,
    df_coords,
    df_continents,
    df_codes_alpha_3,
    df_flag_emoji,
    country_to_emoji,
) = load_data()


def country_to_emoji_func(country):
    if country not in country_to_emoji:
        return country
    return f"{country} ({country_to_emoji[country]})"


@st.cache_resource(ttl=TTL)
def get_main_data():
    """Combine the main data for Quality-of-Life and Culture Fit and Climate into a single dataframe."""
    df = pd.DataFrame({"country": SUPPORTED_COUNTRIES})
    df = df.merge(happy_planet_df, on="country")
    df = df.merge(social_progress_df, on="country")
    df = df.merge(human_freedom_df, on="country")
    df = df.merge(culture_fit_df, on="country")
    df = df.merge(df_temperature, on="country")
    df = df.merge(df_sunshine, on="country")
    df = df.merge(df_continents, on="country")
    df["country_with_emoji"] = df["country"].apply(country_to_emoji_func)
    return df


mdf = get_main_data()


# TODO move to a config file
df_format_dict = {
    "country": "Country",
    "country_with_emoji": "Country with Emoji",
    "overall_score": "Overall Score",
    "cf_score": "Culture Fit Score",
    "ql_score": "Quality-of-Life Score",
    "hp_score": "Happy Planet Score",
    "sp_score": "Social Progress Score",
    "hf_score": "Human Freedom Score",
    "overall_score_rank": "Overall Score Rank",
    "cf_score_rank": "Culture Fit Score Rank",
    "ql_score_rank": "Quality-of-Life Score Rank",
    "hp_score_rank": "Happy Planet Score Rank",
    "sp_score_rank": "Social Progress Score Rank",
    "hf_score_rank": "Human Freedom Score Rank",
    "cf_score_weighted": "Culture Fit Score (weighted)",
    "ql_score_weighted": "Quality-of-Life Score (weighted)",
    "hp_score_weighted": "Happy Planet Score (weighted)",
    "sp_score_weighted": "Social Progress Score (weighted)",
    "hf_score_weighted": "Human Freedom Score (weighted)",
    "english_ratio": "English Speaking Ratio",
    "average_temperature_celsius": "Average Temperature (°C)",
    "average_sunshine_hours_per_day": "Average Daily Hours of Sunshine",
    "continent": "Continent",
    "satisfies_filters": "Satisfies Filters",
}
for dimension in dimensions_info.DIMENSIONS:
    df_format_dict[dimension] = dimensions_info.DIMENSIONS_INFO[dimension]["name"]
    df_format_dict[f"{dimension}_rank"] = f"{df_format_dict[dimension]} Rank"


def df_format_func(key):
    return df_format_dict[key]


# TODO move to a config file
app_options_codes = ["cf", "ql", "hp", "sp", "hf"]

# TODO move to config file
overall_fields = [
    "overall_score",
    "cf_score",
    "ql_score",
]
quality_of_life_fields = [
    "hp_score",
    "sp_score",
    "hf_score",
]
culture_fields = dimensions_info.DIMENSIONS
climate_fields = ["average_temperature_celsius", "average_sunshine_hours_per_day"]
geography_fields = ["continent"]


def culture_fit_reference_callback():
    if state.reference_country == NONE_COUNTRY:
        return

    country_info = culture_fit_data_dict[state.reference_country]

    for dimension in dimensions_info.DIMENSIONS:
        state[dimension] = getattr(country_info, dimension)

    st.toast(f"Culture Fit Preferences set to Reference Country :blue[**{state.reference_country}**]", icon="🎛️")


def quality_of_life_reference_callback():
    if state.reference_country == NONE_COUNTRY:
        return

    fields = ["hp_score_min", "sp_score_min", "hf_score_min"]
    dfs = [happy_planet_df, social_progress_df, human_freedom_df]
    for field, df in zip(fields, dfs):
        state[field] = floor(df.set_index("country").loc[state.reference_country].item() * 100) / 100

    st.toast(f"Quality-of-Life Filters set to Reference Country :blue[**{state.reference_country}**]", icon="🎛️")


# TODO move this to a data file
culture_fit_score_help = (
    "Culture Fit Score measures how closely a national culture matches your preferences, as determined by [average"
    " cityblock similarity](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html)"
    " of the culture dimension vectors of the nation and your ideal."
)

quality_of_life_score_help = (
    "Quality-of-Life Score measures how high the quality of life is expected to be based on your preferences."
)

happy_planet_score_help = (
    "The [Happy Planet Index](https://happyplanetindex.org/learn-about-the-happy-planet-index/) mesaures sustainable"
    " wellbeing, ranking countries by how efficiently they deliver long, happy lives using our limited environmental"
    " resources."
)

social_progress_score_help = (
    "The [Social Progress Index](https://www.socialprogress.org/global-index-2022overview/) measures the capacity of a"
    " society to meet the basic human needs of its citizens, establish the building blocks that allow citizens and"
    " communities to enhance and sustain the quality of their lives, and create the conditions for all individuals to"
    " reach their full potential."
)

human_freedom_score_help = (
    "The [Human Freedom Index](https://www.cato.org/human-freedom-index/2022) measures the state of human freedom in"
    " the world based on a broad measure that encompasses personal, civil, and economic freedom."
)


english_speaking_ratio_help = (
    "Ratio of people who speak English as a mother tongue or foreign language to the total population."
)

average_temperature_help = (
    "[Average yearly temperature](https://en.wikipedia.org/wiki/List_of_countries_by_average_yearly_temperature),"
    " calculated by averaging the minimum and maximum daily temperatures in the country, averaged for the years"
    " 1961-1990, based on gridded climatologies from the Climatic Research Unit elaborated in 2011."
)

average_sunshine_help = (
    "[Average daily hours of sunshine](https://en.wikipedia.org/wiki/List_of_cities_by_sunshine_duration) are the"
    " number of hours of sunshine per day, averaged over the entire year for one or more years, with the median taken"
    " over one or more cities in a country."
)

continent_help = (
    "[Continent in which the country is located](https://simple.wikipedia.org/wiki/List_of_countries_by_continents)."
)

dimensionality_reducer_name_to_class_map = {
    "PCA": PCA,
    "FastICA": FastICA,
    "NMF": NMF,
    "MiniBatchSparsePCA": MiniBatchSparsePCA,
    "SparsePCA": SparsePCA,
    "TruncatedSVD": TruncatedSVD,
    "t-SNE": TSNE,
    "UMAP": UMAP,
}

unit_interval_1pt_range_list = np.arange(0.0, 1.0 + 0.01, 0.01).round(2).tolist()
unit_interval_5pt_range_list = np.arange(0.0, 1.0 + 0.05, 0.05).round(2).tolist()


@st.cache_data(ttl=TTL)
def get_dimension_reduced_df(df, dimensionality_reducer_name, dimensionality_reducer_kwargs):
    dimensionality_reducer_class = dimensionality_reducer_name_to_class_map[dimensionality_reducer_name]
    dimensionality_reducer = dimensionality_reducer_class(**dimensionality_reducer_kwargs)

    projection = dimensionality_reducer.fit_transform(df)
    df_projection = pd.DataFrame(projection).rename(columns={0: "x", 1: "y"})
    df_projection.index = df.index
    return df_projection


def get_options_from_query_params():
    query_params = st.experimental_get_query_params()
    app_options = AppOptions()
    for field in dataclasses.fields(app_options):
        if field.name in query_params:
            query_param_val = query_params[field.name]

            # This takes care of extracting singleton lists
            # (required due to experimental_get_query_params implementation)
            if len(query_param_val) == 1:
                query_param_val = query_param_val[0]

            # This takes care of converting from string to the proper data type for the field
            query_param_val = field.type(query_param_val)

            # Finally overwrite the field in app_options with the query_param_val
            setattr(app_options, field.name, query_param_val)

    return app_options


def get_options_from_ui():
    app_options = AppOptions()

    def score_strip_plot(df, field, xmin, xmax):
        # Add a little buffer to the plot limits to catch endpoints
        dx = xmax - xmin
        xmin_, xmax_ = xmin - (dx / 100), xmax + (dx / 100)
        fig = px.strip(
            df,
            x=field,
            labels={dimension: df_format_func(dimension)},
            hover_name="country_with_emoji",
            orientation="h",
            range_x=(xmin_, xmax_),
            height=200,
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    def culture_fit_preferences_func():
        mdf100 = mdf.copy()
        mdf100[dimensions_info.DIMENSIONS] *= 100

        for dimension in dimensions_info.DIMENSIONS:
            dimension_info = dimensions_info.DIMENSIONS_INFO[dimension]

            slider_str = f'**{dimension_info["name"]} ({dimension_info["abbreviation"]})**'
            caption_str = f'*{dimension_info["question"]}*'
            help_str = f'{dimension_info["description"]}'

            st.write(slider_str)
            st.caption(caption_str, help=help_str)
            preference_val = st.slider(
                slider_str,
                min_value=0,
                max_value=100,
                step=5,
                key=dimension,
                label_visibility="collapsed",
            )
            score_strip_plot(mdf100, dimension, 0, 100)

            setattr(app_options, f"culture_fit_preference_{dimension}", preference_val)

    def quality_of_life_preferences_func():
        # TODO construct these programmatically
        st.subheader(
            ":memo: :orange[*Make sure the weights add up to 100*]",
            anchor=False,
            help=(
                "*Why do the weights have to add up to 100?* These are competing interests, so we need to know how"
                " important each one is relative to the others."
            ),
        )

        slider_str = "Happy Planet Score Weight"
        caption_str = (
            "*How much do you value living in a country that delivers long, happy lives without damaging the"
            " environment?* (🔵 Progressive bias)"
        )
        st.write(slider_str)
        st.caption(caption_str, help=happy_planet_score_help)
        app_options.hp_score_weight = st.select_slider(
            slider_str,
            options=unit_interval_5pt_range_list,
            key="hp_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )

        slider_str = "Social Progress Score Weight"
        caption_str = (
            "*How much do you value living in a country that meets basic human needs, lays the foundations of"
            " wellbeing, and creates opportunities for all individuals to reach their full potential?* (🟣 Center bias)"
        )
        st.write(slider_str)
        st.caption(caption_str, help=social_progress_score_help)
        app_options.sp_score_weight = st.select_slider(
            slider_str,
            options=unit_interval_5pt_range_list,
            key="sp_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )

        slider_str = "Human Freedom Score Weight"
        caption_str = (
            "*How much do you value living in a country with a broad range of personal, civil, and economic freedoms?*"
            " (🔴 Libertarian-conservative bias)"
        )
        st.write(slider_str)
        st.caption(caption_str, help=human_freedom_score_help)
        app_options.hf_score_weight = st.select_slider(
            slider_str,
            options=unit_interval_5pt_range_list,
            key="hf_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )

    def overall_preferences_func():
        st.subheader(
            ":memo: :orange[*Make sure the weights add up to 100*]",
            anchor=False,
            help=(
                "*Why do the weights have to add up to 100?* These are competing interests, so we need to know how"
                " important each one is relative to the others."
            ),
        )

        slider_str = "Culture Fit Score Weight"
        caption_str = "*How much do you value living in a country whose culture aligns with your preferences?*"
        st.write(slider_str)
        st.caption(caption_str, help=culture_fit_score_help)
        app_options.cf_score_weight = st.select_slider(
            slider_str,
            options=unit_interval_5pt_range_list,
            key="cf_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )

        slider_str = "Quality-of-Life Score Weight"
        caption_str = "*How much do you value living in a country with high quality-of-life?*"
        st.write(slider_str)
        st.caption(caption_str, help=quality_of_life_score_help)
        app_options.ql_score_weight = st.select_slider(
            slider_str,
            options=unit_interval_5pt_range_list,
            key="ql_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )

    def quality_of_life_filters_func():
        # TODO construct these programmatically

        slider_str = "Happy Planet Score Min"
        caption_str = "*What is the minimum Happy Planet Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=happy_planet_score_help)
        app_options.hp_score_min = st.select_slider(
            slider_str,
            options=unit_interval_1pt_range_list,
            key="hp_score_min",
            label_visibility="collapsed",
        )
        score_strip_plot(mdf, "hp_score", 0.0, 1.0)

        slider_str = "Social Progress Score Min"
        caption_str = "*What is the minimum Social Progress Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=social_progress_score_help)
        app_options.sp_score_min = st.select_slider(
            slider_str,
            options=unit_interval_1pt_range_list,
            key="sp_score_min",
            label_visibility="collapsed",
        )
        score_strip_plot(mdf, "sp_score", 0.0, 1.0)

        slider_str = "Human Freedom Score Min"
        caption_str = "*What is the minimum Human Freedom Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=human_freedom_score_help)
        app_options.hf_score_min = st.select_slider(
            slider_str,
            options=unit_interval_1pt_range_list,
            key="hf_score_min",
            label_visibility="collapsed",
        )
        score_strip_plot(mdf, "hf_score", 0.0, 1.0)

    def overall_filters_func():
        slider_str = "Culture Fit Score Min"
        caption_str = "*What is the minimum Culture Fit Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=culture_fit_score_help)
        app_options.cf_score_min = st.select_slider(
            slider_str,
            options=unit_interval_1pt_range_list,
            key="cf_score_min",
            label_visibility="collapsed",
        )

        slider_str = "Quality-of-Life Score Min"
        caption_str = "*What is the minimum Quality-of-Life Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=quality_of_life_score_help)
        app_options.ql_score_min = st.select_slider(
            slider_str,
            options=unit_interval_1pt_range_list,
            key="ql_score_min",
            label_visibility="collapsed",
        )

    def language_filters_func():
        slider_str = ":speaking_head_in_silhouette: English Speaking Ratio Min"
        caption_str = "*What is the minimum proportion of English speakers you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=english_speaking_ratio_help)
        app_options.english_ratio_min = st.select_slider(
            slider_str,
            options=unit_interval_1pt_range_list,
            key="english_ratio_min",
            label_visibility="collapsed",
        )

    def climate_filters_func():
        slider_str = ":thermometer: Average Temperature (°C) Range"
        caption_str = "*What is the range of average temperature (°C) you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=average_temperature_help)
        app_options.average_temperature_celsius_min, app_options.average_temperature_celsius_max = st.slider(
            slider_str,
            -10,
            30,
            (-10, 30),
            format="%d°C",
            key="average_temperature_celsius_range",
            label_visibility="collapsed",
        )
        score_strip_plot(mdf, "average_temperature_celsius", -10, 30)

        slider_str = ":sunny: Average Daily Hours of Sunshine Range"
        caption_str = "*What is the range of average daily hours of sunshine you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=average_sunshine_help)
        app_options.average_sunshine_hours_per_day_min, app_options.average_sunshine_hours_per_day_max = st.slider(
            slider_str,
            3.0,
            10.0,
            (3.0, 10.0),
            step=0.1,
            format="%.1f hours/day",
            key="average_sunshine_hours_per_day_range",
            label_visibility="collapsed",
        )
        score_strip_plot(mdf, "average_sunshine_hours_per_day", 3.0, 10.0)

    def geography_filters_func():
        slider_str = "🌎🌍🌏 Continents"
        caption_str = "*What continents do you want to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=continent_help)
        app_options.continents = st.multiselect(
            slider_str,
            options=CONTINENT_OPTIONS,
            key="continents",
            label_visibility="collapsed",
        )

    st.title("Options", anchor=False)
    flexecute_kwargs = dict(expanded=False, conditional=False, subheader=True)
    st.header("Preferences", anchor=False)
    flexecute(
        func=culture_fit_preferences_func,
        label="Culture Fit Preferences",
        **flexecute_kwargs,
    )
    flexecute(
        func=quality_of_life_preferences_func,
        label="Quality-of-Life Preferences",
        **flexecute_kwargs,
    )
    flexecute(func=overall_preferences_func, label="Overall Preferences", **flexecute_kwargs)
    st.header("Filters", anchor=False)
    flexecute(
        func=quality_of_life_filters_func,
        label="Quality-of-Life Filters",
        **flexecute_kwargs,
    )
    flexecute(func=overall_filters_func, label="Overall Score Filters", **flexecute_kwargs)
    flexecute(func=language_filters_func, label="Language Filters", **flexecute_kwargs)
    flexecute(func=climate_filters_func, label="Climate Filters", **flexecute_kwargs)
    flexecute(func=geography_filters_func, label="Geography Filters", **flexecute_kwargs)

    return app_options


def initialize_widget_state_from_app_options(app_options):
    # Effectively set the first-time default for certain widgets by initializing
    # a value assigned to its key in session state before the widgets are
    # instantiated for the first time.
    # See https://discuss.streamlit.io/t/why-do-default-values-cause-a-session-state-warning/15485/27
    for dimension in dimensions_info.DIMENSIONS:
        state[dimension] = getattr(app_options, f"culture_fit_preference_{dimension}")

    # TODO move list to config
    for code in app_options_codes:
        weight_field = f"{code}_score_weight"
        min_field = f"{code}_score_min"
        state[weight_field] = getattr(app_options, weight_field)
        state[min_field] = getattr(app_options, min_field)

    state["english_ratio_min"] = getattr(app_options, "english_ratio_min")
    # Special handling for (min, max) range params
    state["average_temperature_celsius_range"] = (
        getattr(app_options, "average_temperature_celsius_min"),
        getattr(app_options, "average_temperature_celsius_max"),
    )
    state["average_sunshine_hours_per_day_range"] = (
        getattr(app_options, "average_sunshine_hours_per_day_min"),
        getattr(app_options, "average_sunshine_hours_per_day_max"),
    )
    state["continents"] = getattr(app_options, "continents")


def first_run_per_session():
    # Only pull the query_params on the first run e.g. to support deeplinking.
    # Otherwise, only use the options that have been set in the session.
    # This helps avoid a race condition between getting options via query_params and getting options via the UI.
    state.app_options = get_options_from_query_params()

    initialize_widget_state_from_app_options(state.app_options)

    state.initialized = True


def reset_options_callback():
    app_options = AppOptions()
    initialize_widget_state_from_app_options(app_options)


def update_options_callback():
    st.toast("Updated Options", icon="🎛️")


def get_options():
    with st.form(key="options_form"):
        app_options = get_options_from_ui()
        st.form_submit_button(
            label="Update Options", type="primary", use_container_width=True, on_click=update_options_callback
        )

    st.header("Option Modifiers", anchor=False)

    # Reference Country
    reference_country_options = [NONE_COUNTRY] + sorted(list(culture_fit_data_dict))
    st.selectbox(
        "Reference Country",
        options=reference_country_options,
        format_func=country_to_emoji_func,
        key="reference_country",
        help="Use this to set certain preferences to the selected country.",
    )
    cols = st.columns(2)
    with cols[0]:
        st.button(
            label="Set :blue[**Culture Fit Preferences**] to Reference Country",
            on_click=culture_fit_reference_callback,
            use_container_width=True,
        )
    with cols[1]:
        st.button(
            label="Set :blue[**Quality-of-Life Filters**] to Reference Country",
            on_click=quality_of_life_reference_callback,
            use_container_width=True,
        )

    st.divider()

    st.button(
        "Reset Options to Default",
        use_container_width=True,
        on_click=reset_options_callback,
    )

    return app_options


def get_user_ideal(app_options):
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
        adjective="user",
    )
    return user_ideal


def process_data_culture_fit(df, app_options):
    user_ideal = get_user_ideal(app_options)

    all_countries = list(culture_fit_data_dict.values())

    distances, max_distance = distance_calculations.compute_distances(
        countries_from=[user_ideal],
        countries_to=all_countries,
        distance_metric="Manhattan",
    )
    distances = distances.sort_values("user")
    distances = (
        distances / 100
    )  # Divide by the maximum possible deviation in each dimension, which is 100, to make this a unit distance.
    distances = distances / len(
        dimensions_info.DIMENSIONS
    )  # Divide by the number of dimensions to make this an average over dimensions.
    culture_fit_score = 1 - distances  # Define similarity as 1 - distance
    culture_fit_score = culture_fit_score.reset_index()
    culture_fit_score = culture_fit_score.rename(columns={"index": "country", "user": "cf_score"})

    df = df.merge(culture_fit_score, on="country")

    return df


def process_data_language_prevalence(df, app_options):
    if app_options.do_filter_english:
        df = df.merge(df_english[["country", "english_ratio"]], on="country")
    return df


def process_data_overall_score(df, app_options):
    # Make copies to protect app_options from modification

    # Quality-of-Life Score
    ql_subcodes = ["hp", "sp", "hf"]

    ql_weights = {f"{code}_score": deepcopy(getattr(app_options, f"{code}_score_weight")) for code in ql_subcodes}

    ql_weight_sum = sum([ql_weights[key] for key in ql_weights])
    for key in ql_weights:
        ql_weights[key] /= ql_weight_sum

    ql_score_names_to_weight = [f"{code}_score" for code in ql_subcodes]
    for score_name in ql_score_names_to_weight:
        df[f"{score_name}_weighted"] = df[score_name] * ql_weights[score_name]

    df["ql_score"] = 0.0
    for score_name in ql_score_names_to_weight:
        df["ql_score"] += df[f"{score_name}_weighted"]

    # Overall Score
    overall_subcodes = ["cf", "ql"]

    overall_weights = {
        f"{code}_score": deepcopy(getattr(app_options, f"{code}_score_weight")) for code in overall_subcodes
    }

    overall_weight_sum = sum([overall_weights[key] for key in overall_weights])
    for key in overall_weights:
        overall_weights[key] /= overall_weight_sum

    overall_score_names_to_weight = [f"{code}_score" for code in overall_subcodes]
    for score_name in overall_score_names_to_weight:
        df[f"{score_name}_weighted"] = df[score_name] * overall_weights[score_name]

    df["overall_score"] = 0.0
    for score_name in overall_score_names_to_weight:
        df["overall_score"] += df[f"{score_name}_weighted"]

    df = df.sort_values("overall_score", ascending=False)

    return df


def process_data_ranks(df):
    fields_to_rank = overall_fields + culture_fields + quality_of_life_fields
    for field in fields_to_rank:
        df[f"{field}_rank"] = df[field].rank(ascending=False, method="min").astype(int)
    return df


# TODO move to config files
culture_fit_codes = ["cf"]
quality_of_life_codes = ["ql"]
happy_planet_codes = ["hp"]
social_progress_codes = ["sp"]
human_freedom_codes = ["hf"]


def filter_by_codes(df, app_options, codes):
    for code in codes:
        threshold = getattr(app_options, f"{code}_score_min")
        df["satisfies_filters"] = df["satisfies_filters"] & (df[f"{code}_score"] > threshold)
    return df


@st.cache_data(ttl=TTL)
def process_data_filters(df, app_options):
    df["satisfies_filters"] = True
    if app_options.do_filter_culture_fit:
        df = filter_by_codes(df, app_options, culture_fit_codes)

    if app_options.do_filter_quality_of_life:
        df = filter_by_codes(df, app_options, quality_of_life_codes)

    if app_options.do_filter_happy_planet:
        df = filter_by_codes(df, app_options, happy_planet_codes)

    if app_options.do_filter_social_progress:
        df = filter_by_codes(df, app_options, social_progress_codes)

    if app_options.do_filter_freedom:
        df = filter_by_codes(df, app_options, human_freedom_codes)

    if app_options.do_filter_english:
        df["satisfies_filters"] = df["satisfies_filters"] & (df["english_ratio"] > app_options.english_ratio_min)

    if app_options.do_filter_temperature:
        df["satisfies_filters"] = (
            df["satisfies_filters"]
            & (df["average_temperature_celsius"] > app_options.average_temperature_celsius_min)
            & (df["average_temperature_celsius"] < app_options.average_temperature_celsius_max)
        )

    if app_options.do_filter_sunshine:
        df["satisfies_filters"] = (
            df["satisfies_filters"]
            & (df["average_sunshine_hours_per_day"] > app_options.average_sunshine_hours_per_day_min)
            & (df["average_sunshine_hours_per_day"] < app_options.average_sunshine_hours_per_day_max)
        )

    if app_options.do_filter_continents:
        df["satisfies_filters"] = df["satisfies_filters"] & (df["continent"].isin(app_options.continents))

    return df


@st.cache_data(ttl=TTL)
def process_data(app_options):
    df = mdf
    df = process_data_culture_fit(df, app_options)
    df = process_data_language_prevalence(df, app_options)
    df = process_data_overall_score(df, app_options)
    df = process_data_ranks(df)
    num_total = df.shape[0]  # do this before filtering to get all rows
    df = process_data_filters(df, app_options)

    return df, num_total


def get_google_maps_url(lat: float, lon: float) -> str:
    url_base = "https://www.google.com/maps"
    zoom_level = 5.0
    url = f"{url_base}/@{lat},{lon},{zoom_level}z"
    return url


def open_and_st_markdown(path, encoding="utf8"):
    st.markdown(open(path, encoding=encoding).read())


def run_ui_section_welcome():
    st.title("🌎 :blue[Terra]", anchor=False)
    st.caption("Find the right country for you!")
    st.subheader("What is Terra?", anchor=False)

    # Get part of the README and display it
    whole_README_str = open("./README.md", encoding="utf8").read()
    search_str = "Terra is an app"
    welcome_str = whole_README_str[whole_README_str.find(search_str) :]
    st.markdown(welcome_str)

    # st.image("./assets/data_to_recommendation.png", width=200)
    st.components.v1.iframe("https://globe.gl/example/countries-population/", height=600, scrolling=False)


def run_ui_section_results(df, app_options, num_total):
    focus_container = st.container()
    show_unacceptable_container = st.container()

    with show_unacceptable_container:
        st.success(f"Found {df[df['satisfies_filters']].shape[0]} countries that satisfy filters.")
        show_unacceptable = st.checkbox(
            f"Show results for {df[~df['satisfies_filters']].shape[0]} more countries that do not satisfy filters."
        )
        if not show_unacceptable:
            df = df[df["satisfies_filters"]]

    if df.shape[0] == 0:
        st.warning("No matches found! Try adjusting the filters to be less strict.")
        return

    best_match_country = df.iloc[0]["country"]  # We sorted by overall score previously in process_data_overall_score()

    with st.expander("Select Country", expanded=True):
        cols = st.columns(2)
        with cols[1]:
            sort_by_col = st.selectbox(
                "Sort By",
                options=[
                    "overall_score",
                    "cf_score",
                    "ql_score",
                    "country",
                ],
                format_func=lambda x: {
                    "overall_score": "Overall Score",
                    "cf_score": "Culture Fit Score",
                    "ql_score": "Quality-of-Life Score",
                    "country": "Alphabetical",
                }[x],
            )

        if sort_by_col == "country":
            ascending = True
        else:
            ascending = False

        df_sorted = df.sort_values(sort_by_col, ascending=ascending).reset_index().drop(columns="index")
        countries = list(df_sorted["country"])

        # TODO use the _rank cols for this since we have them now
        def get_rank_prefix(country):
            return df_sorted[df_sorted.country == country].index[0].item() + 1

        def get_rank_prefix_str(country, sort_by_col):
            if sort_by_col == "country":
                return ""
            else:
                return f"({get_rank_prefix(country)}) "

        with cols[0]:
            selected_country = st.selectbox(
                "Country",
                options=countries,
                format_func=lambda x: f"{get_rank_prefix_str(x, sort_by_col)}{x}",
            )

    selected_country_row = df.set_index("country").loc[selected_country]
    best_match_country_row = df.set_index("country").loc[best_match_country]

    with focus_container:
        selected_country_header_str = "Selected Country"
        if selected_country == best_match_country:
            selected_country_header_str += " (Best Match)"
        st.header(
            f"{selected_country_header_str}: :blue[{selected_country_row['country_with_emoji']}]",
            anchor=False,
        )

        # Display a warning if default options are detected
        if app_options == AppOptions():
            st.warning(
                'It looks like you are using the default options, try going to the "Options" tab and changing some'
                " things! :blush:"
            )

    # Prep lists for later
    plottable_fields = overall_fields + quality_of_life_fields + culture_fields + geography_fields

    # Special handling for optional fields
    optional_fields = [
        "english_ratio",
        "average_temperature_celsius",
        "average_sunshine_hours_per_day",
    ]
    for field in optional_fields:
        if field in df.columns:
            plottable_fields += [field]

    numeric_plottable_fields = [x for x in plottable_fields if x != "continent"]

    plottable_field_default_index = plottable_fields.index("overall_score")
    numeric_plottable_field_default_index = numeric_plottable_fields.index("overall_score")

    def execute_world_factbook():
        # CIA World Factbook viewer
        cia_world_factbook_url = world_factbook_utils.get_world_factbook_url(selected_country)
        st.markdown(f"[Open in new tab]({cia_world_factbook_url})")
        st.components.v1.iframe(cia_world_factbook_url, height=600, scrolling=True)

    def execute_google_maps():
        # Getting the coords here instead of merging df_coords earlier helps avoid data loss for missing rows.
        latlon_row = df_coords.loc[selected_country]
        lat = latlon_row.latitude
        lon = latlon_row.longitude

        google_maps_url = get_google_maps_url(lat, lon)

        st.markdown(
            f"[Open in new tab]({google_maps_url})",
            help=(
                "Google Maps cannot be embedded freely; doing so requires API usage, which is not tractable for this"
                " app. As an alternative, simply open the link in a new tab."
            ),
        )

    def execute_selected_country_details():
        st.subheader("CIA World Factbook", anchor=False)
        execute_world_factbook()

        st.subheader("Google Maps", anchor=False)
        execute_google_maps()

    def detailed_country_breakdown(fields, name):
        if name == "Culture Dimensions":
            cols_row1 = st.columns(len(fields) // 2)
            cols_row2 = st.columns(len(fields) // 2)
            cols = cols_row1 + cols_row2
        else:
            cols = st.columns(len(fields))
        for col, field in zip(cols, fields):
            with col:
                val_subcol, rank_subcol = st.columns(2)
                val_subcol.metric(df_format_func(field), utils.pct_fmt(selected_country_row[field]))
                rank = selected_country_row[f"{field}_rank"]
                rank_subcol.metric(f"{df_format_func(field)} Rank", f"{rank} of {num_total}")

                fig = px.box(
                    df,
                    y=field,
                    labels={field: df_format_func(field)},
                    points="all",
                    hover_name="country_with_emoji",
                    orientation="v",
                )

                fig.add_hline(
                    best_match_country_row[field],
                    line_dash="dash",
                    line_color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
                    opacity=0.7,
                    annotation_text=f"{best_match_country} (Best Match)",
                    annotation_position="top right",
                    annotation_font_color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
                )

                fig.add_hline(
                    selected_country_row[field],
                    line_dash="dash",
                    line_color="black",
                    opacity=0.7,
                    annotation_text=f"{selected_country} (Selected)",
                    annotation_position="bottom right",
                    annotation_font_color="black",
                )

                if name == "Culture Dimensions":
                    ref_val = getattr(app_options, f"culture_fit_preference_{field}") * 0.01
                    fig.add_hline(
                        ref_val,
                        line_dash="dash",
                        line_color="orange",
                        opacity=0.7,
                        annotation_text="(User Ideal)",
                        annotation_position="top left",
                        annotation_font_color="orange",
                    )
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

    def execute_score_distributions():
        st.subheader("Overall Score Distributions", anchor=False)
        detailed_country_breakdown(fields=overall_fields, name="Overall Scores")

        st.subheader("Culture Dimension Distributions", anchor=False)
        detailed_country_breakdown(fields=culture_fields, name="Culture Dimensions")

        st.subheader("Quality-of-Life Score Distributions", anchor=False)
        detailed_country_breakdown(fields=quality_of_life_fields, name="Quality-of-Life Scores")

    def execute_overall_score_contributions(df_top_N):
        st.subheader("Overall Score Contributions", anchor=False)
        fig = px.bar(
            df_top_N,
            x="country_with_overall_score_rank",
            y=[
                "Culture Fit Score (weighted)",
                "Quality-of-Life Score (weighted)",
            ],
            labels={"country_with_overall_score_rank": "Country"},
        )
        for idx, row in df_top_N.iterrows():
            fig.add_annotation(
                x=row["country_with_overall_score_rank"],
                y=row["Overall Score"],
                yanchor="bottom",
                showarrow=False,
                align="left",
                text=f"{utils.pct_fmt(row['Overall Score'])}",
                font={"size": 12},
            )
        st.plotly_chart(fig, use_container_width=True)

    def execute_ql_score_contributions(df_top_N):
        st.subheader("Quality-of-Life Score Contributions", anchor=False)
        fig = px.bar(
            df_top_N,
            x="country_with_ql_score_rank",
            y=[
                "Happy Planet Score (weighted)",
                "Social Progress Score (weighted)",
                "Human Freedom Score (weighted)",
            ],
            labels={"country_with_ql_score_rank": "Country"},
        )
        for idx, row in df_top_N.iterrows():
            fig.add_annotation(
                x=row["country_with_ql_score_rank"],
                y=row["Quality-of-Life Score"],
                yanchor="bottom",
                showarrow=False,
                align="left",
                text=f"{utils.pct_fmt(row['Quality-of-Life Score'])}",
                font={"size": 12},
            )
        st.plotly_chart(fig, use_container_width=True)

    def execute_score_contributions():
        cols = st.columns(2)
        with cols[0]:
            N = st.number_input(
                "Number of Top Matching Countries to show",
                min_value=1,
                max_value=100,
                value=20,
            )
        with cols[1]:
            sort_by_field = st.selectbox(
                "Sort By",
                options=[
                    "overall_score",
                    "cf_score",
                    "ql_score",
                    "hp_score",
                    "sp_score",
                    "hf_score",
                ],
                format_func=df_format_func,
            )

        # Create the top N dataframe
        df_top_N = df.head(N)
        df_top_N["country_with_overall_score_rank"] = (
            df_top_N["country"] + " (" + df_top_N["overall_score_rank"].astype(str) + ")"
        )
        df_top_N["country_with_ql_score_rank"] = (
            df_top_N["country"] + " (" + df_top_N["ql_score_rank"].astype(str) + ")"
        )
        df_top_N = df_top_N.rename(columns=df_format_dict)
        df_top_N = df_top_N.sort_values(df_format_func(sort_by_field), ascending=False)

        execute_overall_score_contributions(df_top_N)
        execute_ql_score_contributions(df_top_N)

    def generate_choropleth(df, name, cmap):
        df = df.reset_index()
        fig = px.choropleth(
            df,
            locationmode="country names",
            locations="country",
            color=name,
            hover_name="country_with_emoji",
            color_continuous_scale=cmap,
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig

    def execute_world_map():
        cols = st.columns(4)
        with cols[0]:
            field_for_world_map = st.selectbox(
                label="Field to Plot",
                options=plottable_fields,
                index=plottable_field_default_index,
                format_func=df_format_func,
                key="field_to_plot_world_map",
            )
        with cols[1]:
            world_map_resolution = st.selectbox(
                "Resolution",
                options=[50, 110],
                index=1,
                help=(
                    "Lower numbers will render finer details, but will run slower. Resolution 50 needed for small"
                    " countries e.g. Singapore."
                ),
            )
        with cols[2]:
            world_map_projection_type = st.selectbox(
                label="Projection Type",
                options=map_options.PLOTLY_MAP_PROJECTION_TYPES,
                index=map_options.PLOTLY_MAP_PROJECTION_TYPES.index("robinson"),
                format_func=lambda s: s.title(),
                help=(
                    'See the "Map Projections" section of https://plotly.com/python/map-configuration/ for more'
                    " details."
                ),
            )

        with cols[3]:
            cmap_for_world_map = st.selectbox(
                label="Colormap",
                options=color_options.COLORMAP_OPTIONS,
                index=color_options.COLORMAP_OPTIONS.index(color_options.DEFAULT_COLORMAP),
                help="See https://plotly.com/python/builtin-colorscales/",
            )
        fig = generate_choropleth(df, field_for_world_map, cmap_for_world_map)
        fig.update_geos(resolution=world_map_resolution)
        fig.update_geos(projection_type=world_map_projection_type)
        fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
        fig.update_layout(geo_bgcolor=config.STREAMLIT_CONFIG["theme"]["backgroundColor"])
        st.plotly_chart(fig, use_container_width=True)

    def get_data_for_globe(df, field_for_globe, cmap_for_globe):
        # TODO just merge the df_coords and df to get the data needed
        df = df.reset_index()

        field_min = df[field_for_globe].min()
        field_max = df[field_for_globe].max()
        pointsData = []
        labelsData = []
        globe_colormap = matplotlib.cm.get_cmap(f"cmo.{cmap_for_globe}")
        for idx, row in df.iterrows():
            country = row.country

            latlon_row = df_coords.loc[country]
            lat = latlon_row.latitude
            lon = latlon_row.longitude

            field_val = row[field_for_globe]
            field_rel_val = (field_val - field_min) / (field_max - field_min)

            size_min = 0.2
            size_max = 1.0
            size = size_min + (size_max - size_min) * field_rel_val
            color_rgba = globe_colormap(field_rel_val)
            color_hex = rgb2hex(color_rgba, keep_alpha=True)

            point = {"lat": lat, "lng": lon, "size": size, "color": color_hex}
            pointsData.append(point)

            label = {
                "lat": lat,
                "lng": lon,
                "size": 0.5,
                "color": color_hex,
                "text": f"{country} ({field_val:.2f})",
            }
            labelsData.append(label)

        return pointsData, labelsData

    def execute_globe():
        cols = st.columns(4)
        with cols[0]:
            field_for_globe = st.selectbox(
                label="Field to Plot",
                options=numeric_plottable_fields,
                index=numeric_plottable_field_default_index,
                format_func=df_format_func,
                key="field_to_plot_globe",
            )
        with cols[1]:
            day_or_night = st.selectbox("Day or Night", options=["day", "night"])
        with cols[2]:
            widget_width = st.number_input("Widget Width (pixels)", min_value=100, max_value=4000, value=600)
            widget_height = st.number_input("Widget Height (pixels)", min_value=100, max_value=4000, value=600)
        with cols[3]:
            cmap_for_globe = st.selectbox(
                label="Colormap",
                options=color_options.COLORMAP_OPTIONS,
                index=color_options.COLORMAP_OPTIONS.index(color_options.DEFAULT_COLORMAP),
                help="See https://matplotlib.org/cmocean/",
            )

        pointsData, labelsData = get_data_for_globe(df, field_for_globe, cmap_for_globe)
        streamlit_globe(
            pointsData=pointsData,
            labelsData=labelsData,
            daytime=day_or_night,
            width=widget_width,
            height=widget_height,
        )

    def execute_results_data():
        df_for_table = df.rename(columns=df_format_dict).set_index("Country")
        st.dataframe(df_for_table, use_container_width=True)
        st.download_button("Download", df_for_table.to_csv().encode("utf-8"), "results.csv")

    def set_dr_fields_callback(fields):
        st.session_state.dr_fields = fields

    def execute_dimred_and_clustering():
        # Use containers to have the plot above the options, since the options will take up a lot of space
        plot_container = st.container()
        options_container = st.container()

        # Set default for multiselect
        if "dr_fields" not in st.session_state:
            set_dr_fields_callback(culture_fields)

        with options_container:
            dr_fields = st.multiselect(
                "Fields for Dimensionality Reduction & Clustering",
                options=numeric_plottable_fields,
                format_func=df_format_func,
                key="dr_fields",
            )

            cols = st.columns(4)
            with cols[0]:
                st.button(
                    "Set Fields to All",
                    use_container_width=True,
                    on_click=set_dr_fields_callback,
                    args=[numeric_plottable_fields],
                )
            with cols[1]:
                st.button(
                    "Set Fields to Culture Dimensions",
                    use_container_width=True,
                    on_click=set_dr_fields_callback,
                    args=[culture_fields],
                )
            with cols[2]:
                st.button(
                    "Set Fields to Quality-of-Life Dimensions",
                    use_container_width=True,
                    on_click=set_dr_fields_callback,
                    args=[quality_of_life_fields],
                )
            with cols[3]:
                st.button(
                    "Set Fields to Climate Dimensions",
                    use_container_width=True,
                    on_click=set_dr_fields_callback,
                    args=[climate_fields],
                )

            df_for_dr = df.set_index("country")[dr_fields]

            dimensionality_reducer_name = st.selectbox(
                "Dimensionality Reduction Method",
                options=[
                    "UMAP",
                    "t-SNE",
                    "PCA",
                    "SparsePCA",
                    "TruncatedSVD",
                    "FastICA",
                    "NMF",
                ],
            )
            with st.form("dimesionality_reduction_options"):
                dimensionality_reducer_kwargs = {}

                if dimensionality_reducer_name in [
                    "PCA",
                    "SparsePCA",
                    "TruncatedSVD",
                    "FastICA",
                    "NMF",
                ]:
                    cols = st.columns(0 + 1)

                elif dimensionality_reducer_name == "t-SNE":
                    cols = st.columns(2 + 1)
                    with cols[0]:
                        perplexity = st.slider(
                            "Perplexity",
                            min_value=1.0,
                            max_value=30.0,
                            value=10.0,
                            help=(
                                "The perplexity is related to the number of nearest neighbors that is used in other"
                                " manifold learning algorithms. Larger datasets usually require a larger perplexity."
                                " Different values can result in significantly different results. The perplexity must"
                                " be less than the number of samples. See"
                                " https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"
                            ),
                        )
                    with cols[1]:
                        early_exaggeration = st.slider(
                            "Early Exaggeration",
                            min_value=1.0,
                            max_value=30.0,
                            value=10.0,
                            help=(
                                "Controls how tight natural clusters in the original space are in the embedded space"
                                " and how much space will be between them. For larger values, the space between natural"
                                " clusters will be larger in the embedded space. Again, the choice of this parameter is"
                                " not very critical. If the cost function increases during initial optimization, the"
                                " early exaggeration factor or the learning rate might be too high. See"
                                " https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"
                            ),
                        )

                    dimensionality_reducer_kwargs["perplexity"] = perplexity
                    dimensionality_reducer_kwargs["early_exaggeration"] = early_exaggeration

                elif dimensionality_reducer_name == "UMAP":
                    cols = st.columns(2 + 1)
                    with cols[0]:
                        n_neighbors = st.slider(
                            "Number of Neighbors",
                            min_value=1,
                            max_value=50,
                            value=15,
                            help=(
                                "This parameter controls how UMAP balances local versus global structure in the data."
                                " It does this by constraining the size of the local neighborhood UMAP will look at"
                                " when attempting to learn the manifold structure of the data. This means that low"
                                " values will force UMAP to concentrate on very local structure (potentially to the"
                                " detriment of the big picture), while large values will push UMAP to look at larger"
                                " neighborhoods of each point when estimating the manifold structure of the data,"
                                " losing fine detail structure for the sake of getting the broader of the data. See"
                                " https://umap-learn.readthedocs.io/en/latest/parameters.html"
                            ),
                        )
                    with cols[1]:
                        min_dist = st.slider(
                            "Minimum Distance in Projected Space",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.2,
                            help=(
                                "This parameter controls how tightly UMAP is allowed to pack points together. It, quite"
                                " literally, provides the minimum distance apart that points are allowed to be in the"
                                " low dimensional representation. This means that low values will result in clumpier"
                                " embeddings. This can be useful if you are interested in clustering, or in finer"
                                " topological structure. Larger values will prevent UMAP from packing points together"
                                " and will focus on the preservation of the broad topological structure instead. See"
                                " https://umap-learn.readthedocs.io/en/latest/parameters.html"
                            ),
                        )

                    dimensionality_reducer_kwargs["n_neighbors"] = n_neighbors
                    dimensionality_reducer_kwargs["min_dist"] = min_dist

                with cols[-1]:
                    random_state = st.number_input("Random State", min_value=0, max_value=10, value=0, step=1)
                    dimensionality_reducer_kwargs["random_state"] = random_state

                st.form_submit_button("Update Dimensionality Reduction Options", use_container_width=True)

            df_projection = get_dimension_reduced_df(
                df_for_dr, dimensionality_reducer_name, dimensionality_reducer_kwargs
            )

            clustering_method_name = st.selectbox("Clustering Method", options=["HDBSCAN", "Hierarchical"])

            with st.form("clustering_options"):
                if clustering_method_name == "HDBSCAN":
                    cols = st.columns(4)
                    with cols[0]:
                        min_cluster_size = st.slider(
                            "Min Cluster Size",
                            min_value=2,
                            max_value=20,
                            step=1,
                            value=3,
                            help=(
                                "The smallest size grouping that you wish to consider a cluster. See"
                                " https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size."
                            ),
                        )
                    with cols[1]:
                        min_samples = st.slider(
                            "Min Samples",
                            min_value=1,
                            max_value=20,
                            step=1,
                            value=2,
                            help=(
                                "How conservative you want you clustering to be. The larger the value you provide, the"
                                " more conservative the clustering - more points will be declared as noise, and"
                                " clusters will be restricted to progressively more dense areas. See"
                                " https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size."
                            ),
                        )
                    with cols[2]:
                        cluster_selection_epsilon = st.slider(
                            "Cluster Selection Epsilon",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            help=(
                                "Ensures that clusters below the given threshold are not split up any further. See"
                                " https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-cluster-selection-epsilon."
                            ),
                        )
                elif clustering_method_name == "Hierarchical":
                    cols = st.columns(3)
                    with cols[0]:
                        distance_metric = st.selectbox(
                            "Distance Metric",
                            options=clustering_options.PDIST_METRIC_OPTIONS,
                            help=(
                                "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html"
                            ),
                        )
                    with cols[1]:
                        linkage_method = st.selectbox(
                            "Linkage Method",
                            options=clustering_options.LINKAGE_METHOD_OPTIONS,
                            help=(
                                "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html"
                            ),
                        )

                with cols[-1]:
                    cluster_in_projected_space = st.checkbox(
                        "Cluster in Projected Space",
                        value=False,
                        help=(
                            "If False, clustering will occur in the original pre-dimension-reduced space. If True,"
                            " clustering will occur in the projected post-dimension-reduced space. Because t-SNE and"
                            " UMAP do not preserve distances, clustering in the projected space is not as"
                            " meaningful/principled, but can give more intuitive clusterings when viewed on the plot."
                        ),
                    )

                st.form_submit_button("Update Clustering Options", use_container_width=True)

        if cluster_in_projected_space:
            df_for_clustering = df_projection
        else:
            df_for_clustering = df_for_dr

        with st.form("dim_red_cluster_plot_options"):
            if clustering_method_name == "HDBSCAN":
                cols = st.columns(3)
                with cols[0]:
                    show_country_text = st.checkbox("Show Country Name Text on Plot", value=True)
                with cols[1]:
                    marker_size_field = st.selectbox(
                        "Marker Size Field",
                        options=plottable_fields,
                        format_func=df_format_func,
                    )
                with cols[2]:
                    marker_size_power = st.slider(
                        "Marker Size Power",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        help=(
                            "Power to which to raise the field's value. Higher powers will exaggerate differences"
                            " between points, while lower values will diminish them. A power of 1 will make the marker"
                            " size linearly proportional to the field value. A power of 0 will make all points the same"
                            " size, regardless of the field value."
                        ),
                    )
            elif clustering_method_name == "Hierarchical":
                cols = st.columns(2)
                with cols[0]:
                    color_threshold = st.slider(
                        "Cluster Color Threshold",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.5,
                        help="Lower values will result in more clusters.",
                    )
                with cols[1]:
                    orientation = st.selectbox(
                        "Plot Orientation",
                        options=["bottom", "top", "right", "left"],
                    )

            st.form_submit_button("Update Clustering Plot Options", use_container_width=True)

        if clustering_method_name == "HDBSCAN":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
            clusterer.fit(df_for_clustering)
            df_clusters = pd.DataFrame(clusterer.labels_).rename(columns={0: "cluster"}).astype(str)
            df_clusters.index = df_for_clustering.index
            df_for_dr_plot = pd.concat([df.set_index("country"), df_projection, df_clusters], axis=1).reset_index()
            df_for_dr_plot["marker_size"] = df_for_dr_plot[marker_size_field] ** marker_size_power
            category_orders = {"cluster": [str(i) for i in range(-1, max(clusterer.labels_))]}
            scatter_kwargs = dict(
                x="x",
                y="y",
                hover_name="country_with_emoji",
                hover_data=["overall_score"],
                color="cluster",
                color_discrete_map=color_options.CLUSTER_COLOR_SEQUENCE_MAP,
                category_orders=category_orders,
                size="marker_size",
            )

            if show_country_text:
                scatter_kwargs["text"] = "country"

            with plot_container:
                fig = px.scatter(df_for_dr_plot, **scatter_kwargs)
                if show_country_text:
                    fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

        elif clustering_method_name == "Hierarchical":
            distfun = partial(distance.pdist, metric=distance_metric)
            linkagefun = partial(hierarchy.linkage, method=linkage_method)
            fig = ff.create_dendrogram(
                df_for_clustering,
                orientation=orientation,
                labels=df_for_clustering.index,
                distfun=distfun,
                linkagefun=linkagefun,
                color_threshold=color_threshold,
            )
            fig.add_hline(y=color_threshold, line_dash="dash", line_color="white", opacity=0.5)

            with plot_container:
                st.plotly_chart(fig, use_container_width=True)

    def execute_pair_plot():
        with st.form("pairplot_options"):
            fields_for_pairplot = st.multiselect(
                "Fields for Pair Plot",
                options=plottable_fields,
                default=quality_of_life_fields,
                format_func=df_format_func,
            )
            st.form_submit_button("Update Pair Plot Options")

        if len(fields_for_pairplot) < 2:
            st.warning("Need at least 2 fields for pair plot!")
        else:
            df_for_plot = df.rename(columns=df_format_dict)

            if len(fields_for_pairplot) == 2:
                fig = px.scatter(
                    df_for_plot,
                    x=df_format_func(fields_for_pairplot[0]),
                    y=df_format_func(fields_for_pairplot[1]),
                    hover_name="Country with Emoji",
                )
            else:
                fig = px.scatter_matrix(
                    df_for_plot,
                    dimensions=[df_format_dict[x] for x in fields_for_pairplot],
                    hover_name="Country with Emoji",
                )
                fig.update_traces(diagonal_visible=False, showupperhalf=False)

            st.plotly_chart(fig, use_container_width=True)

    flexecute(
        func=execute_selected_country_details,
        label="Selected Country Details",
        expanded=True,
        checkbox_value=True,
    )
    flexecute(func=execute_score_distributions, label="Score Distributions")
    flexecute(func=execute_score_contributions, label="Score Contributions")
    flexecute(func=execute_world_map, label="World Map")
    flexecute(func=execute_globe, label="Globe")
    flexecute(
        func=execute_dimred_and_clustering,
        label="Dimensionality Reduction & Clustering",
    )
    flexecute(func=execute_pair_plot, label="Pair Plots")
    flexecute(func=execute_results_data, label="Results Table")


def run_ui_subsection_culture_fit_help():
    open_and_st_markdown("./culture_fit/culture_fit_help_intro.md")

    # Programmatically generate the help for national culture dimensions
    st.markdown("## What are the 6 dimensions of national culture?")
    dim_tabs = st.tabs([dimensions_info.DIMENSIONS_INFO[dimension]["name"] for dimension in dimensions_info.DIMENSIONS])
    for dim_idx, dimension in enumerate(dimensions_info.DIMENSIONS):
        with dim_tabs[dim_idx]:
            dimension_info = dimensions_info.DIMENSIONS_INFO[dimension]

            st.markdown(f'### {dimension_info["name"]} ({dimension_info["abbreviation"]})')
            cols = st.columns([4, 2])
            with cols[0]:
                st.markdown(f'*{dimension_info["question"]}*')
                st.markdown(dimension_info["description"])
            with cols[1]:
                st.video(dimension_info["hofstede_youtube_video_url"])

    open_and_st_markdown("./culture_fit/culture_fit_help_outro.md")


def run_ui_section_help():
    with st.expander("Tutorial 🏫"):
        open_and_st_markdown("./help/tutorial.md")
        st.divider()
        st.success(
            "Now that you know the general flow, you can either jump right in and start playing with the app! For a"
            " better understanding, we highly recommend reading the rest of the help section. :blush:"
        )

    with st.expander("Culture Fit 🗺️"):
        run_ui_subsection_culture_fit_help()

    with st.expander("Happy Planet 😊"):
        st.components.v1.iframe("https://happyplanetindex.org/", height=600, scrolling=True)
        open_and_st_markdown("./help/happy_planet_help.md")

    with st.expander("Social Progress 📈"):
        st.caption(
            "The [Social Progress Imperative website](https://www.socialprogress.org/) cannot be embedded in this app."
        )
        st.caption(
            "As an alternative, the [Harvard Business School Institute for Strategy and Competitiveness"
            " page](https://www.isc.hbs.edu/research-areas/Pages/social-progress-index.aspx) is embedded here."
        )
        st.components.v1.iframe(
            "https://www.isc.hbs.edu/research-areas/Pages/social-progress-index.aspx",
            height=600,
            scrolling=True,
        )
        open_and_st_markdown("./help/social_progress_help.md")

    with st.expander("Human Freedom 🎊"):
        st.caption(
            "Neither the [Cato Institute website](https://www.cato.org/human-freedom-index/2022) nor the [Fraser"
            " Institute website](https://www.fraserinstitute.org/studies/human-freedom-index-2022) can be embedded in"
            " this app."
        )
        st.caption(
            "As an alternative, the [Wikipedia article on freedom"
            " indices](https://en.wikipedia.org/wiki/List_of_freedom_indices#Prominent_indices), which references the"
            " Human Freedom Index, is embedded here."
        )
        st.components.v1.iframe(
            "https://en.wikipedia.org/wiki/List_of_freedom_indices#Prominent_indices",
            height=600,
            scrolling=True,
        )
        open_and_st_markdown("./help/human_freedom_help.md")

    with st.expander("Language Prevalence 💬"):
        open_and_st_markdown("./help/language_prevalence_help.md")

    with st.expander("Climate 🌡️"):
        st.components.v1.iframe(
            "https://education.nationalgeographic.org/resource/all-about-climate/",
            height=600,
            scrolling=True,
        )
        open_and_st_markdown("./help/climate_help.md")

    with st.expander("Geography 🏞️"):
        st.components.v1.iframe(
            "https://www.nationalgeographic.org/education/what-is-geography/",
            height=600,
            scrolling=True,
        )
        st.components.v1.iframe(
            "https://education.nationalgeographic.org/resource/Continent/",
            height=600,
            scrolling=True,
        )
        open_and_st_markdown("./help/geography_help.md")

    with st.expander("About Terra ℹ️"):
        open_and_st_markdown("./help/general_help.md")

    with st.expander("Data Sources 📊"):
        open_and_st_markdown("./help/data_sources_help.md")


def run_ui_section_share(app_options):
    terra_url_base = "https://terra-country-recommender.streamlit.app"
    query_params = dataclasses.asdict(app_options)
    query_string = urlencode(query_params, doseq=True)
    url = f"{terra_url_base}/?{query_string}"
    st.write(
        "Copy the link by using the copy-to-clipboard button below, or secondary-click and copy this [link"
        f" address]({url})."
    )
    st.code(url, language="http")


def set_query_params(app_options):
    """Set the query params with all the app_options."""
    st.experimental_set_query_params(**dataclasses.asdict(app_options))


def teardown(app_options):
    """Perform all operations just before the end of the app execution."""

    # Update the state.
    # app_options should not be modified after this point
    state.app_options = app_options

    # Update the query_params.
    set_query_params(app_options)
    return


def check_and_handle_invalid_app_options(app_options):
    invalid_app_options = False

    if not app_options.are_ql_weights_valid:
        st.warning(
            "The Quality-of-Life Preference weights are all zero - the Quality-of-Life Score is not well-defined!"
            " Please set at least one weight greater than zero to continue."
        )
        invalid_app_options = True

    (
        are_ql_weights_valid_100_flag,
        ql_weight_pct_sum,
    ) = app_options.are_ql_weights_valid_100
    if not are_ql_weights_valid_100_flag:
        st.warning(
            f"The Quality-of-Life Preference weights do not add up to 100 (they add up to {ql_weight_pct_sum} right"
            " now) - the Quality-of-Life Score is not well-defined! Please make sure the weights add up to 100."
        )
        invalid_app_options = True

    if not app_options.are_overall_weights_valid:
        st.warning(
            "The Overall Preference weights are all zero - the Overall Score is not well-defined! Please set at least"
            " one weight greater than zero to continue."
        )
        invalid_app_options = True

    (
        are_overall_weights_valid_100_flag,
        overall_weight_pct_sum,
    ) = app_options.are_overall_weights_valid_100
    if not are_overall_weights_valid_100_flag:
        st.warning(
            f"The Overall Preference weights do not add up to 100 (they add up to {overall_weight_pct_sum} right now) -"
            " the Overall Score is not well-defined! Please make sure the weights add up to 100."
        )
        invalid_app_options = True

    return invalid_app_options


def main():
    # NOTE: It is critical to define the tabs first before other operations that
    # conditionally modify the main body of the app to avoid the
    # jump-to-first-tab-on-first-interaction-in-another-tab bug
    tab_names = ["Welcome", "Options", "Results", "Help", "Share"]
    tab_emoji_dict = {
        "Welcome": "👋",
        "Options": "🎛️",
        "Results": "📈",
        "Help": "❓",
        "Share": "🔗",
    }
    tabs = st.tabs([f"{tab_emoji_dict[tab_name]} {tab_name}" for tab_name in tab_names])
    container_dict = {tab_name: tab for tab_name, tab in zip(tab_names, tabs)}

    # Execute static sections that do not depend on app options first before first_run_per_session()
    with container_dict["Welcome"]:
        run_ui_section_welcome()

    # NOTE: There are (nested) tabs defined in run_ui_section_help()
    with container_dict["Help"]:
        run_ui_section_help()

    if "initialized" not in state:
        first_run_per_session()

    with container_dict["Options"]:
        app_options = get_options()
        invalid_app_options = check_and_handle_invalid_app_options(app_options)

    with container_dict["Share"]:
        run_ui_section_share(app_options)

    with container_dict["Results"]:
        if invalid_app_options:
            st.warning("Options are invalid, please correct them to see results here.")
        else:
            df, num_total = process_data(app_options)
            run_ui_section_results(df, app_options, num_total)

    teardown(app_options)


main()


# TODO - re-index all data by country codes
# - https://www.cia.gov/the-world-factbook/references/country-data-codes/
# - https://www.iban.com/country-codes
