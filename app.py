import dataclasses
from copy import deepcopy
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from culture_fit import country_data, distance_calculations, visualisation, dimensions_info
from options import AppOptions, NONE_COUNTRY, PLOTLY_MAP_PROJECTION_TYPES


# Streamlit setup
st.set_page_config(page_title="Terra", page_icon="🌎", layout="wide")

# Pyplot setup
plt.style.use(["dark_background", "./terra.mplstyle"])

# Short convenience alias
state = st.session_state


@st.cache_data
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
    # Coordinates
    df_coords = pd.read_csv("./data/country_coords.csv")
    df_coords = df_coords.set_index("country")
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

    culture_fit_df = pd.DataFrame.from_dict(culture_fit_data_dict, orient='index')[dimensions_info.DIMENSIONS]
    culture_fit_df *= 0.01  # undo 100X scaling
    culture_fit_df = culture_fit_df.rename_axis("country").reset_index()  # move country to column

    # Happy Planet
    happy_planet_df = happy_planet_df[['Country', 'HPI']]
    happy_planet_df = happy_planet_df.rename(columns={'Country': 'country', 'HPI': 'hp_score'})
    happy_planet_df['hp_score'] /= 100

    # Social Progress
    # Pick out just the columns we need and rename country column
    social_progress_cols_keep = ["Country", "Basic Human Needs", "Foundations of Wellbeing", "Opportunity"]
    social_progress_df = social_progress_df[social_progress_cols_keep]
    social_progress_df = social_progress_df.set_index("Country")
    social_progress_df /= 100.0  # Undo 100X scaling
    social_progress_df = social_progress_df.reset_index()
    social_progress_df = social_progress_df.rename(columns={"Country": "country", "Basic Human Needs": "bn_score", "Foundations of Wellbeing": "fw_score", "Opportunity": "op_score",})
    
    # Country emoji
    df_country_to_emoji = df_codes_alpha_3.merge(df_flag_emoji, on="country_code_alpha_3")
    country_to_emoji = df_country_to_emoji[["country", "emoji"]].set_index("country").to_dict()["emoji"]
    return culture_fit_data_dict, culture_fit_df, happy_planet_df, social_progress_df, human_freedom_df, df_english, df_coords, df_codes_alpha_3, df_flag_emoji, country_to_emoji


# Load data
culture_fit_data_dict, culture_fit_df, happy_planet_df, social_progress_df, human_freedom_df, df_english, df_coords, df_codes_alpha_3, df_flag_emoji, country_to_emoji = load_data()


df_format_dict = {
    "country": "Country",
    "overall_score": "Overall Score",
    "cf_score": "Culture Fit Score",
    "hp_score": "Happy Planet Score",
    "bn_score": "Basic Human Needs Score",
    "fw_score": "Foundations of Wellbeing Score",
    "op_score": "Opportunity Score",
    "pf_score": "Personal Freedom Score",
    "ef_score": "Economic Freedom Score",
    "cf_score_weighted": "Culture Fit Score (weighted)",
    "hp_score_weighted": "Happy Planet Score (weighted)",
    "bn_score_weighted": "Basic Human Needs Score (weighted)",
    "fw_score_weighted": "Foundations of Wellbeing Score (weighted)",
    "op_score_weighted": "Opportunity Score (weighted)",
    "pf_score_weighted": "Personal Freedom Score (weighted)",
    "ef_score_weighted": "Economic Freedom Score (weighted)",
    "english_ratio": "English Speaking Ratio",
    "acceptable": "Acceptable",
}
for dimension in dimensions_info.DIMENSIONS:
    df_format_dict[dimension] = dimensions_info.DIMENSIONS_INFO[dimension]['name']


def df_format_func(key):
    return df_format_dict[key]
    

def culture_fit_reference_callback():
    if state.culture_fit_reference_country == NONE_COUNTRY:
        return

    country_info = culture_fit_data_dict[state.culture_fit_reference_country]

    for dimension in dimensions_info.DIMENSIONS:
        state[dimension] = getattr(country_info, dimension)


# TODO move this to a data file
culture_fit_score_help = "Culture Fit Score measures how closely a national culture matches your preferences, as determined by [average cityblock similarity](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html) of the culture dimension vectors of the nation and your ideal."

happy_planet_score_help = "The [Happy Planet Index](https://happyplanetindex.org/learn-about-the-happy-planet-index/) is a measure of sustainable wellbeing, ranking countries by how efficiently they deliver long, happy lives using our limited environmental resources."

basic_needs_score_help = "Basic Human Needs measures the capacity of a society to meet the basic human needs of its citizens. This includes access to nutrition and basic medical care, water and sanitation, shelter, and personal safety."

foundations_wellbeing_score_help = "Foundations of Wellbeing measures the capacity of a society to establish the building blocks that allow citizens and communities to enhance and sustain the quality of their lives. This includes access to basic knowledge via primary and secondary education, access to information and communication, health and wellness, and environmental quality."

opportunity_score_help = "Opportunity measures the capacity of a society to create the conditions for all individuals to reach their full potential. This includes personal rights, personal freedoms, inclusiveness (especially as it relates to marginalized groups), and access to advanced education."

personal_freedom_score_help = "Personal Freedom measures the degree to which members of a country are free to exercise civil liberties. This includes freedom of movement, freedom of religion, freedom of assembly and political action, freedom of the press and information, and freedom to engage in various interpersonal relationships. This also includes the rule of law, security, and safety, which are necessary for meaningful exercise of personal freedoms."

economic_freedom_score_help = "Economic Freedom measures the degree to which members of a country are free to exercise financial liberties. This includes the freedom to trade, the freedom to use sound money. This also includes the size of government, legal system and property rights, and market regulation, which are necessary for meaningful exercise of economic freedoms."

english_speaking_ratio_help = "Ratio of people who speak English as a mother tongue or foreign language to the total population."


def get_options_from_query_params():
    query_params = st.experimental_get_query_params()
    app_options = AppOptions()
    for field in dataclasses.fields(app_options):
        if field.name in query_params:
            query_param_val = query_params[field.name]

            # This takes care of extracting singleton lists (required due to experimental_get_query_params implementation)
            if len(query_param_val) == 1:
                query_param_val = query_param_val[0]

            # This takes care of converting from string to the proper data type for the field
            query_param_val = field.type(query_param_val)

            # Finally overwrite the field in app_options with the query_param_val
            setattr(app_options, field.name, query_param_val)

    return app_options


def get_options_from_ui():
    app_options = AppOptions()

    with st.expander("Culture Fit Preferences"):
        for dimension in dimensions_info.DIMENSIONS:
            dimension_info = dimensions_info.DIMENSIONS_INFO[dimension]

            slider_str = f'{dimension_info["name"]} ({dimension_info["abbreviation"]})'
            help_str = f'{dimension_info["name"]} ({dimension_info["abbreviation"]}): *{dimension_info["question"]}* \n\n {dimension_info["description"]}'

            preference_val = st.slider(slider_str, min_value=0, max_value=100, help=help_str, key=dimension)

            setattr(app_options, f"culture_fit_preference_{dimension}", preference_val)

    with st.expander("Overall Preferences"):
        app_options.cf_score_weight = st.slider(
            "Culture Fit Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.cf_score_weight,
            help=culture_fit_score_help,
        )
        app_options.hp_score_weight = st.slider(
            "Happy Planet Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.hp_score_weight,
            help=happy_planet_score_help,
        )
        app_options.bn_score_weight = st.slider(
            "Basic Human Needs Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.bn_score_weight,
            help=basic_needs_score_help,
        )
        app_options.fw_score_weight = st.slider(
            "Foundations of Wellbeing Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.fw_score_weight,
            help=foundations_wellbeing_score_help,
        )
        app_options.op_score_weight = st.slider(
            "Opportunity Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.op_score_weight,
            help=opportunity_score_help,
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

    with st.expander("Filters"):
        app_options.cf_score_min = st.slider(
            "Culture Fit Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.cf_score_min,
            help=culture_fit_score_help,
        )
        app_options.hp_score_min = st.slider(
            "Happy Planet Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.hp_score_min,
            help=happy_planet_score_help,
        )
        app_options.bn_score_min = st.slider(
            "Basic Human Needs Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.bn_score_min,
            help=basic_needs_score_help,
        )
        app_options.fw_score_min = st.slider(
            "Foundations of Wellbeing Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.fw_score_min,
            help=foundations_wellbeing_score_help,
        )
        app_options.op_score_min = st.slider(
            "Opportunity Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.op_score_min,
            help=opportunity_score_help,
        )
        app_options.pf_score_min = st.slider(
            "Personal Freedom Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.pf_score_min,
            help=personal_freedom_score_help,
        )
        app_options.ef_score_min = st.slider(
            "Economic Freedom Score Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.ef_score_min,
            help=economic_freedom_score_help,
        )
        app_options.english_ratio_min = st.slider(
            "English Speaking Ratio Min",
            min_value=0.0,
            max_value=1.0,
            value=app_options.english_ratio_min,
            help=english_speaking_ratio_help,
        )
        app_options.year_min, app_options.year_max = st.slider(
            "Year Range",
            min_value=2000,
            max_value=2020,
            value=(app_options.year_min, app_options.year_max),
            help="Years over which to aggregate statistics. Only affects Human Freedom scores.",
        )
    return app_options


def initialize_widget_state_from_app_options(app_options):
    # Effectively set the first-time default for certain widgets by initializing
    # a value assigned to its key in session state before the widgets are
    # instantiated for the first time.
    # See https://discuss.streamlit.io/t/why-do-default-values-cause-a-session-state-warning/15485/27
    for dimension in dimensions_info.DIMENSIONS:
        state[dimension] = getattr(app_options, f"culture_fit_preference_{dimension}")


def first_run_per_session():
    # Only pull the query_params on the first run e.g. to support deeplinking.
    # Otherwise, only use the options that have been set in the session.
    # This helps avoid a race condition between getting options via query_params and getting options via the UI.
    state.app_options = get_options_from_query_params()

    initialize_widget_state_from_app_options(state.app_options)

    state.initialized = True


def get_options():
    with st.sidebar:

        # Reference Country
        culture_fit_reference_country_options = [NONE_COUNTRY] + sorted(list(culture_fit_data_dict))
        st.selectbox(
            "Reference Country",
            options=culture_fit_reference_country_options,
            key="culture_fit_reference_country",
        )
        st.button(
            label="Set Culture Fit Preferences to Reference Country",
            on_click=culture_fit_reference_callback,
        )

        # Options
        with st.form(key="options_form"):
            app_options = get_options_from_ui()
            st.form_submit_button(label="Update Options")

    return app_options


def process_data_happy_planet(df, app_options):
    df = df.merge(happy_planet_df, on='country')
    return df


def process_data_social_progress(df, app_options):
    df = df.merge(social_progress_df, on='country')
    return df


@st.cache_data
def process_data_human_freedom(df, app_options):
    human_freedom_df_year_filtered = human_freedom_df.query(f"{app_options.year_min} <= year <= {app_options.year_max}")
    freedom_score_cols = ["pf_score", "ef_score"]
    df = human_freedom_df_year_filtered.groupby(["country"])[freedom_score_cols].mean()
    df[freedom_score_cols] *= 0.1  # undo 10X scaling
    df = df.reset_index()
    return df


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
        countries_from=[user_ideal], countries_to=all_countries, distance_metric="Manhattan"
    )
    distances = distances.sort_values("user")
    distances = distances / 100  # Divide by the maximum possible deviation in each dimension, which is 100, to make this a unit distance.
    distances = distances / len(dimensions_info.DIMENSIONS)  # Divide by the number of dimensions to make this an average over dimensions.
    culture_fit_score = 1 - distances  # Define similarity as 1 - distance
    culture_fit_score = culture_fit_score.reset_index()
    culture_fit_score = culture_fit_score.rename(columns={"index": "country", "user": "cf_score"})

    df = df.merge(culture_fit_score, on="country")
    df = df.merge(culture_fit_df, on="country")

    return df


def process_data_language_prevalence(df, app_options):
    if app_options.do_filter_english:
        df = df.merge(df_english[["country", "english_ratio"]], on="country")
    return df


def process_data_overall_score(df, app_options):
    # Make copies to protect app_options from modification
    twoletter_codes_to_weight = ["cf", "hp", "bn", "fw", "op", "pf", "ef"]
    weights = {f"{twoletter_code}_score": deepcopy(getattr(app_options, f"{twoletter_code}_score_weight")) for twoletter_code in twoletter_codes_to_weight}

    weight_sum = sum([weights[key] for key in weights])
    for key in weights:
        weights[key] /= weight_sum

    score_names_to_weight = [f"{twoletter_code}_score" for twoletter_code in twoletter_codes_to_weight]
    for score_name in score_names_to_weight:
        df[f'{score_name}_weighted'] = df[score_name] * weights[score_name]

    df["overall_score"] = 0.0
    for key in weights:
        df["overall_score"]  = df["overall_score"] + df[f'{key}_weighted']

    df = df.sort_values("overall_score", ascending=False)

    return df

# TODO move to config files
culture_fit_codes = ['cf']
happy_planet_codes = ['hp']
social_progress_codes = ['bn', 'fw', 'op']
human_freedom_codes = ['pf', 'ef']

def filter_by_codes(df, codes):
    for code in codes:
        threshold = getattr(app_options, f'{code}_score_min')
        df["acceptable"] = df["acceptable"] & (df[f"{code}_score"] > threshold)
    return df


@st.cache_data
def process_data_filters(df, app_options):
    df["acceptable"] = True
    if app_options.do_filter_culture_fit:
        df = filter_by_codes(df, culture_fit_codes)

    if app_options.do_filter_happy_planet:
        df = filter_by_codes(df, happy_planet_codes)

    if app_options.do_filter_social_progress:
        df = filter_by_codes(df, social_progress_codes)

    if app_options.do_filter_freedom:
        df = filter_by_codes(df, human_freedom_codes)

    if app_options.do_filter_english:
        df["acceptable"] = df["acceptable"] & (df["english_ratio"] > app_options.english_ratio_min)

    # Drop unacceptable rows
    df = df[df["acceptable"]]
    
    return df


@st.cache_data
def process_data(app_options):
    df = None
    df = process_data_human_freedom(df, app_options)
    df = process_data_happy_planet(df, app_options)
    df = process_data_social_progress(df, app_options)
    df = process_data_culture_fit(df, app_options)
    df = process_data_language_prevalence(df, app_options)
    df = process_data_overall_score(df, app_options)
    df = process_data_filters(df, app_options)

    return df


@st.cache_data
def get_world_factbook_url(country: str) -> str:
    url_base = "https://www.cia.gov/the-world-factbook/countries"
    country_slug = country.lower().replace(" ", "-")
    url = f"{url_base}/{country_slug}/"
    return url


@st.cache_data
def get_google_maps_url(lat: float, lon: float) -> str:
    url_base = "https://www.google.com/maps"
    zoom_level = 5.0
    url = f"{url_base}/@{lat},{lon},{zoom_level}z"
    return url


def run_ui_section_title():
    st.title("🌎 :blue[Terra]", anchor=False)
    st.caption("Find the right country for you!")


def run_ui_section_best_match(df):
    best_match_row = df.iloc[0]
    best_match_country = str(best_match_row.country)
    best_match_country_emoji = country_to_emoji[best_match_country]

    st.header("Your Best Match Country", anchor=False)
    cols = st.columns([4, 2])
    with cols[0]:
        st.header(f":blue[{best_match_country}] ({best_match_country_emoji})", anchor=False)
    with cols[1]:
        st.image(visualisation.country_urls.COUNTRY_URLS[best_match_country], width=100)

    # CIA World Factbook viewer
    cia_world_factbook_url = get_world_factbook_url(best_match_country)
    with st.expander("CIA World Factbook"):
        st.markdown(f"[Open in new tab]({cia_world_factbook_url})")
        st.components.v1.iframe(cia_world_factbook_url, height=600, scrolling=True)

    # Google Maps viewer
    # Getting the coords here instead of merging the df_coords earlier helps avoid potential data loss for missing rows.
    latlon_row = df_coords.loc[best_match_country]
    lat = latlon_row.latitude
    lon = latlon_row.longitude

    google_maps_url = get_google_maps_url(lat, lon)
    with st.expander("Google Maps"):
        st.markdown(f"[Open in new tab]({google_maps_url})")
        st.caption(
            "Google Maps cannot be embedded freely; doing so requires API usage, which is not tractable for this app. As an alternative, simply open the link in a new tab."
        )


# TODO replace pyplot radar plots with plotly radar plots
# See https://plotly.com/python/radar-chart/
def get_radar(country_names, user_ideal):
    dimensions = distance_calculations.compute_dimensions(
        [culture_fit_data_dict[country_name] for country_name in country_names]
    )
    reference = distance_calculations.compute_dimensions([user_ideal])
    radar = visualisation.generate_radar_plot(dimensions, reference)
    return radar


def run_ui_section_top_n_matches(df, app_options):
    st.header(f"Top Matching Countries", anchor=False)

    N = st.number_input(
        "Number of Top Matching countries to show",
        min_value=1,
        max_value=40,
        value=10,
    )

    df_top_N = df.head(N).rename(columns=df_format_dict)

    def pct_fmt(x):
        return f"{round(100*x, 2):.0f}%"

    with st.expander("Score Contributions", expanded=True):
        fig = px.bar(
            df_top_N,
            x="Country",
            y=[
                "Culture Fit Score (weighted)",
                "Happy Planet Score (weighted)",
                "Basic Human Needs Score (weighted)",
                "Foundations of Wellbeing Score (weighted)",
                "Opportunity Score (weighted)",
                "Personal Freedom Score (weighted)",
                "Economic Freedom Score (weighted)",
            ],
        )
        for idx, row in df_top_N.iterrows():
            fig.add_annotation(
                x=row['Country'],
                y=row["Overall Score"],
                yanchor="bottom",
                showarrow=False,
                align="left",
                text=f"{pct_fmt(row['Overall Score'])}",
                font={"size": 12},
            )
        fig.update_layout(legend=dict(orientation="v", yanchor="top", y=-0.3, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)
        

    with st.expander("Culture Fit Radar Plots"):
        show_radar = st.checkbox(
            label="Show radar plots",
            value=False,
        )
        if show_radar:
            st.caption('', help="The dashed :red[red shape] depicts your preferences.")
            radar = get_radar(country_names=df_top_N['Country'], user_ideal=get_user_ideal(app_options))
            st.pyplot(radar)
        else:
            st.info('Enable "Show radar plots" to populate this section. Note that enabling radar plots can slow rendering time significantly.')


def run_ui_section_all_matches(df):
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

    plottable_fields = ["overall_score", "cf_score", "hp_score", "bn_score", "fw_score", "op_score", "pf_score", "ef_score"]
    plottable_fields += dimensions_info.DIMENSIONS
    if "english_ratio" in df.columns:
        plottable_fields += ["english_ratio"]
    
    plottable_field_default_index = plottable_fields.index("overall_score")

    with st.expander("World Map", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            field_for_world_map = st.selectbox(
                label="Field to Plot",
                options=plottable_fields,
                index=plottable_field_default_index,
                format_func=df_format_func,
            )
        with cols[1]:
            world_map_resolution = st.selectbox("Resolution", options=[50, 110], index=1, help="Lower numbers will render finer details, but will run slower. Resolution <= 50 needed for small countries e.g. Singapore.")
        with cols[2]:
            world_map_projection_type = st.selectbox(
                label="Projection Type",
                options=PLOTLY_MAP_PROJECTION_TYPES,
                index=PLOTLY_MAP_PROJECTION_TYPES.index("robinson"),
                format_func=lambda s: s.title(),
                help='See the "Map Projections" section of https://plotly.com/python/map-configuration/ for more details.'
            )
        fig = generate_choropleth(df, field_for_world_map)
        fig.update_geos(resolution=world_map_resolution)
        fig.update_geos(projection_type=world_map_projection_type)
        fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
        fig.update_layout(geo_bgcolor="#0E1117")  # manually match the theme backgroundColor
        st.plotly_chart(fig, use_container_width=True)


    with st.expander("Results Data"):
        df_for_table = df.rename(columns=df_format_dict).set_index('Country').drop('Acceptable', axis='columns')

        st.dataframe(df_for_table, use_container_width=True)
        st.download_button("Download", df_for_table.to_csv().encode('utf-8'), "results.csv")

    with st.expander("Flag Plot"):
        with st.form("plot_options"):
            cols = st.columns(2)
            with cols[0]:
                x_column = st.selectbox("x-axis", options=plottable_fields, index=plottable_fields.index('pf_score'), format_func=df_format_func)
            with cols[1]:
                y_column = st.selectbox("y-axis", options=plottable_fields, index=plottable_fields.index('ef_score'), format_func=df_format_func)
            st.form_submit_button("Update Plot Options")
        scatterplot = visualisation.generate_scatterplot(df.set_index("country"), x_column, y_column)
        st.bokeh_chart(scatterplot, use_container_width=True)
    
    with st.expander("Pair Plot"):
        show_pair_plot = st.checkbox(
            label="Show pair plot",
            value=False,
        )
        if show_pair_plot:
            with st.form("pairplot_options"):
                cols = st.columns(2)
                with cols[0]:
                    x_fields_for_pairplot = st.multiselect("X Fields for Pair Plot", options=plottable_fields, default=["cf_score", "hp_score", "bn_score", "fw_score", "op_score", "pf_score", "ef_score"], format_func=df_format_func)
                    x_len = len(x_fields_for_pairplot)
                with cols[1]:
                    y_fields_for_pairplot = st.multiselect("Y Fields for Pair Plot", options=plottable_fields, default=['overall_score'], format_func=df_format_func)
                    y_len = len(y_fields_for_pairplot)
                                
                st.form_submit_button("Update Pair Plot Options")
                fig = sns.pairplot(data=df.rename(columns=df_format_dict), 
                                   x_vars=[df_format_dict[x] for x in x_fields_for_pairplot],
                                   y_vars=[df_format_dict[y] for y in y_fields_for_pairplot],
                                   )
            if x_len * y_len > 10:
                st.warning('Selected fields will attempt to generate many plots, rendering may take a long time (be prepared to wait or change your selection).')
            st.pyplot(fig, use_container_width=True)
        else:
            st.info('Enable "Show pair plot" to populate this section. Note that enabling pair plots can slow rendering time significantly.')


def run_ui_subsection_culture_fit_help():
    st.markdown(open("./culture_fit/culture_fit_help_intro.md", encoding="utf8").read())

    # Programmatically generate the help for national culture dimensions
    st.markdown("## What are the 6 dimensions of national culture?")
    dim_tabs = st.tabs([dimensions_info.DIMENSIONS_INFO[dimension]['name'] for dimension in dimensions_info.DIMENSIONS])
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

    st.markdown(open("./culture_fit/culture_fit_help_outro.md", encoding="utf8").read())


def run_ui_section_help():
    st.header("Help", anchor=False)

    with st.expander("Tutorial 🏫"):
        st.success("If you have not already, we highly recommend reading the rest of the help section before proceeding. :blush:")
        st.markdown(open("./help/tutorial.md", encoding="utf8").read())

    with st.expander("About Terra ℹ️"):
        run_ui_section_title()
        st.markdown(open("./help/general_help.md", encoding="utf8").read())

    with st.expander("Culture Fit 🗺️"):
        run_ui_subsection_culture_fit_help()

    with st.expander("Happy Planet 😊"):
        st.markdown(open("./help/happy_planet_help.md", encoding="utf8").read())

    with st.expander("Social Progress 📈"):
        st.markdown(open("./help/social_progress_help.md", encoding="utf8").read())

    with st.expander("Human Freedom 🎊"):
        st.markdown(open("./help/human_freedom_help.md", encoding="utf8").read())

    with st.expander("Language Prevalence 💬"):
        st.markdown(open("./help/language_prevalence_help.md", encoding="utf8").read())

    with st.expander("Data Sources 📊"):
        st.markdown(open("./help/data_sources_help.md", encoding="utf8").read())


def run_ui_section_share(app_options):
    terra_url_base = 'https://terra-country-recommender.streamlit.app'
    query_params = dataclasses.asdict(app_options)
    query_string = urlencode(query_params, doseq=True)
    url = f'{terra_url_base}/?{query_string}'
    st.write(f"Copy the link below or copy this [link address]({url}).")
    st.code(url, language='http')


def set_query_params(app_options):
    # Set the query params with all the app_options
    st.experimental_set_query_params(**dataclasses.asdict(app_options))


def teardown(app_options):
    # Update the state
    # app_options should not be modified after this point
    state.app_options = app_options

    # Update the query_params
    set_query_params(app_options)
    return


def check_if_app_options_are_default(app_options):
    app_options_default = AppOptions()
    for key in dataclasses.asdict(app_options_default):
        if getattr(app_options, key) != getattr(app_options_default, key):
            return False
    return True


################################################################################
## Main
################################################################################

if not "initialized" in state:
    first_run_per_session()


app_options = get_options()

if check_if_app_options_are_default(app_options):
    st.info(
        "It looks like you are using the default app options. Try opening the sidebar and changing some things! :blush:"
    )

df = process_data(app_options)

no_matches = df.shape[0] == 0
if no_matches:
    st.warning("No matches found! Try adjusting the filters to be less strict.")
else:
    tabs = st.tabs(["Best Match", "Top Matches", "All Matches", "Help", "Share"])
    with tabs[0]:
        run_ui_section_best_match(df)
    with tabs[1]:
        run_ui_section_top_n_matches(df, app_options)
    with tabs[2]:
        run_ui_section_all_matches(df)
    with tabs[3]:
        run_ui_section_help()
    with tabs[4]:
        run_ui_section_share(app_options)

teardown(app_options)


# TODO - re-index all data by country codes
# - https://www.cia.gov/the-world-factbook/references/country-data-codes/
# - https://www.iban.com/country-codes