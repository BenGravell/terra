import dataclasses
from copy import deepcopy

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from culture_fit import country_data
from culture_fit import distance_calculations
from culture_fit import visualisation
from culture_fit import dimensions_info

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

    # Human Freedom
    human_freedom_df = pd.read_csv("data/human-freedom-index-2022.csv")
    # Culture Fit
    culture_fit_data_dict = country_data.get_country_dict()
    # English speaking
    df_english = pd.read_csv("data/english_speaking.csv")
    # Coordinates
    df_coords = pd.read_csv("data/country_coords.csv")
    df_coords = df_coords.set_index("country")

    return human_freedom_df, culture_fit_data_dict, df_english, df_coords


@st.cache_data
def load_country_to_emoji():
    # Country codes alpha3
    df_codes_alpha_3 = pd.read_csv("data/country_codes_alpha_3.csv")
    # Flag emoji
    df_flag_emoji = pd.read_csv("data/country_flag_emoji.csv")

    df_country_to_emoji = df_codes_alpha_3.merge(df_flag_emoji, on="country_code_alpha_3")
    return df_country_to_emoji[["country", "emoji"]].set_index("country").to_dict()["emoji"]


# Load data
human_freedom_df, culture_fit_data_dict, df_english, df_coords = load_data()
COUNTRY_TO_EMOJI = load_country_to_emoji()


def culture_fit_reference_callback():
    if state.culture_fit_reference_country == NONE_COUNTRY:
        return

    country_info = culture_fit_data_dict[state.culture_fit_reference_country]

    for dimension in dimensions_info.DIMENSIONS:
        state[dimension] = getattr(country_info, dimension)


# TODO move this to a data file
culture_fit_score_help = "Culture Fit Score measures how closely a national culture matches your preferences, as determined by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of the culture dimension vectors of the nation and your ideal."
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

    with st.expander("Culture Fit preferences"):
        for dimension in dimensions_info.DIMENSIONS:
            dimension_info = dimensions_info.DIMENSIONS_INFO[dimension]

            slider_str = f'{dimension_info["name"]} ({dimension_info["abbreviation"]})'
            help_str = f'{dimension_info["name"]} ({dimension_info["abbreviation"]}): *{dimension_info["question"]}* \n\n {dimension_info["description"]}'

            preference_val = st.slider(slider_str, min_value=0, max_value=100, help=help_str, key=dimension)

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
            help=culture_fit_score_help,
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

    with st.expander("Weights"):
        app_options.cf_score_weight = st.slider(
            "Culture Fit Score Weight",
            min_value=0.0,
            max_value=1.0,
            value=app_options.cf_score_weight,
            help=culture_fit_score_help,
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
        culture_fit_reference_country_options = [NONE_COUNTRY] + sorted(list(culture_fit_data_dict))
        st.selectbox(
            "Reference Country",
            options=culture_fit_reference_country_options,
            key="culture_fit_reference_country",
        )
        st.button(
            label="Set Culture Fit preferences to selected reference country",
            on_click=culture_fit_reference_callback,
        )

        with st.form(key="options_form"):
            app_options = get_options_from_ui()
            st.form_submit_button(label="Update Options")

    return app_options


@st.cache_data
def process_data_human_freedom(df, app_options):
    human_freedom_df_year_filtered = human_freedom_df.query(f"{app_options.year_min} <= year <= {app_options.year_max}")
    freedom_score_cols = ["pf_score", "ef_score"]
    df = human_freedom_df_year_filtered.groupby(["country"])[freedom_score_cols].mean()
    df[freedom_score_cols] *= 0.1  # undo 10X scaling
    df = df.reset_index()
    return df


@st.cache_data
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
        ivr=app_options.culture_fit_preference_ind,  # not a typo
        adjective="user",
    )
    return user_ideal


@st.cache_data
def process_data_culture_fit(df, app_options):
    user_ideal = get_user_ideal(app_options)

    all_countries = list(culture_fit_data_dict.values())

    distances, max_distance = distance_calculations.compute_distances(
        countries_from=[user_ideal], countries_to=all_countries, distance_metric="Cosine"
    )
    distances = distances.sort_values("user")
    culture_fit_score = (1 - distances).reset_index()
    culture_fit_score = culture_fit_score.rename(columns={"index": "country", "user": "cf_score"})

    df = df.merge(culture_fit_score, on="country")

    return df


@st.cache_data
def process_data_language_prevalence(df, app_options):
    if app_options.do_filter_english:
        df = df.merge(df_english[["country", "english_ratio"]], on="country")
    return df


@st.cache_data
def process_data_overall_score(df, app_options):
    # Make copies to protect app_options from modification
    pf_score_weight = deepcopy(app_options.pf_score_weight)
    ef_score_weight = deepcopy(app_options.ef_score_weight)
    cf_score_weight = deepcopy(app_options.cf_score_weight)

    weight_sum = pf_score_weight + ef_score_weight + cf_score_weight
    pf_score_weight /= weight_sum
    ef_score_weight /= weight_sum
    cf_score_weight /= weight_sum

    df["pf_score_weighted"] = df["pf_score"] * pf_score_weight
    df["ef_score_weighted"] = df["ef_score"] * ef_score_weight
    df["cf_score_weighted"] = df["cf_score"] * cf_score_weight

    df["overall_score"] = df["pf_score_weighted"] + df["ef_score_weighted"] + df["cf_score_weighted"]
    df = df.sort_values("overall_score", ascending=False)

    return df


@st.cache_data
def process_data_filters(df, app_options):
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

    return df


@st.cache_data
def process_data(app_options):
    df = None
    df = process_data_human_freedom(df, app_options)
    df = process_data_culture_fit(df, app_options)
    df = process_data_language_prevalence(df, app_options)
    df = process_data_overall_score(df, app_options)
    df = process_data_filters(df, app_options)

    return df


def get_world_factbook_url(country: str) -> str:
    url_base = "https://www.cia.gov/the-world-factbook/countries"
    country_slug = country.lower().replace(" ", "-")
    url = f"{url_base}/{country_slug}/"
    return url


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
    best_match_country_emoji = COUNTRY_TO_EMOJI[best_match_country]

    st.header("Your Best Match Country:", anchor=False)
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

    # Google Earth viewer
    # Getting the coords here instead of merging the df_coords earlier helps avoid potential data loss for missing rows.
    latlon_row = df_coords.loc[best_match_country]
    lat = latlon_row.latitude
    lon = latlon_row.longitude

    google_maps_url = get_google_maps_url(lat, lon)
    with st.expander("Google Earth"):
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
        max_value=20,
        value=5,
    )

    show_radar = st.checkbox(
        label="Show radar plots",
        value=False,
        help="Skipping radar plots can significantly improve rendering time.",
    )

    column_remap = {
        "overall_score": "Overall Score",
        "pf_score": "Personal Freedom Score",
        "ef_score": "Economic Freedom Score",
        "cf_score": "Cultural Fit Score",
        "pf_score_weighted": "Personal Freedom Score (weighted)",
        "ef_score_weighted": "Economic Freedom Score (weighted)",
        "cf_score_weighted": "Cultural Fit Score (weighted)",
    }
    df_top_N = df.head(N).rename(columns=column_remap)

    def pct_fmt(x):
        return f"{round(100*x, 2):.0f}%"

    st.subheader("Score Contributions", anchor=False)
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

    st.subheader("Culture Fit Radar Plots", anchor=False, help="The dashed :red[red shape] depicts your preferences.")
    if show_radar:
        radar = get_radar(country_names=df_top_N.country, user_ideal=get_user_ideal(app_options))
        st.pyplot(radar)
    else:
        st.info('Enable "Show radar plots" to populate this section.')


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

    with st.expander("World Map", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            # field_for_world_map_options = ["overall_score", "cf_score", "pf_score", "ef_score", "english_ratio"]
            field_for_world_map_options = df.columns.to_list()
            field_for_world_map_default_index = field_for_world_map_options.index("overall_score")
            field_for_world_map = st.selectbox(
                label="Field to Plot",
                options=field_for_world_map_options,
                index=field_for_world_map_default_index,
            )

        with cols[1]:
            world_map_projection_type = st.selectbox(
                label="Projection Type",
                options=PLOTLY_MAP_PROJECTION_TYPES,
                index=PLOTLY_MAP_PROJECTION_TYPES.index("robinson"),
                format_func=lambda s: s.title(),
            )

        fig = generate_choropleth(df, field_for_world_map)
        fig.update_geos(projection_type=world_map_projection_type)
        fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
        fig.update_layout(geo_bgcolor="#0E1117")
        st.plotly_chart(fig, use_container_width=True)

    dfc = df.set_index("country")

    with st.expander("Raw Results Data"):
        st.dataframe(dfc, use_container_width=True)

    with st.expander("Raw Results Plot"):
        cols = st.columns(2)
        with cols[0]:
            x_column = st.selectbox("x-axis", options=dfc.columns, index=0)
        with cols[1]:
            y_column = st.selectbox("y-axis", options=dfc.columns, index=1)
        scatterplot = visualisation.generate_scatterplot(dfc, x_column, y_column)
        st.bokeh_chart(scatterplot, use_container_width=True)


def run_ui_subsection_culture_fit_help():
    st.markdown(open("./culture_fit/culture_fit_help_intro.md").read())

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

    st.markdown(open("./culture_fit/culture_fit_help_outro.md").read())


def run_ui_section_help():
    st.header("Help", anchor=False)

    with st.expander("About This App 🛈"):
        st.markdown(open("./help/general_help.md").read())

    with st.expander("Culture Fit 🗺️"):
        run_ui_subsection_culture_fit_help()

    with st.expander("Human Freedom 🎊"):
        st.markdown(open("./help/human_freedom_help.md").read())

    with st.expander("Language Prevalence 💬"):
        st.markdown(open("./help/language_prevalence_help.md").read())

    with st.expander("Data Sources 📊"):
        st.markdown(open("./help/data_sources_help.md").read())


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

run_ui_section_title()


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
    tabs = st.tabs(["Best Match", "Top Matches", "All Matches", "Help"])
    with tabs[0]:
        run_ui_section_best_match(df)
    with tabs[1]:
        run_ui_section_top_n_matches(df, app_options)
    with tabs[2]:
        run_ui_section_all_matches(df)
    with tabs[3]:
        run_ui_section_help()

teardown(app_options)
