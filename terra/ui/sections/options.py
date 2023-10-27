import math

import numpy as np
import plotly.express as px
import streamlit as st

import terra.app_options as ao
from terra.ui.sections import UISection, ExpanderWrappedUISection, SequentialUISection
from terra.culture_fit import dimensions_info
from terra.data_handling.loading import DATA, DATA_DICT
from terra.data_handling.utils import df_format_func
from terra import session_state_manager as ssm
from terra.data_handling.processing import FieldListBank


NONE_COUNTRY = "(none)"


# TODO move to __init__ or utils
def create_unit_interval_pt_range_list(pt: float):
    return np.arange(0.0, 1.0 + pt, pt).round(2).tolist()


UNIT_INTERVAL_1PT_RANGE_LIST = create_unit_interval_pt_range_list(0.01)
UNIT_INTERVAL_5PT_RANGE_LIST = create_unit_interval_pt_range_list(0.05)


def culture_fit_reference_callback():
    if st.session_state.reference_country == NONE_COUNTRY:
        return

    country_info = DATA_DICT["Culture Fit"].d[st.session_state.reference_country]

    # TODO use ssm to set. Create a helper set_culture_dimensions
    for dimension in dimensions_info.DIMENSIONS:
        st.session_state[dimension] = getattr(country_info, dimension)

    st.toast(
        f"Culture Fit Preferences set to Reference Country :blue[**{st.session_state.reference_country}**]", icon="🎛️"
    )


def quality_of_life_reference_callback():
    if st.session_state.reference_country == NONE_COUNTRY:
        return

    fields = ["hp_score_min", "sp_score_min", "hf_score_min"]
    dataset_names = ["Happy Planet", "Social Progress", "Human Freedom"]
    for field, dataset_name in zip(fields, dataset_names, strict=True):
        df = DATA_DICT[dataset_name].df

        # TODO use ssm to set. Create a helper set_quality_of_life_filters
        st.session_state[field] = (
            math.floor(df.set_index("country").loc[st.session_state.reference_country].item() * 100) / 100
        )

    st.toast(
        f"Quality-of-Life Filters set to Reference Country :blue[**{st.session_state.reference_country}**]", icon="🎛️"
    )


# TODO uppercase
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


def score_strip_plot(df, dimension, xmin, xmax):
    # Add a little buffer to the plot limits to catch endpoints
    dx = xmax - xmin
    xmin_, xmax_ = xmin - (dx / 100), xmax + (dx / 100)
    fig = px.strip(
        df,
        x=dimension,
        labels={dimension: df_format_func(dimension)},
        hover_name="country_with_emoji",
        orientation="h",
        range_x=(xmin_, xmax_),
        height=200,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def reset_options_callback():
    app_options = ao.AppOptions()
    ssm.set_app_options(app_options)


def update_options_callback():
    st.toast("Updated Options", icon="🎛️")


class CultureFitPreferencesSection(UISection):
    def run(self, app_options) -> None:
        merged_df100 = DATA.merged_df.copy()
        merged_df100[dimensions_info.DIMENSIONS] *= 100

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
            score_strip_plot(merged_df100, dimension, 0, 100)

            setattr(app_options, f"culture_fit_preference_{dimension}", preference_val)


class QualityOfLifePreferencesSection(UISection):
    def run(self, app_options) -> None:
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
            options=UNIT_INTERVAL_5PT_RANGE_LIST,
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
            options=UNIT_INTERVAL_5PT_RANGE_LIST,
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
            options=UNIT_INTERVAL_5PT_RANGE_LIST,
            key="hf_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )


class OverallPreferencesSection(UISection):
    def run(self, app_options) -> None:
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
            options=UNIT_INTERVAL_5PT_RANGE_LIST,
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
            options=UNIT_INTERVAL_5PT_RANGE_LIST,
            key="ql_score_weight",
            label_visibility="collapsed",
            format_func=lambda x: round(100 * x),
        )


class PreferencesSection(UISection):
    def __init__(self) -> None:
        self.name = "Preferences"

        self.seq = SequentialUISection(
            [
                ExpanderWrappedUISection(
                    child=CultureFitPreferencesSection(),
                    name="Culture Fit Preferences",
                ),
                ExpanderWrappedUISection(
                    child=QualityOfLifePreferencesSection(),
                    name="Quality-of-Life Preferences",
                ),
                ExpanderWrappedUISection(
                    child=OverallPreferencesSection(),
                    name="Overall Preferences",
                ),
            ]
        )

    def run(self, app_options) -> None:
        st.header(self.name, anchor=False)
        self.seq.run(app_options)


class QualityOfLifeFiltersSection(UISection):
    def run(self, app_options) -> None:
        # TODO construct these programmatically

        slider_str = "Happy Planet Score Min"
        caption_str = "*What is the minimum Happy Planet Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=happy_planet_score_help)
        app_options.hp_score_min = st.select_slider(
            slider_str,
            options=UNIT_INTERVAL_1PT_RANGE_LIST,
            key="hp_score_min",
            label_visibility="collapsed",
        )
        score_strip_plot(DATA.merged_df, "hp_score", 0.0, 1.0)

        slider_str = "Social Progress Score Min"
        caption_str = "*What is the minimum Social Progress Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=social_progress_score_help)
        app_options.sp_score_min = st.select_slider(
            slider_str,
            options=UNIT_INTERVAL_1PT_RANGE_LIST,
            key="sp_score_min",
            label_visibility="collapsed",
        )
        score_strip_plot(DATA.merged_df, "sp_score", 0.0, 1.0)

        slider_str = "Human Freedom Score Min"
        caption_str = "*What is the minimum Human Freedom Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=human_freedom_score_help)
        app_options.hf_score_min = st.select_slider(
            slider_str,
            options=UNIT_INTERVAL_1PT_RANGE_LIST,
            key="hf_score_min",
            label_visibility="collapsed",
        )
        score_strip_plot(DATA.merged_df, "hf_score", 0.0, 1.0)


class OverallFiltersSection(UISection):
    def run(self, app_options) -> None:
        slider_str = "Culture Fit Score Min"
        caption_str = "*What is the minimum Culture Fit Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=culture_fit_score_help)
        app_options.cf_score_min = st.select_slider(
            slider_str,
            options=UNIT_INTERVAL_1PT_RANGE_LIST,
            key="cf_score_min",
            label_visibility="collapsed",
        )

        slider_str = "Quality-of-Life Score Min"
        caption_str = "*What is the minimum Quality-of-Life Score you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=quality_of_life_score_help)
        app_options.ql_score_min = st.select_slider(
            slider_str,
            options=UNIT_INTERVAL_1PT_RANGE_LIST,
            key="ql_score_min",
            label_visibility="collapsed",
        )


class LanguageFiltersSection(UISection):
    def run(self, app_options) -> None:
        slider_str = ":speaking_head_in_silhouette: English Speaking Ratio Min"
        caption_str = "*What is the minimum proportion of English speakers you are willing to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=english_speaking_ratio_help)
        app_options.english_ratio_min = st.select_slider(
            slider_str,
            options=UNIT_INTERVAL_1PT_RANGE_LIST,
            key="english_ratio_min",
            label_visibility="collapsed",
        )


class ClimateFiltersSection(UISection):
    def run(self, app_options) -> None:
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
        score_strip_plot(DATA.merged_df, "average_temperature_celsius", -10, 30)

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
        score_strip_plot(DATA.merged_df, "average_sunshine_hours_per_day", 3.0, 10.0)


class GeographyFiltersSection(UISection):
    def run(self, app_options) -> None:
        slider_str = "🌎🌍🌏 Continents"
        caption_str = "*What continents do you want to consider?*"
        st.write(slider_str)
        st.caption(caption_str, help=continent_help)
        app_options.continents = st.multiselect(
            slider_str,
            options=ao.CONTINENT_OPTIONS,
            key="continents",
            label_visibility="collapsed",
        )


class FiltersSection(UISection):
    def __init__(self) -> None:
        self.name = "Filters"

        self.seq = SequentialUISection(
            [
                ExpanderWrappedUISection(
                    child=QualityOfLifeFiltersSection(),
                    name="Quality-of-Life Filters",
                ),
                ExpanderWrappedUISection(
                    child=OverallFiltersSection(),
                    name="Overall Score Filters",
                ),
                ExpanderWrappedUISection(
                    child=LanguageFiltersSection(),
                    name="Language Filters",
                ),
                ExpanderWrappedUISection(
                    child=ClimateFiltersSection(),
                    name="Climate Filters",
                ),
                ExpanderWrappedUISection(
                    child=GeographyFiltersSection(),
                    name="Geography Filters",
                ),
            ]
        )

    def run(self, app_options) -> None:
        st.header(self.name, anchor=False)
        self.seq.run(app_options)


class OptionsFromUISection(UISection):
    def __init__(self) -> None:
        self.seq = SequentialUISection(
            [
                PreferencesSection(),
                FiltersSection(),
            ]
        )

    def run(self) -> ao.AppOptions:
        app_options = ao.AppOptions()
        st.title("Options", anchor=False)
        self.seq.run(app_options)
        return app_options


class OptionsFormSection(UISection):
    def run(self):
        options_section = OptionsFromUISection()
        with st.form(key="options_form"):
            app_options = options_section.run()
            st.form_submit_button("Update Options", type="primary", on_click=update_options_callback)
        return app_options


class OptionsModifiersSection(UISection):
    def run(self):
        st.header("Option Modifiers", anchor=False)

        # Reference Country
        flb = FieldListBank()
        required_columns = flb.quality_of_life_fields + flb.culture_fields
        reference_country_options = [NONE_COUNTRY] + sorted(DATA.merged_df.dropna(subset=required_columns)["country"])

        def country_to_emoji_func(country):
            country_to_emoji_dict = (
                DATA.merged_df[["country", "country_with_emoji"]].set_index("country").to_dict()["country_with_emoji"]
            )
            return country_to_emoji_dict.get(country)

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
            on_click=reset_options_callback,
        )


class OptionsSection(UISection):
    def __init__(self) -> None:
        self.options_form_section = OptionsFormSection()
        self.options_mod_section = OptionsModifiersSection()

    def run(self):
        app_options = self.options_form_section.run()
        self.options_mod_section.run()
        for message in app_options.are_all_options_valid[1]:
            st.warning(message)
        # Make sure the app_options in the session state is in sync with the widget keys.
        # Note that we cannot call ssm.set_app_options() here because we cannot modify
        # session state keys for streamlit widgets after they are instantiated.
        # We can read from those keys, but not write to them.
        # However we have full read/write control over the app_options.
        ssm.set_("app_options", app_options)
        # app_options should not be modified after this point,
        # neither in the app body or in the session state.
        return app_options


if __name__ == "__main__":
    # Use this to run just the section as a standalone app
    # TODO unit test this in isolation when streamlit unit testing framework becomes available
    OptionsSection().run()
