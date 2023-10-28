import streamlit as st

from terra.culture_fit import dimensions_info
from terra.ui.sections import UISection, ExpanderWrappedUISection, SequentialUISection
from terra.ui import open_and_markdown_in_streamlit_help


class TutorialSection(UISection):
    def run(self) -> None:
        open_and_markdown_in_streamlit_help("tutorial.md")
        st.divider()
        st.success(
            "Now that you know the general flow, you can jump right in and start playing with the app! For a better"
            " understanding, we highly recommend reading the rest of the help section. :blush:"
        )


class CultureFitSection(UISection):
    def run(self) -> None:
        open_and_markdown_in_streamlit_help("culture_fit_help_intro.md")

        # Programmatically generate the help for national culture dimensions
        st.markdown("## What are the 6 dimensions of national culture?")
        dim_tabs = st.tabs(
            [dimensions_info.DIMENSIONS_INFO[dimension]["name"] for dimension in dimensions_info.DIMENSIONS]
        )
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

        open_and_markdown_in_streamlit_help("culture_fit_help_outro.md")


class HappyPlanetSection(UISection):
    def run(self) -> None:
        st.components.v1.iframe("https://happyplanetindex.org/", height=600, scrolling=True)
        open_and_markdown_in_streamlit_help("happy_planet_help.md")


class SocialProgressSection(UISection):
    def run(self) -> None:
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
        open_and_markdown_in_streamlit_help("social_progress_help.md")


class HumanFreedomSection(UISection):
    def run(self) -> None:
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
        open_and_markdown_in_streamlit_help("human_freedom_help.md")


class LanguagePrevalenceSection(UISection):
    def run(self) -> None:
        open_and_markdown_in_streamlit_help("language_prevalence_help.md")


class ClimateSection(UISection):
    def run(self) -> None:
        st.components.v1.iframe(
            "https://education.nationalgeographic.org/resource/all-about-climate/",
            height=600,
            scrolling=True,
        )
        open_and_markdown_in_streamlit_help("climate_help.md")


class GeographySection(UISection):
    def run(self) -> None:
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
        open_and_markdown_in_streamlit_help("geography_help.md")


class AboutSection(UISection):
    def run(self) -> None:
        open_and_markdown_in_streamlit_help("general_help.md")


class DataSourcesSection(UISection):
    def run(self) -> None:
        open_and_markdown_in_streamlit_help("data_sources_help.md")


class HelpSection(UISection):
    def __init__(self) -> None:
        self.seq = SequentialUISection(
            [
                ExpanderWrappedUISection(
                    child=TutorialSection(),
                    name="Tutorial",
                    icon="🏫",
                ),
                ExpanderWrappedUISection(
                    child=CultureFitSection(),
                    name="Culture Fit",
                    icon="🗺️",
                ),
                ExpanderWrappedUISection(
                    child=HappyPlanetSection(),
                    name="Happy Planet",
                    icon="😊",
                ),
                ExpanderWrappedUISection(
                    child=SocialProgressSection(),
                    name="Social Progress",
                    icon="📈",
                ),
                ExpanderWrappedUISection(
                    child=TutorialSection(),
                    name="Human Freedom",
                    icon="🎊",
                ),
                ExpanderWrappedUISection(
                    child=TutorialSection(),
                    name="Language Prevalence",
                    icon="💬",
                ),
                ExpanderWrappedUISection(
                    child=ClimateSection(),
                    name="Climate",
                    icon="🌡️",
                ),
                ExpanderWrappedUISection(
                    child=GeographySection(),
                    name="Geography",
                    icon="🏞️",
                ),
                ExpanderWrappedUISection(
                    child=AboutSection(),
                    name="About Terra",
                    icon="🛈",
                ),
                ExpanderWrappedUISection(
                    child=DataSourcesSection(),
                    name="Data Sources",
                    icon="📊",
                ),
            ]
        )

    def run(self) -> None:
        self.seq.run()


if __name__ == "__main__":
    # Use this to run just the section as a standalone app
    # TODO unit test this in isolation when streamlit unit testing framework becomes available
    from terra import app_config

    app_config.streamlit_setup()

    HelpSection().run()
