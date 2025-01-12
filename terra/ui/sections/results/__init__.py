import streamlit as st

import terra.app_options as ao
from terra.data_handling.loading import DATA
from terra.data_handling.processing import process_data
from terra import session_state_manager as ssm
from terra.ui.sections import UISection, ExpanderWrappedUISection, SequentialUISection
from terra.ui.sections.results.select_country import SelectCountrySection
from terra.ui.sections.results.details import SelectedCountryDetailsSection
from terra.ui.sections.results.score_distributions import ScoreDistributionsSection
from terra.ui.sections.results.score_contributions import ScoreContributionsSection
from terra.ui.sections.results.world_map import WorldMapSection
from terra.ui.sections.results.globe import GlobeSection
from terra.ui.sections.results.cluster import DimensionalityReductionClusteringSection
from terra.ui.sections.results.pair_plots import PairPlotsSection
from terra.ui.sections.results.table import ResultsTableSection


def results_prep_ops(df, app_options):
    filter_info_container = st.container()
    focus_container = st.container()

    with filter_info_container:
        # Display a warning if default options are detected
        if app_options == ao.AppOptions():
            st.info(
                'It looks like you are using the default options, try going to the "Options" page and changing some'
                " things! :blush:"
            )

        # Show info about the number of matching countries
        num_countries_satisfy_filters = df[df["satisfies_filters"]].shape[0]
        num_countries_not_satisfy_filters = df[~df["satisfies_filters"]].shape[0]

        if num_countries_satisfy_filters > 0:
            st.success(f"Found {num_countries_satisfy_filters} countries that satisfy filters.")
        else:
            st.warning("No countries found that satisfy filters! Try adjusting the filters to be less strict.")

        # Allow user to override the filters and show results for countries that do not satify the filter criteria
        show_unacceptable = st.toggle(
            f"Show results for {num_countries_not_satisfy_filters} more countries that do not satisfy filters."
        )
        if not show_unacceptable:
            df = df[df["satisfies_filters"]]

        # If after all filtering & options the df is empty, there is nothing to do, so return early
        if df.empty:
            return

    best_match_country = df.iloc[0]["country"]  # We sorted by overall score previously in process_data_overall_score()
    selected_country = ExpanderWrappedUISection(child=SelectCountrySection(), name="Select Country").run(df)
    selected_country_row = df.set_index("country").loc[selected_country]

    with focus_container:
        selected_country_header_str = "Selected Country"
        if selected_country == best_match_country:
            selected_country_header_str += " (Best Match)"
        st.header(
            f"{selected_country_header_str}: :blue[{selected_country_row['country_with_emoji']}]",
            anchor=False,
        )

    return df, selected_country


class ResultsSection(UISection):
    def __init__(self) -> None:
        self.seq = SequentialUISection(
            [
                ExpanderWrappedUISection(
                    child=SelectedCountryDetailsSection(),
                    name="Selected Country Details",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=ScoreDistributionsSection(),
                    name="Score Distributions",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=ScoreContributionsSection(),
                    name="Score Contributions",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=WorldMapSection(),
                    name="World Map",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=GlobeSection(),
                    name="Globe",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=DimensionalityReductionClusteringSection(),
                    name="Dimensionality Reduction & Clustering",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=PairPlotsSection(),
                    name="Pair Plots",
                    conditional=True,
                ),
                ExpanderWrappedUISection(
                    child=ResultsTableSection(),
                    name="Results Table",
                    conditional=True,
                ),
            ]
        )

    def run(self):
        app_options = ssm.get_("app_options")
        if not app_options.are_all_options_valid[0]:
            st.warning("Options are invalid, please correct them to see results here.")
            return

        df, num_total = process_data(app_options)
        df, selected_country = results_prep_ops(df, app_options)
        self.seq.run(df, app_options, num_total, selected_country)


if __name__ == "__main__":
    # Use this to run just the section as a standalone app
    # TODO unit test this in isolation when streamlit unit testing framework becomes available
    from terra import app_config

    app_config.streamlit_setup()

    df = DATA.merged_df.copy()
    app_options = ao.AppOptions()
    num_total = len(df)
    ResultsSection().run(df, app_options, num_total)
