import pandas as pd
import streamlit as st

import terra.app_options as ao
from terra.utils.google_maps import get_google_maps_url
from terra.utils.world_factbook import get_world_factbook_url
from terra.data_handling.loading import DATA_DICT
from terra.ui.sections import UISection


class WorldFactbookSection(UISection):
    """CIA World Factbook viewer."""

    def run(self, selected_country):
        cia_world_factbook_url = get_world_factbook_url(selected_country)
        st.link_button("Open in new tab", cia_world_factbook_url)
        st.components.v1.iframe(cia_world_factbook_url, height=600, scrolling=True)


class GoogleMapsSection(UISection):
    """Google Maps viewer."""

    def run(self, selected_country):
        # TODO use DATA.merged_df directly, data loss should not be an issue.
        # Getting the coords here instead of merging df_coords earlier helps avoid data loss for missing rows.
        latlon_row = DATA_DICT["Coordinates"].df.set_index("country").loc[selected_country]
        lat = latlon_row.latitude
        lon = latlon_row.longitude

        google_maps_url = get_google_maps_url(lat, lon)

        st.link_button(
            "Open in new tab",
            google_maps_url,
            help=(
                "Google Maps cannot be embedded freely; doing so requires API usage, which is not tractable for this"
                " app. As an alternative, simply open the link in a new tab."
            ),
        )


class SelectedCountryDetailsSection(UISection):
    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        st.subheader("CIA World Factbook", anchor=False)
        WorldFactbookSection().run(selected_country)

        st.subheader("Google Maps", anchor=False)
        GoogleMapsSection().run(selected_country)
