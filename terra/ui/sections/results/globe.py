import dataclasses
from typing import TypeAlias

import pandas as pd
import matplotlib
from matplotlib.colors import rgb2hex
import streamlit as st
from streamlit_globe import streamlit_globe

import terra.app_options as ao
from terra import color_options
from terra.data_handling.utils import df_format_func
from terra.data_handling.processing import FieldListBank
from terra.ui.sections import UISection


PointsData: TypeAlias = list[dict[str : float | str]]
LabelsData: TypeAlias = list[dict[str : float | str]]

SIZE_MIN = 0.2
SIZE_MAX = 1.0


@dataclasses.dataclass
class GlobeData:
    pointsData: PointsData
    labelsData: LabelsData


@dataclasses.dataclass
class GlobeOptions:
    field: str
    day_or_night: str
    widget_width: int
    widget_height: int
    cmap: str


def get_data_for_globe(df, field, cmap) -> GlobeData:
    required_fields = ["country", "latitude", "longitude", field]
    gdf = df[required_fields].copy().dropna()

    field_min = gdf[field].min()
    field_max = gdf[field].max()
    pointsData = []
    labelsData = []
    globe_colormap = matplotlib.cm.get_cmap(f"cmo.{cmap}")

    for idx, row in gdf.iterrows():
        # Extract
        country = row["country"]
        lat = row["latitude"]
        lon = row["longitude"]
        field_val = row[field]

        # Process
        field_rel_val = (field_val - field_min) / (field_max - field_min)
        size = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * field_rel_val
        color_rgba = globe_colormap(field_rel_val)
        color_hex = rgb2hex(color_rgba, keep_alpha=True)

        # Store
        point = {
            "lat": lat,
            "lng": lon,
            "size": size,
            "color": color_hex,
        }
        label = {
            "lat": lat,
            "lng": lon,
            "size": 0.5,
            "color": color_hex,
            "text": f"{country} ({field_val:.2f})",
        }
        pointsData.append(point)
        labelsData.append(label)

    return GlobeData(pointsData, labelsData)


class GlobeOptionsSection(UISection):
    def run(self, df: pd.DataFrame) -> GlobeOptions:
        flb = FieldListBank(df)

        with st.form("globe_options"):
            cols = st.columns(5)
            with cols[0]:
                field = st.selectbox(
                    label="Field to Plot",
                    options=flb.numeric_plottable_fields,
                    index=flb.numeric_plottable_field_default_index,
                    format_func=df_format_func,
                    key="field_to_plot_globe",
                )
            with cols[1]:
                day_or_night = st.selectbox("Day or Night", options=["day", "night"])
            with cols[2]:
                cmap = st.selectbox(
                    label="Colormap",
                    options=color_options.COLORMAP_OPTIONS,
                    index=color_options.COLORMAP_OPTIONS.index(color_options.DEFAULT_COLORMAP),
                    help="See https://matplotlib.org/cmocean/",
                )
            with cols[3]:
                widget_width = st.number_input("Widget Width (pixels)", min_value=100, max_value=4000, value=600)
            with cols[4]:
                widget_height = st.number_input("Widget Height (pixels)", min_value=100, max_value=4000, value=600)

            st.form_submit_button("Update Globe Options")

        return GlobeOptions(field, day_or_night, widget_width, widget_height, cmap)


class GlobeSection(UISection):
    def run(
        self,
        df: pd.DataFrame,
        app_options: ao.AppOptions = None,
        num_total: int = None,
        selected_country: str = None,
    ):
        globe_options = GlobeOptionsSection().run(df)
        globe_data = get_data_for_globe(df, globe_options.field, globe_options.cmap)
        streamlit_globe(
            pointsData=globe_data.pointsData,
            labelsData=globe_data.labelsData,
            daytime=globe_options.day_or_night,
            width=globe_options.widget_width,
            height=globe_options.widget_height,
        )


if __name__ == "__main__":
    from terra import app_config
    from terra.data_handling.loading import DATA

    app_config.streamlit_setup()
    GlobeSection().run(DATA.merged_df)
