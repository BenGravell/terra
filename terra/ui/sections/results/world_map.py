import pandas as pd
import plotly.express as px
import streamlit as st

import terra.app_options as ao
from terra import streamlit_config
from terra import color_options
from terra.data_handling.utils import df_format_func
from terra.data_handling.processing import FieldListBank
from terra.ui.sections import UISection


PLOTLY_MAP_PROJECTION_TYPES = [
    "robinson",
    "orthographic",
    "kavrayskiy7",
    "winkel tripel",
    "equirectangular",
    "mercator",
    "natural earth",
    "miller",
    "eckert4",
    "azimuthal equal area",
    "azimuthal equidistant",
    "conic equal area",
    "conic conformal",
    "conic equidistant",
    "gnomonic",
    "stereographic",
    "mollweide",
    "hammer",
    "transverse mercator",
    "albers usa",
    "aitoff",
    "sinusoidal",
]


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


class WorldMapSection(UISection):
    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        flb = FieldListBank(df)
        cols = st.columns(4)
        with cols[0]:
            field_for_world_map = st.selectbox(
                label="Field to Plot",
                options=flb.plottable_fields,
                index=flb.plottable_field_default_index,
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
                options=PLOTLY_MAP_PROJECTION_TYPES,
                index=PLOTLY_MAP_PROJECTION_TYPES.index("robinson"),
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
        fig.update_layout(geo_bgcolor=streamlit_config.STREAMLIT_CONFIG["theme"]["backgroundColor"])
        st.plotly_chart(fig, use_container_width=True)
