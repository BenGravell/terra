import pandas as pd
import plotly.express as px
import streamlit as st

import terra.app_options as ao
from terra.data_handling.utils import df_format_dict, df_format_func
from terra.ui.sections import UISection
from terra.data_handling.processing import FieldListBank


class PairPlotsSection(UISection):
    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        flb = FieldListBank(df)

        with st.form("pairplot_options"):
            fields_for_pairplot = st.multiselect(
                "Fields for Pair Plot",
                options=flb.plottable_fields,
                default=flb.quality_of_life_fields,
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
