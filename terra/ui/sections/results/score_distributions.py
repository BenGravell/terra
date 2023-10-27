import dataclasses

import pandas as pd
import plotly.express as px
import streamlit as st

import terra.app_options as ao
from terra import streamlit_config
from terra.utils.formatting import pct_fmt
from terra.data_handling.utils import df_format_func
from terra.data_handling.processing import FieldListBank
from terra.ui.sections import UISection, SequentialUISection


@dataclasses.dataclass
class ScoreDistributionsByFieldsSection(UISection):
    name: str
    fields: list[str]

    def make_cols(self):
        # TODO do not hardcode this logic, perform it dynamically based on
        # len(self.fields) creating a grid of cols & rows and flattening the
        # matrix into the flat list of self.cols
        if self.name == "Culture Dimension Distributions":
            cols_row1 = st.columns(len(self.fields) // 2)
            cols_row2 = st.columns(len(self.fields) // 2)
            self.cols = cols_row1 + cols_row2
        else:
            self.cols = st.columns(len(self.fields))

    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        st.subheader(self.name, anchor=False)

        self.make_cols()

        best_match_country = df.iloc[0]["country"]  # Assumes we sorted by overall score previously
        selected_country_row = df.set_index("country").loc[selected_country]
        best_match_country_row = df.set_index("country").loc[best_match_country]

        for col, field in zip(self.cols, self.fields):
            with col:
                st.metric(df_format_func(field), pct_fmt(selected_country_row[field]))
                rank = selected_country_row[f"{field}_rank"]
                st.metric(f"{df_format_func(field)} Rank", f"{rank} of {num_total}")

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
                    line_color=streamlit_config.STREAMLIT_CONFIG["theme"]["primaryColor"],
                    opacity=0.7,
                    annotation_text=f"{best_match_country} (Best Match)",
                    annotation_position="top right",
                    annotation_font_color=streamlit_config.STREAMLIT_CONFIG["theme"]["primaryColor"],
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

                if self.name == "Culture Dimension Distributions":
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


class ScoreDistributionsSection(UISection):
    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        flb = FieldListBank(df)

        seq = SequentialUISection(
            [
                ScoreDistributionsByFieldsSection(fields=flb.overall_fields, name="Overall Score Distributions"),
                ScoreDistributionsByFieldsSection(fields=flb.culture_fields, name="Culture Dimension Distributions"),
                ScoreDistributionsByFieldsSection(
                    fields=flb.quality_of_life_fields, name="Quality-of-Life Score Distributions"
                ),
            ]
        )
        seq.run(df, app_options, num_total, selected_country)


if __name__ == "__main__":
    from terra import app_config
    from terra.data_handling.processing import process_data

    app_config.streamlit_setup()
    app_options = ao.AppOptions()
    df, num_total = process_data(app_options)
    selected_country = "United States"
    ScoreDistributionsSection().run(df, app_options, num_total, selected_country)
