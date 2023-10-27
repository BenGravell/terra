import dataclasses

import pandas as pd
import plotly.express as px
import streamlit as st

import terra.app_options as ao
from terra.utils.formatting import pct_fmt
from terra.data_handling.utils import df_format_dict, df_format_func
from terra.ui.sections import UISection, SequentialUISection


@dataclasses.dataclass
class ScoreContributionsDisplayOptions:
    show_percentages_above_bars: bool = True


@dataclasses.dataclass
class ScoreContributionsDataOptions:
    N: int = 10
    sort_by_field: str = "overall_score"


@dataclasses.dataclass
class ScoreContributionsOptions:
    data_options: ScoreContributionsDataOptions
    display_options: ScoreContributionsDisplayOptions


@dataclasses.dataclass
class ScoreContributionsByFieldsSection(UISection):
    name: str
    contribution_fields: list[str]
    aggregation_field: str
    country_with_aggregation_rank_field: str

    @property
    def title(self):
        return f"{self.name} Score Contributions"

    @property
    def country_labels(self):
        return {self.country_with_aggregation_rank_field: "Country"}

    def run(self, df: pd.DataFrame, options: ScoreContributionsOptions):
        st.subheader(self.title, anchor=False)
        fig = px.bar(
            df,
            x=self.country_with_aggregation_rank_field,
            y=self.contribution_fields,
            labels=self.country_labels,
        )
        if options.display_options.show_percentages_above_bars:
            for idx, row in df.iterrows():
                fig.add_annotation(
                    x=row[self.country_with_aggregation_rank_field],
                    y=row[self.aggregation_field],
                    yanchor="bottom",
                    showarrow=False,
                    align="left",
                    text=f"{pct_fmt(row[self.aggregation_field])}",
                    font={"size": 12},
                )
        st.plotly_chart(fig, use_container_width=True)


class ScoreContributionsOptionsSection(UISection):
    def run(self):
        with st.form("score_contributions_options_form"):
            cols = st.columns(2)
            with cols[0]:
                N = st.number_input(
                    "Number of Top Matching Countries to show",
                    min_value=1,
                    max_value=100,
                    value=20,
                )
            with cols[1]:
                sort_by_field = st.selectbox(
                    "Sort By",
                    options=[
                        "overall_score",
                        "cf_score",
                        "ql_score",
                        "hp_score",
                        "sp_score",
                        "hf_score",
                    ],
                    format_func=df_format_func,
                )
            show_percentages_above_bars = st.toggle("Show Percentages Above Bars", value=True)
            st.form_submit_button("Update Score Contributions Options")

        data_options = ScoreContributionsDataOptions(N, sort_by_field)
        display_options = ScoreContributionsDisplayOptions(show_percentages_above_bars)
        return ScoreContributionsOptions(data_options, display_options)


class ScoreContributionsSection(UISection):
    def run(
        self, df: pd.DataFrame, app_options: ao.AppOptions = None, num_total: int = None, selected_country: str = None
    ):
        # Get options from UI
        options = ScoreContributionsOptionsSection().run()

        # Create the top N dataframe
        df_top_N = df.head(options.data_options.N)
        df_top_N["country_with_overall_score_rank"] = (
            df_top_N["country"] + " (" + df_top_N["overall_score_rank"].astype(str) + ")"
        )
        df_top_N["country_with_ql_score_rank"] = (
            df_top_N["country"] + " (" + df_top_N["ql_score_rank"].astype(str) + ")"
        )
        df_top_N = df_top_N.rename(columns=df_format_dict)
        df_top_N = df_top_N.sort_values(df_format_func(options.data_options.sort_by_field), ascending=False)

        # Run UI data display sections
        seq = SequentialUISection(
            [
                ScoreContributionsByFieldsSection(
                    name="Overall",
                    contribution_fields=[
                        "Culture Fit Score (weighted)",
                        "Quality-of-Life Score (weighted)",
                    ],
                    aggregation_field="Overall Score",
                    country_with_aggregation_rank_field="country_with_overall_score_rank",
                ),
                ScoreContributionsByFieldsSection(
                    name="Quality-of-Life",
                    contribution_fields=[
                        "Happy Planet Score (weighted)",
                        "Social Progress Score (weighted)",
                        "Human Freedom Score (weighted)",
                    ],
                    aggregation_field="Quality-of-Life Score",
                    country_with_aggregation_rank_field="country_with_ql_score_rank",
                ),
            ]
        )
        seq.run(df_top_N, options)


if __name__ == "__main__":
    from terra import app_config
    from terra.data_handling.processing import process_data

    app_config.streamlit_setup()
    df, num_total = process_data()
    ScoreContributionsSection().run(df)
