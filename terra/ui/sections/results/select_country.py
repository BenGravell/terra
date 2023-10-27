import pandas as pd
import streamlit as st

from terra.ui.sections import UISection


class SelectCountrySection(UISection):
    def run(self, df: pd.DataFrame) -> str:
        cols = st.columns(2)
        with cols[1]:
            sort_by_col = st.selectbox(
                "Sort By",
                options=[
                    "overall_score",
                    "cf_score",
                    "ql_score",
                    "country",
                ],
                format_func=lambda x: {
                    "overall_score": "Overall Score",
                    "cf_score": "Culture Fit Score",
                    "ql_score": "Quality-of-Life Score",
                    "country": "Alphabetical",
                }[x],
            )

        if sort_by_col == "country":
            ascending = True
        else:
            ascending = False

        df_sorted = df.sort_values(sort_by_col, ascending=ascending).reset_index().drop(columns="index")
        countries = list(df_sorted["country"])

        # TODO use the _rank columns of the df for this
        def get_rank_prefix(country):
            return df_sorted[df_sorted.country == country].index[0].item() + 1

        def get_rank_prefix_str(country, sort_by_col):
            if sort_by_col == "country":
                return ""
            else:
                return f"({get_rank_prefix(country)}) "

        with cols[0]:
            selected_country = st.selectbox(
                "Country",
                options=countries,
                format_func=lambda x: f"{get_rank_prefix_str(x, sort_by_col)}{x}",
            )

        return selected_country
