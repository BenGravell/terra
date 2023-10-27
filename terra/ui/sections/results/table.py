import pandas as pd
import streamlit as st

import terra.app_options as ao
from terra.ui.sections import UISection

from terra.data_handling.utils import df_format_dict


class ResultsTableSection(UISection):
    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        df_for_table = df.rename(columns=df_format_dict).set_index("Country")
        st.dataframe(df_for_table, use_container_width=True)
        df_for_download = df_for_table.to_csv().encode("utf-8")
        st.download_button("Download", df_for_download, "results.csv")
