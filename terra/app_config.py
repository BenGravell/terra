import matplotlib.pyplot as plt
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import terra.streamlit_config as streamlit_config


TTL = 1 * 60 * 60  # Default time-to-live for streamlit cache, in seconds


def streamlit_setup():
    st.set_page_config(page_title="Terra", page_icon="🌎", layout="wide")

    # Custom CSS ("css hack") to set the background color of all iframe components to white.
    # This is needed because most websites assume a white background and will be illegible
    # when iframed without this when a dark background is used.
    st.markdown(
        """
    <style>iframe {background-color: white;}</style>
    """,
        unsafe_allow_html=True,
    )

    style_metric_cards(
        border_left_color=streamlit_config.STREAMLIT_CONFIG["theme"]["primaryColor"],
        border_color=streamlit_config.STREAMLIT_CONFIG["theme"]["secondaryBackgroundColor"],
        background_color=streamlit_config.STREAMLIT_CONFIG["theme"]["backgroundColor"],
        border_size_px=2,
        border_radius_px=20,
        box_shadow=False,
    )

    # Pyplot setup
    plt.style.use(["dark_background", "./terra.mplstyle"])
