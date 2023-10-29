import matplotlib.pyplot as plt
import streamlit as st

# from streamlit_extras.metric_cards import style_metric_cards

import terra.streamlit_config as streamlit_config


TTL = 1 * 60 * 60  # Default time-to-live for streamlit cache, in seconds


# TODO: Use streamlit_extras.metric_cards.style_metric_cards once support for streamlit 1.28.0 is provided.
# NOTE: Copied & modified from streamlit_extras.metric_cards.style_metric_cards
# To support change in streamlit 1.28.0 from 1.27.2 to the div name for st.metric.
# (OLD) data-testid="metric-container"
# (NEW) data-testid="stMetric"
def style_metric_cards(
    background_color: str = "#FFF",
    border_size_px: int = 1,
    border_color: str = "#CCC",
    border_radius_px: int = 5,
    border_left_color: str = "#9AD8E1",
    box_shadow: bool = True,
):
    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    st.markdown(
        f"""
        <style>
            div[data-testid="stMetric"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
