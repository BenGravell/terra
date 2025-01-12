import matplotlib.pyplot as plt
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import terra.streamlit_config as streamlit_config
from terra import session_state_manager as ssm


def style_iframe_background_color(color: str) -> None:
    """Custom CSS ("css hack") to set the background color of all iframe components.

    This is needed because most websites assume a white background and will be illegible
    when iframed without this when a dark background is used.
    """

    st.markdown(
        "<style>iframe {background-color: " + color + ";}</style>",
        unsafe_allow_html=True,
    )


def streamlit_setup() -> None:
    st.set_page_config(page_title="Terra", page_icon="🌎", layout="wide")

    style_metric_cards(
        border_left_color=streamlit_config.STREAMLIT_CONFIG["theme"]["primaryColor"],
        border_color=streamlit_config.STREAMLIT_CONFIG["theme"]["secondaryBackgroundColor"],
        background_color=streamlit_config.STREAMLIT_CONFIG["theme"]["backgroundColor"],
        border_size_px=2,
        border_radius_px=20,
        box_shadow=False,
    )

    style_iframe_background_color(color="white")

    # pyplot setup
    plt.style.use(["dark_background", "./terra.mplstyle"])


class AppManager:
    def __enter__(self):
        streamlit_setup()
        ssm.initialize_run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ssm.finalize_run()
        return False
