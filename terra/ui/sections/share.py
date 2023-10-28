import dataclasses
from urllib.parse import urlencode

import streamlit as st

import terra.app_options as ao
from terra.constants import TERRA_URL_BASE
from terra.ui.sections import UISection
from terra import session_state_manager as ssm


class ShareSection(UISection):
    def run(self) -> None:
        app_options = ssm.get_("app_options")
        query_params = dataclasses.asdict(app_options)
        query_string = urlencode(query_params, doseq=True)
        url = f"{TERRA_URL_BASE}/?{query_string}"
        st.write(
            "Copy the link by using the copy-to-clipboard button below, or secondary-click and copy this [link"
            f" address]({url})."
        )
        st.code(url, language="http")


if __name__ == "__main__":
    # Use this to run just the section as a standalone app
    # TODO unit test this in isolation when streamlit unit testing framework becomes available
    from terra import app_config

    app_config.streamlit_setup()

    ShareSection().run(ao.AppOptions())
