import streamlit as st

from terra.resource_utils import get_assets_file_path
from terra.ui.sections import UISection


class WelcomeSection(UISection):
    def run(self) -> None:
        st.title("🌎 :blue[Terra]", anchor=False)
        st.caption("Find the right country for you!")
        cols = st.columns([2, 1])

        with cols[0]:
            st.subheader("What is :blue[Terra]?", divider="blue", anchor=False)
            # Get part of the README and display it
            # TODO use absolute path for README
            README_path = "./README.md"
            whole_README_str = open(README_path, encoding="utf8").read()
            search_str = "Terra is an app"
            start_index = whole_README_str.find(search_str)
            welcome_str = whole_README_str[start_index:]
            welcome_str = welcome_str.replace("Terra", ":blue[Terra]")
            st.markdown(welcome_str)
            st.markdown(":blue[Terra] includes data regarding:")

            category_array = [
                [
                    "National Culture",
                    "Quality-of-Life",
                    "",
                ],
                [
                    "Language",
                    "Climate",
                    "Geography",
                ],
            ]

            for category_row_list in category_array:
                subcols = st.columns(len(category_row_list))
                for category, subcol in zip(category_row_list, subcols, strict=True):
                    if category:
                        subcol.metric(category, "")

        with cols[1]:
            with open(get_assets_file_path("welcome.jpg"), "rb") as f:
                welcome_image = f.read()
            st.image(welcome_image)


if __name__ == "__main__":
    # Use this to run just the section as a standalone app
    # TODO unit test this in isolation when streamlit unit testing framework becomes available
    from terra import app_config

    app_config.streamlit_setup()

    WelcomeSection().run()
