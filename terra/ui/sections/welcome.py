import streamlit as st

from terra.ui.sections import UISection


class WelcomeSection(UISection):
    def run(self) -> None:
        st.title("🌎 :blue[Terra]", anchor=False)
        st.caption("Find the right country for you!")
        st.subheader("What is Terra?", anchor=False)

        # Get part of the README and display it
        # TODO use absolute path for README
        README_path = "./README.md"
        whole_README_str = open(README_path, encoding="utf8").read()
        search_str = "Terra is an app"
        welcome_str = whole_README_str[whole_README_str.find(search_str) :]
        st.markdown(welcome_str)

        st.components.v1.iframe("https://globe.gl/example/countries-population/", height=600, scrolling=False)


if __name__ == "__main__":
    # Use this to run just the section as a standalone app
    # TODO unit test this in isolation when streamlit unit testing framework becomes available
    WelcomeSection().run()
