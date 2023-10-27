import dataclasses
from typing import Any

import streamlit as st

from terra import app_config
from terra.data_handling.processing import process_data
from terra.ui.sections.welcome import WelcomeSection
from terra.ui.sections.share import ShareSection
from terra.ui.sections.help import HelpSection
from terra.ui.sections.options import OptionsSection
from terra.ui.sections.results import ResultsSection
from terra import session_state_manager as ssm


@dataclasses.dataclass
class ContainerInfo:
    """Class to hold a container and info about it."""

    name: str
    icon: str | None = None
    container: Any | None = None

    @property
    def label(self) -> str:
        return f"{self.icon} {self.name}"


@dataclasses.dataclass
class ContainerInfoCollection:
    container_infos: list[ContainerInfo]

    def asdict(self):
        return {container_info.name: container_info for container_info in self.container_infos}

    def create_tabs(self):
        # Create tabs and set each tab as container in container_infos
        tabs = st.tabs([container_info.label for container_info in self.container_infos])
        for container_info, tab in zip(self.container_infos, tabs, strict=True):
            container_info.container = tab


def run_ui():
    # NOTE: It is critical to define the tabs first before other operations that
    # conditionally modify the main body of the app to avoid the
    # jump-to-first-tab-on-first-interaction-in-another-tab bug
    container_info_collection = ContainerInfoCollection(
        [
            ContainerInfo(name="Welcome", icon="👋"),
            ContainerInfo(name="Options", icon="🎛️"),
            ContainerInfo(name="Results", icon="📈"),
            ContainerInfo(name="Share", icon="🔗"),
            ContainerInfo(name="Help", icon="❓"),
        ]
    )
    container_info_collection.create_tabs()
    container_infos_dict = container_info_collection.asdict()

    with container_infos_dict["Welcome"].container:
        WelcomeSection().run()

    # NOTE: There are (nested) tabs defined in run_ui_section_help()
    with container_infos_dict["Help"].container:
        HelpSection().run()

    with container_infos_dict["Options"].container:
        app_options = OptionsSection().run()

    with container_infos_dict["Share"].container:
        ShareSection().run(app_options)

    with container_infos_dict["Results"].container:
        if not app_options.are_all_options_valid[0]:
            st.warning("Options are invalid, please correct them to see results here.")
        else:
            df, num_total = process_data(app_options)
            ResultsSection().run(df, app_options, num_total)


def main():
    app_config.streamlit_setup()
    ssm.initialize_run()
    run_ui()
    ssm.finalize_run()


if __name__ == "__main__":
    main()
