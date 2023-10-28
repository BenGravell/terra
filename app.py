from terra import app_config
from terra.ui.sections import LabeledContainerWrappedUISection, SequentialLabeledContainerWrappedUISection
from terra.ui.sections.welcome import WelcomeSection
from terra.ui.sections.share import ShareSection
from terra.ui.sections.help import HelpSection
from terra.ui.sections.options import OptionsSection
from terra.ui.sections.results import ResultsSection
from terra import session_state_manager as ssm


def run_ui():
    seq = SequentialLabeledContainerWrappedUISection(
        [
            LabeledContainerWrappedUISection(
                name="Welcome",
                icon="👋",
                child=WelcomeSection(),
            ),
            LabeledContainerWrappedUISection(
                name="Options",
                icon="🎛️",
                child=OptionsSection(),
            ),
            LabeledContainerWrappedUISection(
                name="Results",
                icon="📈",
                child=ResultsSection(),
            ),
            LabeledContainerWrappedUISection(
                name="Share",
                icon="🔗",
                child=ShareSection(),
            ),
            LabeledContainerWrappedUISection(
                name="Help",
                icon="❓",
                child=HelpSection(),
            ),
        ]
    )
    # NOTE: It is critical to define the tabs first before other operations that
    # conditionally modify the main body of the app to avoid the
    # jump-to-first-tab-on-first-interaction-in-another-tab bug
    seq.create_tabs()
    seq.run()


def main():
    app_config.streamlit_setup()
    ssm.initialize_run()
    run_ui()
    ssm.finalize_run()


if __name__ == "__main__":
    main()
