from terra.app_manager import AppManager
from terra.ui.sections.help import HelpSection


def run():
    HelpSection().run()


if __name__ == "__main__":
    with AppManager():
        run()
