from terra.app_manager import AppManager
from terra.ui.sections.options import OptionsSection


def run():
    OptionsSection().run()


if __name__ == "__main__":
    with AppManager():
        run()
