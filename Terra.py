from terra.app_manager import AppManager
from terra.ui.sections.welcome import WelcomeSection


def run():
    WelcomeSection().run()


if __name__ == "__main__":
    with AppManager():
        run()
