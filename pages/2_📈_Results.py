from terra.app_manager import AppManager
from terra.ui.sections.results import ResultsSection


def run():
    ResultsSection().run()


if __name__ == "__main__":
    with AppManager():
        run()
