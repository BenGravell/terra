from terra.app_manager import AppManager
from terra.ui.sections.share import ShareSection


def run():
    ShareSection().run()


if __name__ == "__main__":
    with AppManager():
        run()
