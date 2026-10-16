"""
App entry point
Builds Qt application object, main window, and gives control to the Qt event loop
ui.py is responsible for mostly everything else
"""

import logging
import sys

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def  main() -> None:
    """
    Start GUI and block until the window is fully closed
    """
    # Filter warning "Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits."
    # This warning is expected to show up in normal operation and is benign
    logging.getLogger("matplotlib.axes._base").addFilter(lambda record: not record.getMessage(). startswith("Ignoring fixed"))
    
    # Have to create QApplication before anything else
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # This will start the Qt event loop and only passes its return value to sys.exit() once the application has fully closed
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
