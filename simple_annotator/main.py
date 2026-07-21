import logging
import sys

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def  main() -> None:
    # Filter warning "Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits."
    # This warning is expected to show up in normal operation and is benign
    logging.getLogger("matplotlib.axes._base").addFilter(lambda record: not record.getMessage(). startswith("Ignoring fixed"))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
