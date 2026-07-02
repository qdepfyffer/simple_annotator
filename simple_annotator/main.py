import sys
from PySide6.QtWidgets import QApplication, QMainWindow

# Set up main window
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Simple Annotator")
        self.resize(1000, 700)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
