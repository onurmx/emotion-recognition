from ctypes import alignment
import sys
from PySide2.QtCore import (
    QSize,
    Qt
)
from PySide2.QtGui import (
    QColor,
    QPalette
)
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QStackedWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLayout,
    QWidget
)

class PageTwo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QStackedLayout()
        self.setLayout(self.layout)
        self.button = QPushButton("Back")
        self.button.clicked.connect(self.back_page)
        self.layout.addWidget(self.button)

    def back_page(self):
        self.parent().setCurrentIndex(self.parent().currentIndex() - 1)


class WelcomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout1 = QVBoxLayout()
        self.layout2 = QHBoxLayout()
        self.layout3 = QHBoxLayout()
        self.layout1.addLayout(self.layout2)
        self.layout1.addLayout(self.layout3)
        self.setLayout(self.layout1)

        self.label1 = QLabel("Welcome to the Emotify App!")
        self.label1.setStyleSheet("font-size: 45px; font-weight: bold;")
        self.label1.setAlignment(Qt.AlignCenter)
        self.layout2.addWidget(self.label1)

        self.button = QPushButton("Next")
        self.button.clicked.connect(self.next_page)
        self.button.setStyleSheet("font-size: 20px;")
        self.layout3.addWidget(self.button)
        self.layout3.addWidget(self.button, alignment=Qt.AlignCenter)

    def next_page(self):
        self.parent().setCurrentIndex(self.parent().currentIndex() + 1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotify")
        self.setFixedSize(QSize(950, 700))

        self.stackedwidget = QStackedWidget()

        self.welcome_page = WelcomePage(self)
        self.page_two = PageTwo(self)

        self.stackedwidget.addWidget(self.welcome_page)
        self.stackedwidget.addWidget(self.page_two)

        self.setCentralWidget(self.stackedwidget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
