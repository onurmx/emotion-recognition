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


class PageOne(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QStackedLayout()
        self.setLayout(self.layout)
        self.button = QPushButton("Next")
        self.button.clicked.connect(self.next_page)
        self.layout.addWidget(self.button)

    def next_page(self):
        self.parent().setCurrentIndex(self.parent().currentIndex() + 1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotify")
        self.setFixedSize(QSize(950, 700))

        self.stackedwidget = QStackedWidget()

        self.page_one = PageOne(self)
        self.page_two = PageTwo(self)

        self.stackedwidget.addWidget(self.page_one)
        self.stackedwidget.addWidget(self.page_two)

        self.setCentralWidget(self.stackedwidget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
