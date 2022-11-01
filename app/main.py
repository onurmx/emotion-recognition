import sys
import loadpage
import trainorloadpage
import welcomepage
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotify")
        self.setFixedSize(QSize(950, 700))

        self.welcome_page = welcomepage.WelcomePage(self)
        # self.train_or_load_page = trainorloadpage.TrainOrLoadPage(self)
        # self.load_page = loadpage.LoadPage(self)
        self.show_page(self.welcome_page)

    def show_page(self, page):
        self.setCentralWidget(page)
        page.show()

    def hide_page(self, page):
        page.hide()
        self.setCentralWidget(None)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
