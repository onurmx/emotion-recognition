from PySide2.QtCore import (
    QPoint
)
from PySide2.QtWidgets import (
    QPushButton,
    QWidget
)


class TrainOrLoadPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.button1 = QPushButton("Train Model")
        self.button1.setParent(self)
        self.button1.setFixedSize(200, 100)
        self.button1.move(QPoint(self.parent().size().width() / 3 - self.button1.size().width() / 2, 200))
        self.button1.clicked.connect(self.train_page)
        self.button1.setStyleSheet("font-size: 20px;")

        self.button2 = QPushButton("Load Model")
        self.button2.setParent(self)
        self.button2.setFixedSize(200, 100)
        self.button2.move(QPoint(2 * self.parent().size().width() / 3 - self.button2.size().width() / 2, 200))
        self.button2.clicked.connect(self.load_page)
        self.button2.setStyleSheet("font-size: 20px;")

        self.button3 = QPushButton("Back")
        self.button3.setParent(self)
        self.button3.setFixedSize(200, 100)
        self.button3.move(QPoint(2 * self.parent().size().width() / 4 - self.button3.size().width() / 2, 475))
        self.button3.clicked.connect(self.back_page)
        self.button3.setStyleSheet("font-size: 20px;")

    def train_page(self):
        self.parent().show_page(self.parent().train_model_page)

    def load_page(self):
        self.parent().show_page(self.parent().load_model_page)

    def back_page(self):
        self.parent().show_page(self.parent().welcome_page)