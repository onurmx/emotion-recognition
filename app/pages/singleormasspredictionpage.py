from PySide2.QtCore import (
    QPoint
)
from PySide2.QtWidgets import (
    QPushButton,
    QWidget
)

class SingleOrMassPredictionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.button1 = QPushButton("Single Prediction")
        self.button1.setParent(self)
        self.button1.setFixedSize(200, 100)
        self.button1.move(QPoint(self.parent().size().width() / 3 - self.button1.size().width() / 2, 200))
        self.button1.clicked.connect(self.single_prediction_page)
        self.button1.setStyleSheet("font-size: 20px;")

        self.button2 = QPushButton("Mass Prediction")
        self.button2.setParent(self)
        self.button2.setFixedSize(200, 100)
        self.button2.move(QPoint(2 * self.parent().size().width() / 3 - self.button2.size().width() / 2, 200))
        self.button2.clicked.connect(self.mass_prediction_page)
        self.button2.setStyleSheet("font-size: 20px;")

        self.button3 = QPushButton("Back")
        self.button3.setParent(self)
        self.button3.setFixedSize(200, 100)
        self.button3.move(QPoint(2 * self.parent().size().width() / 4 - self.button3.size().width() / 2, 400))
        self.button3.clicked.connect(self.back_page)
        self.button3.setStyleSheet("font-size: 20px;")

    def single_prediction_page(self):
        self.parent().single_prediction_page.is_coming_from_train_page = False
        return self.parent().show_page(self.parent().single_prediction_page)

    def mass_prediction_page(self):
        return NotImplementedError

    def back_page(self):
        self.parent().show_page(self.parent().load_model_page)