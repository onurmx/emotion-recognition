from PySide2.QtCore import (
    QSize,
    Qt,
    QPoint
)
from PySide2.QtWidgets import (
    QPushButton,
    QLabel,
    QWidget
)


class WelcomePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.label = QLabel("Welcome to the Emotify App!")
        self.label.setParent(self)
        self.label.setFixedSize(QSize(800, 100))
        self.label.move(QPoint(self.parent().size().width() / 2 - self.label.size().width() / 2, 100))
        self.label.setStyleSheet("font-size: 45px; font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)

        self.cr_label = QLabel("Onur Caglayan, Sueda Hatipoglu, 2023")
        self.cr_label.setParent(self)
        self.cr_label.setFixedSize(QSize(800, 100))
        self.cr_label.move(QPoint(self.parent().size().width() / 2 - self.cr_label.size().width() / 2, 625))
        self.cr_label.setStyleSheet("font-size: 20px;")
        self.cr_label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Next")
        self.button.setParent(self)
        self.button.setFixedSize(QSize(200, 100))
        self.button.move(QPoint(self.parent().size().width() / 2 - self.button.size().width() / 2, 475))
        self.button.clicked.connect(self.next_page)
        self.button.setStyleSheet("font-size: 20px;")

    def next_page(self):
        self.parent().show_page(self.parent().train_or_load_page)