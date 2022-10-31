from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget
)


class TrainOrLoadPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.master_layout = QVBoxLayout()
        self.opt_layout = QHBoxLayout()
        self.train_layout = QVBoxLayout()
        self.load_layout = QVBoxLayout()
        self.back_layout = QVBoxLayout()
        self.master_layout.addLayout(self.opt_layout)
        self.opt_layout.addLayout(self.train_layout)
        self.opt_layout.addLayout(self.load_layout)
        self.master_layout.addLayout(self.back_layout)
        self.setLayout(self.master_layout)

        self.button1 = QPushButton("Train Model")
        self.button1.setStyleSheet("font-size: 20px;")
        self.button1.setFixedSize(200, 100)
        self.train_layout.addWidget(self.button1, alignment=Qt.AlignCenter)

        self.button2 = QPushButton("Load Model")
        self.button2.setStyleSheet("font-size: 20px;")
        self.button2.setFixedSize(200, 100)
        self.load_layout.addWidget(self.button2, alignment=Qt.AlignCenter)

        self.button3 = QPushButton("Back")
        self.button3.clicked.connect(self.back_page)
        self.button3.setStyleSheet("font-size: 20px;")
        self.button3.setFixedSize(100, 50)
        self.back_layout.addWidget(self.button3, alignment=Qt.AlignCenter)

    def back_page(self):
        self.parent().setCurrentIndex(0)
