from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)


class WelcomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.master_layout = QVBoxLayout()
        self.label_layout = QHBoxLayout()
        self.next_layout = QHBoxLayout()
        self.master_layout.addLayout(self.label_layout)
        self.master_layout.addLayout(self.next_layout)
        self.setLayout(self.master_layout)

        self.label = QLabel("Welcome to the Emotify App!")
        self.label.setStyleSheet("font-size: 45px; font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)
        self.label_layout.addWidget(self.label)

        self.button = QPushButton("Next")
        self.button.clicked.connect(self.next_page)
        self.button.setStyleSheet("font-size: 20px;")
        self.button.setFixedSize(100, 50)
        self.next_layout.addWidget(self.button, alignment=Qt.AlignCenter)

    def next_page(self):
        self.parent().setCurrentIndex(1)
