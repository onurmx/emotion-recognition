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
    QComboBox,
    QGridLayout,
    QLabel,
    QLayout,
    QWidget
)

class LoadPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.master_layout = QVBoxLayout()
        self.backend_layout = QHBoxLayout()
        self.backend_label_layout = QVBoxLayout()
        self.backend_combobox_layout = QVBoxLayout()
        self.model_layout = QHBoxLayout()
        self.model_label_layout = QVBoxLayout()
        self.model_combobox_layout = QVBoxLayout()
        self.dataset_layout = QHBoxLayout()
        self.dataset_label_layout = QVBoxLayout()
        self.dataset_combobox_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.button_next_layout = QVBoxLayout()
        self.button_back_layout = QVBoxLayout()
        self.master_layout.addLayout(self.backend_layout)
        self.master_layout.addLayout(self.model_layout)
        self.master_layout.addLayout(self.dataset_layout)
        self.master_layout.addLayout(self.button_layout)
        self.backend_layout.addLayout(self.backend_label_layout)
        self.backend_layout.addLayout(self.backend_combobox_layout)
        self.model_layout.addLayout(self.model_label_layout)
        self.model_layout.addLayout(self.model_combobox_layout)
        self.dataset_layout.addLayout(self.dataset_label_layout)
        self.dataset_layout.addLayout(self.dataset_combobox_layout)
        self.button_layout.addLayout(self.button_back_layout)
        self.button_layout.addLayout(self.button_next_layout)
        self.setLayout(self.master_layout)

        self.backend_label = QLabel("Backend:")
        self.backend_label.setStyleSheet("font-size: 20px;")
        self.backend_label.setAlignment(Qt.AlignRight)
        self.backend_label_layout.addWidget(self.backend_label, alignment=Qt.AlignVCenter)

        self.backend_combobox = QComboBox()
        self.backend_combobox.setStyleSheet("font-size: 20px;")
        self.backend_combobox.addItem("Tensorflow")
        self.backend_combobox.addItem("PyTorch")
        self.backend_combobox.setFixedSize(200, 50)
        self.backend_combobox_layout.addWidget(self.backend_combobox, alignment=Qt.AlignLeft)

        self.model_label = QLabel("Model:")
        self.model_label.setStyleSheet("font-size: 20px;")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label_layout.addWidget(self.model_label, alignment=Qt.AlignRight)

        self.model_combobox = QComboBox()
        self.model_combobox.setStyleSheet("font-size: 20px;")
        self.model_combobox.addItem("ResNet")
        self.model_combobox.addItem("VGG")
        self.model_combobox.addItem("OnsuNet")
        self.model_combobox.setFixedSize(200, 50)
        self.model_combobox_layout.addWidget(self.model_combobox, alignment=Qt.AlignLeft)

        self.dataset_label = QLabel("Dataset:")
        self.dataset_label.setStyleSheet("font-size: 20px;")
        self.dataset_label.setAlignment(Qt.AlignCenter)
        self.dataset_label_layout.addWidget(self.dataset_label, alignment=Qt.AlignRight)

        self.dataset_combobox = QComboBox()
        self.dataset_combobox.setStyleSheet("font-size: 20px;")
        self.dataset_combobox.addItem("FER2013")
        self.dataset_combobox.addItem("CK+")
        self.dataset_combobox.addItem("KDEF")
        self.dataset_combobox.setFixedSize(200, 50)
        self.dataset_combobox_layout.addWidget(self.dataset_combobox, alignment=Qt.AlignLeft)

        self.button_back = QPushButton("Back")
        self.button_back.clicked.connect(self.back_page)
        self.button_back.setStyleSheet("font-size: 20px;")
        self.button_back.setFixedSize(200, 50)
        self.button_back_layout.addWidget(self.button_back, alignment=Qt.AlignCenter)

        self.button_next = QPushButton("Next")
        self.button_next.setStyleSheet("font-size: 20px;")
        self.button_next.setFixedSize(200, 50)
        self.button_next_layout.addWidget(self.button_next, alignment=Qt.AlignCenter)
    
    def back_page(self):
        self.parent().setCurrentIndex(1)