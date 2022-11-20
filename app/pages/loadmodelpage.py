from PySide2.QtCore import (
    QSize,
    Qt,
    QPoint
)
from PySide2.QtWidgets import (
    QPushButton,
    QLabel,
    QWidget,
    QComboBox
)

class LoadModelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.backend_combobox = QComboBox()
        self.backend_combobox.setParent(self)
        self.backend_combobox.setFixedSize(QSize(200, 50))
        self.backend_combobox.move(QPoint(self.parent().size().width() / 2 - self.backend_combobox.size().width() / 2, 130))
        self.backend_combobox.setStyleSheet("font-size: 20px;")
        self.backend_combobox.addItem("Tensorflow")
        self.backend_combobox.addItem("PyTorch")

        self.backend_label = QLabel("Backend:")
        self.backend_label.setParent(self)
        self.backend_label.setFixedSize(QSize(400, 50))
        self.backend_label.move(
            QPoint(self.parent().size().width() / 2 - self.backend_combobox.size().width() / 2 - self.backend_label.size().width() - 10,
                   142
                   )
        )
        self.backend_label.setStyleSheet("font-size: 20px;")
        self.backend_label.setAlignment(Qt.AlignRight)

        self.model_combobox = QComboBox()
        self.model_combobox.setParent(self)
        self.model_combobox.setFixedSize(QSize(200, 50))
        self.model_combobox.move(QPoint(self.parent().size().width() / 2 - self.model_combobox.size().width() / 2, 230))
        self.model_combobox.setStyleSheet("font-size: 20px;")
        self.model_combobox.addItem("Resnet")
        self.model_combobox.addItem("VGG")
        self.model_combobox.addItem("Onsunet")

        self.model_label = QLabel("Model:")
        self.model_label.setParent(self)
        self.model_label.setFixedSize(QSize(400, 50))
        self.model_label.move(
            QPoint(self.parent().size().width() / 2 - self.model_combobox.size().width() / 2 - self.model_label.size().width() - 10,
                   242
                   )
        )
        self.model_label.setStyleSheet("font-size: 20px;")
        self.model_label.setAlignment(Qt.AlignRight)

        self.dataset_combobox = QComboBox()
        self.dataset_combobox.setParent(self)
        self.dataset_combobox.setFixedSize(QSize(200, 50))
        self.dataset_combobox.move(QPoint(self.parent().size().width() / 2 - self.dataset_combobox.size().width() / 2, 330))
        self.dataset_combobox.setStyleSheet("font-size: 20px;")
        self.dataset_combobox.addItem("FER2013")
        self.dataset_combobox.addItem("CK+")
        self.dataset_combobox.addItem("KDEF")

        self.dataset_label = QLabel("Dataset:")
        self.dataset_label.setParent(self)
        self.dataset_label.setFixedSize(QSize(400, 50))
        self.dataset_label.move(
            QPoint(self.parent().size().width() / 2 - self.dataset_combobox.size().width() / 2 - self.dataset_label.size().width() - 10,
                   342
                   )
        )
        self.dataset_label.setStyleSheet("font-size: 20px;")
        self.dataset_label.setAlignment(Qt.AlignRight)

        self.button_back = QPushButton("Back")
        self.button_back.setParent(self)
        self.button_back.setFixedSize(200, 100)
        self.button_back.move(QPoint(self.parent().size().width() / 3 - self.button_back.size().width() / 2, 475))
        self.button_back.clicked.connect(self.back_page)
        self.button_back.setStyleSheet("font-size: 20px;")

        self.button_next = QPushButton("Next")
        self.button_next.setParent(self)
        self.button_next.setFixedSize(200, 100)
        self.button_next.move(QPoint(2 * self.parent().size().width() / 3 - self.button_next.size().width() / 2, 475))
        self.button_next.clicked.connect(self.next_page)
        self.button_next.setStyleSheet("font-size: 20px;")
    
    def back_page(self):
        self.parent().show_page(self.parent().train_or_load_page)

    def next_page(self):
        self.parent().show_page(self.parent().single_or_mass_prediction_page)
