from PySide2.QtCore import (
    QSize,
    Qt,
    QPoint
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
    QWidget,
    QComboBox,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox
)

class TrainModelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.backend_combobox = QComboBox()
        self.backend_combobox.setParent(self)
        self.backend_combobox.setFixedSize(QSize(200, 50))
        self.backend_combobox.move(QPoint(200, 30))
        self.backend_combobox.setStyleSheet("font-size: 20px;")
        self.backend_combobox.addItem("Tensorflow")
        self.backend_combobox.addItem("PyTorch")

        self.backend_label = QLabel("Backend:")
        self.backend_label.setParent(self)
        self.backend_label.setFixedSize(QSize(100, 50))
        self.backend_label.move(QPoint(100,42))
        self.backend_label.setStyleSheet("font-size: 20px;")
        self.backend_label.setAlignment(Qt.AlignRight)

        self.model_combobox = QComboBox()
        self.model_combobox.setParent(self)
        self.model_combobox.setFixedSize(QSize(200, 50))
        self.model_combobox.move(QPoint(200, 110))
        self.model_combobox.setStyleSheet("font-size: 20px;")
        self.model_combobox.addItem("ResNet")
        self.model_combobox.addItem("VGG")
        self.model_combobox.addItem("OnsuNet")

        self.model_label = QLabel("Model:")
        self.model_label.setParent(self)
        self.model_label.setFixedSize(QSize(100, 50))
        self.model_label.move(QPoint(100,122))
        self.model_label.setStyleSheet("font-size: 20px;")
        self.model_label.setAlignment(Qt.AlignRight)

        self.dataset_combobox = QComboBox()
        self.dataset_combobox.setParent(self)
        self.dataset_combobox.setFixedSize(QSize(200, 50))
        self.dataset_combobox.move(QPoint(200, 190))
        self.dataset_combobox.setStyleSheet("font-size: 20px;")
        self.dataset_combobox.addItem("FER2013")
        self.dataset_combobox.addItem("CK+")
        self.dataset_combobox.addItem("KDEF")

        self.dataset_label = QLabel("Dataset:")
        self.dataset_label.setParent(self)
        self.dataset_label.setFixedSize(QSize(100, 50))
        self.dataset_label.move(QPoint(100,202))
        self.dataset_label.setStyleSheet("font-size: 20px;")
        self.dataset_label.setAlignment(Qt.AlignRight)

        self.optimizer_combobox = QComboBox()
        self.optimizer_combobox.setParent(self)
        self.optimizer_combobox.setFixedSize(QSize(200, 50))
        self.optimizer_combobox.move(QPoint(200, 270))
        self.optimizer_combobox.setStyleSheet("font-size: 20px;")
        self.optimizer_combobox.addItem("Adam")
        self.optimizer_combobox.addItem("SGD")

        self.optimizer_label = QLabel("Optimizer:")
        self.optimizer_label.setParent(self)
        self.optimizer_label.setFixedSize(QSize(100, 50))
        self.optimizer_label.move(QPoint(100,282))
        self.optimizer_label.setStyleSheet("font-size: 20px;")
        self.optimizer_label.setAlignment(Qt.AlignRight)

        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setParent(self)
        self.epoch_spinbox.setFixedSize(QSize(200, 50))
        self.epoch_spinbox.move(QPoint(650, 30))
        self.epoch_spinbox.setStyleSheet("font-size: 20px;")
        self.epoch_spinbox.setMinimum(1)
        self.epoch_spinbox.setMaximum(1000)
        self.epoch_spinbox.setValue(5)

        self.epoch_label = QLabel("Epoch:")
        self.epoch_label.setParent(self)
        self.epoch_label.setFixedSize(QSize(100, 50))
        self.epoch_label.move(QPoint(550,42))
        self.epoch_label.setStyleSheet("font-size: 20px;")
        self.epoch_label.setAlignment(Qt.AlignRight)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setParent(self)
        self.batch_spinbox.setFixedSize(QSize(200, 50))
        self.batch_spinbox.move(QPoint(650, 110))
        self.batch_spinbox.setStyleSheet("font-size: 20px;")
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(512)
        self.batch_spinbox.setValue(32)

        self.batch_label = QLabel("Batch:")
        self.batch_label.setParent(self)
        self.batch_label.setFixedSize(QSize(100, 50))
        self.batch_label.move(QPoint(550,122))
        self.batch_label.setStyleSheet("font-size: 20px;")
        self.batch_label.setAlignment(Qt.AlignRight)

        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setParent(self)
        self.learning_rate_spinbox.setFixedSize(QSize(200, 50))
        self.learning_rate_spinbox.move(QPoint(650, 190))
        self.learning_rate_spinbox.setStyleSheet("font-size: 20px;")
        self.learning_rate_spinbox.setDecimals(6)
        self.learning_rate_spinbox.setMinimum(0.000001)
        self.learning_rate_spinbox.setMaximum(0.999999)
        self.learning_rate_spinbox.setSingleStep(0.000001)
        self.learning_rate_spinbox.setValue(0.001)

        self.learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_label.setParent(self)
        self.learning_rate_label.setFixedSize(QSize(150, 50))
        self.learning_rate_label.move(QPoint(500,202))
        self.learning_rate_label.setStyleSheet("font-size: 20px;")
        self.learning_rate_label.setAlignment(Qt.AlignRight)

        self.factor_spinbox = QDoubleSpinBox()
        self.factor_spinbox.setParent(self)
        self.factor_spinbox.setFixedSize(QSize(200, 50))
        self.factor_spinbox.move(QPoint(650, 270))
        self.factor_spinbox.setStyleSheet("font-size: 20px;")
        self.factor_spinbox.setDecimals(6)
        self.factor_spinbox.setMinimum(0.000001)
        self.factor_spinbox.setMaximum(0.999999)
        self.factor_spinbox.setSingleStep(0.000001)
        self.factor_spinbox.setValue(0.75)

        self.factor_label = QLabel("Factor:")
        self.factor_label.setParent(self)
        self.factor_label.setFixedSize(QSize(100, 50))
        self.factor_label.move(QPoint(550,282))
        self.factor_label.setStyleSheet("font-size: 20px;")
        self.factor_label.setAlignment(Qt.AlignRight)

        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setParent(self)
        self.patience_spinbox.setFixedSize(QSize(200, 50))
        self.patience_spinbox.move(QPoint(650, 350))
        self.patience_spinbox.setStyleSheet("font-size: 20px;")
        self.patience_spinbox.setMinimum(1)
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(5)

        self.patience_label = QLabel("Patience:")
        self.patience_label.setParent(self)
        self.patience_label.setFixedSize(QSize(100, 50))
        self.patience_label.move(QPoint(550,362))
        self.patience_label.setStyleSheet("font-size: 20px;")
        self.patience_label.setAlignment(Qt.AlignRight)


        self.log_screen = QPlainTextEdit()
        self.log_screen.setParent(self)
        self.log_screen.setFixedSize(QSize(890, 120))
        self.log_screen.move(QPoint(30, 430))
        palette = QPalette()
        palette.setColor(QPalette.Base, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        self.log_screen.setPalette(palette)
        self.log_screen.setReadOnly(True)

        self.button_back = QPushButton("Back")
        self.button_back.setParent(self)
        self.button_back.setFixedSize(200, 100)
        self.button_back.move(QPoint(30, 570))
        self.button_back.clicked.connect(self.back_page)
        self.button_back.setStyleSheet("font-size: 20px;")

        self.button_train = QPushButton("Train")
        self.button_train.setParent(self)
        self.button_train.setFixedSize(200, 100)
        self.button_train.move(QPoint(260, 570))
        self.button_train.clicked.connect(self.test)
        self.button_train.setStyleSheet("font-size: 20px;")

        self.button_save = QPushButton("Save")
        self.button_save.setParent(self)
        self.button_save.setFixedSize(200, 100)
        self.button_save.move(QPoint(490, 570))
        self.button_save.setStyleSheet("font-size: 20px;")

        self.button_next = QPushButton("Next")
        self.button_next.setParent(self)
        self.button_next.setFixedSize(200, 100)
        self.button_next.move(QPoint(720, 570))
        # self.button_next.clicked.connect(self.next_page)
        self.button_next.setStyleSheet("font-size: 20px;")
    
    def back_page(self):
        self.parent().show_page(self.parent().train_or_load_page)

    def next_page(self):
        return NotImplementedError

    def test(self):
        print("test")