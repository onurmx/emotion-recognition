from PySide2.QtCore import (
    QPoint
)
from PySide2.QtWidgets import (
    QPushButton,
    QWidget,
    QLineEdit,
    QLabel,
    QFileDialog
)


class MassPredictionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QLabel("Mass Prediction")
        self.label.setParent(self)
        self.label.move(QPoint(self.parent().size().width() / 2 - self.label.size().width() / 2, 50))
        self.label.setStyleSheet("font-size: 30px;")
        self.label.setFixedSize(400, 50)

        self.label = QLabel("Path to images:")
        self.label.setParent(self)
        self.label.move(QPoint(self.parent().size().width() / 2 - self.label.size().width() / 2, 150))
        self.label.setStyleSheet("font-size: 20px;")
        self.label.setFixedSize(400, 50)

        self.path_to_images = QLineEdit()
        self.path_to_images.setParent(self)
        self.path_to_images.setFixedSize(400, 50)
        self.path_to_images.move(QPoint(self.parent().size().width() / 2 - self.path_to_images.size().width() / 2, 200))
        self.path_to_images.setStyleSheet("font-size: 20px;")
        self.path_to_images.setReadOnly(True)

        self.button_browse = QPushButton("Browse")
        self.button_browse.setParent(self)
        self.button_browse.setFixedSize(200, 50)
        self.button_browse.move(QPoint(self.parent().size().width() / 2 - self.button_browse.size().width() / 2, 275))
        self.button_browse.clicked.connect(self.browse)
        self.button_browse.setStyleSheet("font-size: 20px;")
        self.button_browse.setDisabled(True)

        self.label = QLabel("Path to save predictions:")
        self.label.setParent(self)
        self.label.move(QPoint(self.parent().size().width() / 2 - self.label.size().width() / 2, 350))
        self.label.setStyleSheet("font-size: 20px;")
        self.label.setFixedSize(400, 50)

        self.path_to_save_predictions = QLineEdit()
        self.path_to_save_predictions.setParent(self)
        self.path_to_save_predictions.setFixedSize(400, 50)
        self.path_to_save_predictions.move(QPoint(self.parent().size().width() / 2 - self.path_to_save_predictions.size().width() / 2, 400))
        self.path_to_save_predictions.setStyleSheet("font-size: 20px;")
        self.path_to_save_predictions.setReadOnly(True)

        self.button_browse = QPushButton("Browse")
        self.button_browse.setParent(self)
        self.button_browse.setFixedSize(200, 50)
        self.button_browse.move(QPoint(self.parent().size().width() / 2 - self.button_browse.size().width() / 2, 475))
        self.button_browse.clicked.connect(self.browse)
        self.button_browse.setStyleSheet("font-size: 20px;")
        self.button_browse.setDisabled(True)

        self.button_back = QPushButton("Back")
        self.button_back.setParent(self)
        self.button_back.setFixedSize(200, 100)
        self.button_back.move(QPoint(self.parent().size().width() / 3 - self.button_back.size().width() / 2, 550))
        self.button_back.clicked.connect(self.back_page)
        self.button_back.setStyleSheet("font-size: 20px;")

        self.button_predict = QPushButton("Predict")
        self.button_predict.setParent(self)
        self.button_predict.setFixedSize(200, 100)
        self.button_predict.move(QPoint(2 * self.parent().size().width() / 3 - self.button_predict.size().width() / 2, 550))
        self.button_predict.clicked.connect(self.predict)
        self.button_predict.setStyleSheet("font-size: 20px;")

        self.is_coming_from_train_page = False

    def browse(self):
        if self.sender().text() == "Browse":
            self.path_to_images.setText(QFileDialog.getExistingDirectory(self, "Select Directory"))
        else:
            self.path_to_save_predictions.setText(QFileDialog.getExistingDirectory(self, "Select Directory"))

    def back_page(self):
        self.parent().show_page(self.parent().single_or_mass_prediction_page)

    def predict(self):
        return NotImplementedError