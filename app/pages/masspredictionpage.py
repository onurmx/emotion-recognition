import app_utils as au
import cv2
import os

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

        self.label_mass_prediction = QLabel("Mass Prediction")
        self.label_mass_prediction.setParent(self)
        self.label_mass_prediction.move(QPoint(self.parent().size().width() / 2 - self.label_mass_prediction.size().width() / 2, 50))
        self.label_mass_prediction.setStyleSheet("font-size: 30px;")
        self.label_mass_prediction.setFixedSize(400, 50)

        self.label_path_to_images = QLabel("Path to images:")
        self.label_path_to_images.setParent(self)
        self.label_path_to_images.move(QPoint(self.parent().size().width() / 2 - self.label_path_to_images.size().width() / 2, 150))
        self.label_path_to_images.setStyleSheet("font-size: 20px;")
        self.label_path_to_images.setFixedSize(400, 50)

        self.path_to_images = QLineEdit()
        self.path_to_images.setParent(self)
        self.path_to_images.setFixedSize(400, 50)
        self.path_to_images.move(QPoint(self.parent().size().width() / 2 - self.path_to_images.size().width() / 2, 200))
        self.path_to_images.setStyleSheet("font-size: 20px;")
        self.path_to_images.setReadOnly(True)

        self.button_browse_path_to_images = QPushButton("Browse source path")
        self.button_browse_path_to_images.setParent(self)
        self.button_browse_path_to_images.setFixedSize(200, 50)
        self.button_browse_path_to_images.move(QPoint(self.parent().size().width() / 2 - self.button_browse_path_to_images.size().width() / 2, 275))
        self.button_browse_path_to_images.clicked.connect(self.browse)
        self.button_browse_path_to_images.setStyleSheet("font-size: 20px;")

        self.label_path_to_save_predictions = QLabel("Path to save predictions:")
        self.label_path_to_save_predictions.setParent(self)
        self.label_path_to_save_predictions.move(QPoint(self.parent().size().width() / 2 - self.label_path_to_save_predictions.size().width() / 2, 350))
        self.label_path_to_save_predictions.setStyleSheet("font-size: 20px;")
        self.label_path_to_save_predictions.setFixedSize(400, 50)

        self.path_to_save_predictions = QLineEdit()
        self.path_to_save_predictions.setParent(self)
        self.path_to_save_predictions.setFixedSize(400, 50)
        self.path_to_save_predictions.move(QPoint(self.parent().size().width() / 2 - self.path_to_save_predictions.size().width() / 2, 400))
        self.path_to_save_predictions.setStyleSheet("font-size: 20px;")
        self.path_to_save_predictions.setReadOnly(True)

        self.button_browse_path_to_save_predictions = QPushButton("Browse output path")
        self.button_browse_path_to_save_predictions.setParent(self)
        self.button_browse_path_to_save_predictions.setFixedSize(200, 50)
        self.button_browse_path_to_save_predictions.move(QPoint(self.parent().size().width() / 2 - self.button_browse_path_to_save_predictions.size().width() / 2, 475))
        self.button_browse_path_to_save_predictions.clicked.connect(self.browse)
        self.button_browse_path_to_save_predictions.setStyleSheet("font-size: 20px;")

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
        if self.sender().text() == "Browse source path":
            self.path_to_images.setText(QFileDialog.getExistingDirectory(self, "Select Directory"))
        else:
            self.path_to_save_predictions.setText(QFileDialog.getExistingDirectory(self, "Select Directory"))

    def back_page(self):
        self.parent().show_page(self.parent().single_or_mass_prediction_page)

    def predict(self):
        self.button_back.setEnabled(False)
        self.button_predict.setEnabled(False)

        if self.path_to_save_predictions.text() != "":
            output_file = open(os.path.join(self.path_to_save_predictions.text(), "predictions.csv"), "w")

        images = [os.path.join(self.path_to_images.text(), f) for f in os.listdir(self.path_to_images.text()) if os.path.isfile(os.path.join(self.path_to_images.text(), f))]
        images = [f for f in images if f.endswith(".jpg") or f.endswith(".png")]

        for image_path in images:
            image = cv2.imread(image_path)
            faces = au.get_faces(image, self.parent().workdir)
            if len(faces) > 0:
                backend = self.parent().load_model_page.backend_combobox.currentText().lower() if self.is_coming_from_train_page == False else self.parent().train_model_page.backend_combobox.currentText().lower()
                model = self.parent().load_model_page.model_combobox.currentText().lower() if self.is_coming_from_train_page == False else self.parent().train_model_page.model_combobox.currentText().lower()
                dataset = self.parent().load_model_page.dataset_combobox.currentText().lower() if self.is_coming_from_train_page == False else self.parent().train_model_page.dataset_combobox.currentText().lower()
                workdir = self.parent().workdir
                emotions = [au.prediction_generator(image[y:y+h, x:x+w],backend, model, ("ckplus" if dataset =="ck+" else dataset),self.parent().train_model_page.trained_net, workdir) for (x, y, w, h) in faces]
                for emotion in emotions:
                    if self.path_to_save_predictions.text() != "":
                        output_file.write(image_path + "," + emotion + "\n")

        if self.path_to_save_predictions.text() != "":
            output_file.close()

        self.button_back.setEnabled(True)
        self.button_predict.setEnabled(True)