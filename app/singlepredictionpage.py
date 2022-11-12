import cv2
import app_utils as au

from PySide2.QtCore import (
    QSize,
    Qt,
    QPoint
)
from PySide2.QtGui import (
    QColor,
    QPalette,
    QPixmap,
    QImage
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
    QFileDialog
)

class SinglePredictionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.picturebox = QLabel(self)
        self.picturebox.setParent(self)
        self.picturebox.setFixedSize(QSize(890, 500))
        self.picturebox.move(QPoint(30, 30)        )
        self.picturebox.setStyleSheet("border: 1px solid black; background-color: white;")

        self.button1 = QPushButton("Back page")
        self.button1.setParent(self)
        self.button1.setFixedSize(200, 100)
        self.button1.move(QPoint(75, 565))
        self.button1.clicked.connect(self.back_page)
        self.button1.setStyleSheet("font-size: 20px;")

        self.button2 = QPushButton("Load Image")
        self.button2.setParent(self)
        self.button2.setFixedSize(200, 100)
        self.button2.move(QPoint(375, 565))
        self.button2.clicked.connect(self.load_image)
        self.button2.setStyleSheet("font-size: 20px;")

        self.button3 = QPushButton("Predict")
        self.button3.setParent(self)
        self.button3.setFixedSize(200, 100)
        self.button3.move(QPoint(675, 565))
        self.button3.clicked.connect(self.predict)
        self.button3.setStyleSheet("font-size: 20px;")

    def back_page(self):
        self.parent().show_page(self.parent().single_or_mass_prediction_page)

    def load_image(self):
        filters = "PNG File (*.png);;JPEG File (*.jpeg);;JPG File (*.jpg)"
        self.filename, filter = QFileDialog.getOpenFileName(self, filter=filters)
        if self.filename != "":
            QPixmap(self.filename).scaled(self.picturebox.size(), Qt.KeepAspectRatio)
            self.picturebox.setPixmap(QPixmap(self.filename).scaled(self.picturebox.size(), Qt.KeepAspectRatio))
            self.picturebox.setAlignment(Qt.AlignCenter)

    def predict(self):
        image = cv2.imread(self.filename)
        face_cascade = cv2.CascadeClassifier("C:/Users/Onur/Desktop/workbench/haarcascade_frontalface_default.xml")
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image_grayscaled, 1.3, 5)

        # cropped_images = []
        # for (x, y, w, h) in faces:
        #     cropped_images.append(image[y:y+h, x:x+w])
        
        for (x, y, w, h) in faces:
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)

        image = au.image_to_pixmap(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.picturebox.setPixmap(image.scaled(self.picturebox.size(), Qt.KeepAspectRatio))
        self.picturebox.setAlignment(Qt.AlignCenter)