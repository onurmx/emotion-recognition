from PySide2.QtGui import (
    QColor,
    QPalette,
    QPixmap,
    QImage
)

def image_to_pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qImg)
    return pixmap