import cv2
import tensorflow as tf
import torch
import torchvision

from models.pytorch import (
    resnet as resnet_pytorch,
    vgg as vgg_pytorch,
    onsunet as onsunet_pytorch,
)

from models.tensorflow import (
    resnet as resnet_tensorflow,
    vgg as vgg_tensorflow,
    onsunet as onsunet_tensorflow,
)

from PySide2.QtGui import (
    QColor,
    QPalette,
    QPixmap,
    QImage,
    QTextCursor
)

class OutLog:
    def __init__(self, edit):
        self.edit = edit

    def write(self, m):
        self.edit.moveCursor(QTextCursor.End)
        self.edit.insertPlainText(m)

    def flush(self):
        pass

def image_to_pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qImg)
    return pixmap

def get_faces(image, workdir):
    face_cascade = cv2.CascadeClassifier(workdir + "/haarcascade/haarcascade_frontalface_default.xml")
    image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(image_grayscaled, 1.3, 5)

def prediction_generator(image, backend, model, dataset, workdir):
    if backend == "Tensorflow":
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if model != "Onsunet" else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_image = cv2.resize(target_image, (224, 224)) if model != "Onsunet" else cv2.resize(target_image, (48, 48))
        target_image = target_image.reshape(1,224,224,3) if model != "Onsunet" else target_image.reshape(1,48,48,1)
        net = tf.keras.models.load_model(workdir + "/trained_models/tensorflow/" + model.lower() + "/" + dataset.lower() + "/model.h5")
        predictions = net.predict(target_image, verbose=0)
        return fer2013_label_translator(tf.argmax(predictions[0]).numpy())
    elif backend == "PyTorch":
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_image = cv2.resize(target_image, (224, 224)) if model != "Onsunet" else cv2.resize(target_image, (48, 48))
        target_image = torchvision.transforms.ToPILImage()(target_image)
        if model == "Onsunet":
            target_image = torchvision.transforms.Grayscale(num_output_channels=1)(target_image)
        target_image = torchvision.transforms.ToTensor()(target_image)
        if model == "Resnet":
            net = resnet_pytorch.resnet50(img_channels=3, num_classes=7)
        elif model == "VGG":
            net = vgg_pytorch.VGG16(num_classes=7)
        elif model == "Onsunet":
            net = onsunet_pytorch.Onsunet(num_classes=7)
        net.load_state_dict(torch.load(workdir + "/trained_models/pytorch/" + model.lower() + "/" + dataset.lower() + "/model.pt", map_location=torch.device('cpu')))
        net.eval()
        prediction = net(target_image.unsqueeze(0))
        emotion_label = torch.argmax(prediction)
        emotion_label = emotion_label.item()
        emotion_label = fer2013_label_translator(emotion_label)
        return emotion_label

def fer2013_label_translator(emotion_label):
    if emotion_label == 0:
        return "Angry"
    elif emotion_label == 1:
        return "Disgust"
    elif emotion_label == 2:
        return "Fear"
    elif emotion_label == 3:
        return "Happy"
    elif emotion_label == 4:
        return "Sad"
    elif emotion_label == 5:
        return "Surprise"
    elif emotion_label == 6:
        return "Neutral"
