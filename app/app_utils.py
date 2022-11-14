import cv2
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
    QImage
)

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
        if model == "Resnet":
            if dataset == "FER2013":
                return NotImplementedError
            elif dataset == "CK+":
                return NotImplementedError
            elif dataset == "KDEF": 
                return NotImplementedError
        elif model == "VGG":
            if dataset == "FER2013":
                return NotImplementedError
            elif dataset == "CK+":
                return NotImplementedError
            elif dataset == "KDEF": 
                return NotImplementedError
        elif model == "Onsunet":
            if dataset == "FER2013":
                return NotImplementedError
            elif dataset == "CK+":
                return NotImplementedError
            elif dataset == "KDEF": 
                return NotImplementedError
    elif backend == "PyTorch":
        if model == "Resnet":
            if dataset == "FER2013":
                model = resnet_pytorch.resnet50(img_channels=3, num_classes=7)
                model.load_state_dict(torch.load(workdir + "/trained_models/" + backend.lower() + "/" + model.lower() + "/" + dataset.lower() + "/model.pt", map_location=torch.device('cpu')))
                model.eval()
                target_image = cv2.resize(image, (224, 224))
                target_image = torchvision.transforms.ToPILImage()(target_image)
                target_image = torchvision.transforms.ToTensor()(target_image)
                prediction = model(target_image.unsqueeze(0))
                emotion_label = torch.argmax(prediction)
                emotion_label = emotion_label.item()
                emotion_label = fer2013_label_translator(emotion_label)
                return emotion_label
            elif dataset == "CK+":
                return NotImplementedError
            elif dataset == "KDEF": 
                return NotImplementedError
        elif model == "VGG":
            if dataset == "FER2013":
                model = vgg_pytorch.VGG16(num_classes=7)
                model.load_state_dict(torch.load(workdir + "/trained_models/pytorch/vgg/fer2013/model.pt", map_location=torch.device('cpu')))
                model.eval()
                target_image = cv2.resize(image, (224, 224))
                target_image = torchvision.transforms.ToPILImage()(target_image)
                target_image = torchvision.transforms.ToTensor()(target_image)
                prediction = model(target_image.unsqueeze(0))
                emotion_label = torch.argmax(prediction)
                emotion_label = emotion_label.item()
                emotion_label = fer2013_label_translator(emotion_label)
                return emotion_label
            elif dataset == "CK+":
                return NotImplementedError
            elif dataset == "KDEF": 
                return NotImplementedError
        elif model == "Onsunet":
            if dataset == "FER2013":
                return NotImplementedError
            elif dataset == "CK+":
                return NotImplementedError
            elif dataset == "KDEF": 
                return NotImplementedError

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
