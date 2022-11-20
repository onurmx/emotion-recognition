import cv2
import tensorflow as tf
import torch
import torchvision

from models.tensorflow import (
    resnet as resnet_tensorflow,
    vgg as vgg_tensorflow,
    onsunet as onsunet_tensorflow,
)

from datasets.tensorflow import (
    ckplus as ckplus_tensorflow,
    fer2013 as fer2013_tensorflow,
    kdef as kdef_tensorflow,
)

from models.pytorch import (
    resnet as resnet_pytorch,
    vgg as vgg_pytorch,
    onsunet as onsunet_pytorch,
)

from datasets.pytorch import (
    ckplus as ckplus_pytorch,
    fer2013 as fer2013_pytorch,
    kdef as kdef_pytorch,
)

from utils.pytorch import (
    device_management as dm,
    train as train_pytorch,
)

from PySide2.QtGui import (
    QPixmap,
    QImage
)

def trainer(backend, model, dataset, epochs, lr, factor, patience, batch_size, workdir):
    if backend == "tensorflow":
        return NotImplementedError
    elif backend == "pytorch":
        device = dm.get_default_device()

        if model == "resnet":
            net = dm.to_device(resnet_pytorch.resnet50(img_channels=3, num_classes=7), device)
        elif model == "vgg":
            net = dm.to_device(vgg_pytorch.VGG16(num_classes=7), device)
        elif model == "onsunet":
            net = dm.to_device(onsunet_pytorch.Onsunet(num_classes=7), device)

        if dataset == "fer2013":
            train_dl, valid_dl, test_dl = fer2013_pytorch.load_fer2013(
                workdir + "/" + dataset + "/fer2013.csv",
                device,
                48 if model == "onsunet" else 224,
                batch_size=batch_size,
                cfg_OnsuNet= True if model == "onsunet" else False
            )
        elif dataset == "ckplus":
            train_dl, valid_dl, test_dl = ckplus_pytorch.load_ckplus(
                workdir + "/training_datas/" + dataset,
                device,
                48 if model == "onsunet" else 224,
                batch_size=batch_size,
                cfg_OnsuNet= True if model == "onsunet" else False
            )
        elif dataset == "kdef":
            train_dl, valid_dl, test_dl = kdef_pytorch.load_kdef(
                workdir + "/training_datas/" + dataset,
                device,
                48 if model == "onsunet" else 224,
                batch_size=batch_size,
                cfg_OnsuNet= True if model == "onsunet" else False
            )
        
        train_pytorch.fit(
            epochs=epochs,
            lr=lr,
            model=net,
            train_loader=train_dl,
            val_loader=valid_dl,
            factor=factor,
            patience=patience,
            weight_decay=0,
            opt_func=torch.optim.Adam
        )

        return net


def prediction_generator(image, backend, model, dataset, workdir):
    if backend == "tensorflow":
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if model != "onsunet" else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_image = cv2.resize(target_image, (224, 224)) if model != "onsunet" else cv2.resize(target_image, (48, 48))
        target_image = target_image.reshape(1,224,224,3) if model != "onsunet" else target_image.reshape(1,48,48,1)
        net = tf.keras.models.load_model(workdir + "/trained_models/tensorflow/" + model + "/" + dataset + "/model.h5")
        predictions = net.predict(target_image, verbose=0)
        return fer2013_label_translator(tf.argmax(predictions[0]).numpy())
    elif backend == "pytorch":
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_image = cv2.resize(target_image, (224, 224)) if model != "onsunet" else cv2.resize(target_image, (48, 48))
        target_image = torchvision.transforms.ToPILImage()(target_image)
        if model == "onsunet":
            target_image = torchvision.transforms.Grayscale(num_output_channels=1)(target_image)
        target_image = torchvision.transforms.ToTensor()(target_image)
        if model == "resnet":
            net = resnet_pytorch.resnet50(img_channels=3, num_classes=7)
        elif model == "vgg":
            net = vgg_pytorch.VGG16(num_classes=7)
        elif model == "onsunet":
            net = onsunet_pytorch.Onsunet(num_classes=7)
        net.load_state_dict(torch.load(workdir + "/trained_models/pytorch/" + model + "/" + dataset + "/model.pt", map_location=torch.device('cpu')))
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