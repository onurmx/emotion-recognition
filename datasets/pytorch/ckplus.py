import cv2
import numpy as np
import os
import sklearn.model_selection as skl
import torch
import torchvision
import utils.pytorch.device_management as pdm

class CKPLUS(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images=images
        self.labels=labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = image.astype(np.uint8)
        label = label.astype(np.uint8)

        if self.transforms:
            image = self.transforms(image)
        return image, label

def load_ckplus(filepath, device, size, batch_size=64, cfg_OnsuNet = False):
    num_images = 0
    for folder, subfolders, filenames in os.walk(filepath):
        for filename in filenames:
            num_images += 1

    directories = sorted(os.listdir(filepath))
    images = np.zeros(shape=(num_images, size, size, 3 if cfg_OnsuNet == False else 1))
    labels = np.zeros(shape=(num_images))

    index = 0
    for dataset in directories:
        path = os.path.join(filepath, dataset)
        class_num = directories.index(dataset)
        for file in os.listdir(path):
            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR if cfg_OnsuNet == False else cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (size, size)) if cfg_OnsuNet == False else cv2.resize(image, (size, size)).reshape(size, size, 1)
            images[index] = image
            labels[index] = class_num
            index += 1

    x_train, x_test, y_train, y_test = skl.train_test_split(images, labels, test_size=0.2, random_state=1)
    x_train, x_valid, y_train, y_valid = skl.train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    train_transformations = torchvision.transforms.Compose([   
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor()
    ])
    valid_transformations = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor()
    ])

    train_ds = CKPLUS(x_train, y_train, train_transformations)
    valid_ds = CKPLUS(x_valid, y_valid,  valid_transformations)
    test_ds = CKPLUS(x_test, y_test, valid_transformations)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size, num_workers=2, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size, num_workers=2, pin_memory=True)

    torch.cuda.empty_cache()

    train_dl = pdm.DeviceDataLoader(train_dl, device)
    valid_dl = pdm.DeviceDataLoader(valid_dl, device)
    test_dl = pdm.DeviceDataLoader(test_dl, device)

    return train_dl, valid_dl, test_dl