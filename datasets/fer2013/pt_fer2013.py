import torch
import torchvision
import pandas as pd
import numpy as np

class FileReader:
    def __init__(self, csv_file_name):
        self._csv_file_name = csv_file_name
    def read(self):
        self._data = pd.read_csv(self._csv_file_name)

class Data:
    def __init__(self, data):
        self._x_train, self._y_train = [],  []
        self._x_test, self._y_test = [], []
        self._x_valid, self._y_valid = [], []

        for xdx, x in enumerate(data.values):
            pixels = []
            label = None
            for idx, i in enumerate(x[1].split(' ')):
                pixels.append(int(i))
            pixels = np.array(pixels).reshape((1, 48, 48))

            if x[2] == 'Training':
                self._x_train.append(pixels)
                self._y_train.append(int(x[0]))
            elif x[2] == 'PublicTest':
                self._x_test.append(pixels)
                self._y_test.append(int(x[0]))
            else:
                self._x_valid.append(pixels)
                self._y_valid.append(int(x[0]))
        self._x_train, self._y_train = np.array(self._x_train).reshape((len(self._x_train), 1, 48, 48)),\
            np.array(self._y_train, dtype=np.int64)
        self._x_test, self._y_test = np.array(self._x_test).reshape((len(self._x_test), 1, 48, 48)),\
            np.array(self._y_test, dtype=np.int64)
        self._x_valid, self._y_valid = np.array(self._x_valid).reshape((len(self._x_valid), 1, 48, 48)),\
            np.array(self._y_valid, dtype=np.int64)

class FER2013Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.transform = transform
        self._X = X
        self._y = Y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        if self.transform:
          return {'inputs': self.transform(self._X[idx]), 'labels': self._y[idx]}
        return {'inputs': self._X[idx], 'labels': self._y[idx]}

def tf_load_fer2013(filepath):
    file_reader = FileReader(filepath)
    file_reader.read()

    data = Data(file_reader._data)
    data._x_train = np.asarray(data._x_train, dtype=np.float64)
    data._x_train -= np.mean(data._x_train, axis=0)

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(6),
        torchvision.transforms.ColorJitter()
    ])

    train_set = FER2013Dataset(
        data._x_train, data._y_train, transform=preprocess)
    test_set = FER2013Dataset(data._x_valid, data._y_valid)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, num_workers=0, shuffle=False)

    return train_loader, test_loader
