import pandas as pd
import numpy as np
import cv2 as cv

def load_fer2013(filepath):
    # read data
    df = pd.read_csv(filepath)

    # partition data into training, and test sets
    df_train = df[df['Usage'] == 'Training']
    df_test = df[df['Usage'] == 'PublicTest']

    # get pixels
    x_train = df_train['pixels']
    x_test = df_test['pixels']

    # get labels
    y_train = df_train['emotion']
    y_test = df_test['emotion']

    # split pixels
    x_train = np.array([np.fromstring(x, sep=' ') for x in x_train])
    x_test = np.array([np.fromstring(x, sep=' ') for x in x_test])

    # reshape pixels
    x_train = x_train.reshape(len(df_train), 48, 48)
    x_test = x_test.reshape(len(df_test), 48, 48)

    # normalize pixels
    x_train /= 255
    x_test /= 255

    # convert pixels into rgb
    x_train = np.stack((x_train,)*3, axis=-1)
    x_test = np.stack((x_test,)*3, axis=-1)

    # resize images into 224x224
    x_train = np.array([cv.resize(x, (224, 224)) for x in x_train])
    x_test = np.array([cv.resize(x, (224, 224)) for x in x_test])

    return x_train, y_train, x_test, y_test