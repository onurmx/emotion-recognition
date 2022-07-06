import pandas as pd
import numpy as np

def load_fer2013(filepath):
    # read data
    df = pd.read_csv(filepath)

    # get pixels
    x_train = df[df['Usage'] == 'Training']['pixels']
    x_test = df[df['Usage'] == 'PublicTest']['pixels']

    # get labels
    y_train = df[df['Usage'] == 'Training']['emotion']
    y_test = df[df['Usage'] == 'PublicTest']['emotion']

    # destroy dataframe
    del df

    # split pixels
    x_train = np.array([np.fromstring(x, sep=' ') for x in x_train])
    x_test = np.array([np.fromstring(x, sep=' ') for x in x_test])

    # reshape pixels
    x_train = x_train.reshape(len(x_train), 48, 48)
    x_test = x_test.reshape(len(x_test), 48, 48)

    # normalize pixels
    x_train /= 255
    x_test /= 255

    # convert pixels into rgb
    x_train = np.stack((x_train,)*3, axis=-1)
    x_test = np.stack((x_test,)*3, axis=-1)

    # upsample images into 240x240
    x_train = np.array([x.repeat(5, axis=0).repeat(5, axis=1) for x in x_train])
    x_test = np.array([x.repeat(5, axis=0).repeat(5, axis=1) for x in x_test])

    return x_train, y_train, x_test, y_test