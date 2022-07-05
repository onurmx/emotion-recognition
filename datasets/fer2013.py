import pandas as pd
import numpy as np
import cv2 as cv

def load_fer2013(filepath):
    df = pd.read_csv(filepath)
    df_train = df[df['Usage'] == 'Training']
    df_test = df[df['Usage'] == 'PublicTest']

    x_train = []
    x_test = []
    x_train_tmp = []
    x_test_tmp = []

    for image_pixels_string in df_train.iloc[0:,1]:
        x_train.append(np.asarray(image_pixels_string.split(' '), dtype=np.uint8).reshape(48,48))

    for image_pixels_string in df_test.iloc[0:,1]:
        x_test.append(np.asarray(image_pixels_string.split(' '), dtype=np.uint8).reshape(48,48))

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    for image in x_train:
        tmp = []
        for height in image:
            for pixel in height:
                tmp.append(np.asarray([pixel,pixel,pixel]))
        x_train_tmp.append(cv.resize(np.asarray(tmp).reshape(48,48,3),(224,224)))

    for image in x_test:
        tmp = []
        for height in image:
            for pixel in height:
                tmp.append(np.asarray([pixel,pixel,pixel]))
        x_test_tmp.append(cv.resize(np.asarray(tmp).reshape(48,48,3),(224,224)))

    x_train = np.asarray(x_train_tmp)
    x_train_tmp = []
    x_test = np.asarray(x_test_tmp)
    x_test_tmp = []

    y_train = df_train.iloc[0:,0].values
    y_test = df_test.iloc[0:,0].values

    return x_train, y_train, x_test, y_test