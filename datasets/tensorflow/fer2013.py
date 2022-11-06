import cv2
import tensorflow as tf
import pandas as pd
import numpy as np

def load_fer2013(filepath, size, batch_size=64, cfg_OnsuNet = False):
    df = pd.read_csv(filepath)

    x_train = df[df['Usage'] == 'Training']['pixels']
    x_val = df[df['Usage'] == 'PublicTest']['pixels']
    x_test = df[df['Usage'] == 'PrivateTest']['pixels']

    y_train = df[df['Usage'] == 'Training']['emotion'].to_numpy()
    y_val = df[df['Usage'] == 'PublicTest']['emotion'].to_numpy()
    y_test = df[df['Usage'] == 'PrivateTest']['emotion'].to_numpy()

    x_train = np.array([np.fromstring(x, sep=' ') for x in x_train])
    x_val = np.array([np.fromstring(x, sep=' ') for x in x_val])
    x_test = np.array([np.fromstring(x, sep=' ') for x in x_test])

    x_train = x_train.reshape(len(x_train), 48, 48) if cfg_OnsuNet == False else x_train.reshape(len(x_train), 48, 48, 1)
    x_val = x_val.reshape(len(x_val), 48, 48) if cfg_OnsuNet == False else x_val.reshape(len(x_val), 48, 48, 1)
    x_test = x_test.reshape(len(x_test), 48, 48) if cfg_OnsuNet == False else x_test.reshape(len(x_test), 48, 48, 1)

    if cfg_OnsuNet == False: # means convert to 3-channel image
        x_train = np.stack((x_train,)*3, axis=-1)
        x_val = np.stack((x_val,)*3, axis=-1)
        x_test = np.stack((x_test,)*3, axis=-1)

    x_train = np.array([cv2.resize(x, (size, size)) for x in x_train])
    x_val = np.array([cv2.resize(x, (size, size)) for x in x_val])
    x_test = np.array([cv2.resize(x, (size, size)) for x in x_test])

    steps_per_epoch = len(x_train) // batch_size
    validation_steps = len(x_val) // batch_size
    test_steps = len(x_test) // batch_size

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1. / 255
    )

    training_data = train_generator.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    validation_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )

    validation_data = validation_generator.flow(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=True
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )

    testing_data = test_generator.flow(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=True
    )

    return training_data, validation_data, testing_data, steps_per_epoch, validation_steps, test_steps