import tensorflow as tf
import pandas as pd
import numpy as np

def tf_load_fer2013(filepath, convert_to_rgb = False, upsample = None, batch_size=128):
    # read data
    df = pd.read_csv(filepath)

    # get pixels
    x_train = df[df['Usage'] == 'Training']['pixels']
    x_test = df[df['Usage'] == 'PublicTest']['pixels']

    # get labels
    y_train = df[df['Usage'] == 'Training']['emotion'].to_numpy()
    y_test = df[df['Usage'] == 'PublicTest']['emotion'].to_numpy()

    # split pixels
    x_train = np.array([np.fromstring(x, sep=' ') for x in x_train])
    x_test = np.array([np.fromstring(x, sep=' ') for x in x_test])

    # reshape pixels
    x_train = x_train.reshape(len(x_train), 48, 48)
    x_test = x_test.reshape(len(x_test), 48, 48)

    # convert pixels into rgb
    if convert_to_rgb:
        x_train = np.stack((x_train,)*3, axis=-1)
        x_test = np.stack((x_test,)*3, axis=-1)

    # upsample images by upsample factor
    if upsample:
        x_train = np.array([x.repeat(upsample, axis=0).repeat(upsample, axis=1) for x in x_train])
        x_test = np.array([x.repeat(upsample, axis=0).repeat(upsample, axis=1) for x in x_test])

    # calculate steps per epoch and validation steps
    steps_per_epoch = len(x_train) // batch_size
    validation_steps = len(x_test) // batch_size

    # Data augmentation using Image Data Generator for train data
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

    # Data augmentation using Image Data Generator for test data
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )

    testing_data = test_generator.flow(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=True
    )

    return training_data, testing_data, steps_per_epoch, validation_steps