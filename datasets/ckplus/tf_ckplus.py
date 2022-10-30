import cv2
import numpy as np
import os
import sklearn.model_selection as skl
import tensorflow as tf

def tf_load_ckplus(filepath, image_height = 48, image_width = 48, batch_size=64, cfg_OnsuNet = True):
    num_images = 0
    for folder, subfolders, filenames in os.walk(filepath):
        for filename in filenames:
            num_images += 1

    directories = sorted(os.listdir(filepath))
    images = np.zeros(shape=(num_images, image_height, image_width, 3 if cfg_OnsuNet == False else 1))
    labels = np.zeros(shape=(num_images))

    index = 0
    for dataset in directories:
        path = os.path.join(filepath, dataset)
        class_num = directories.index(dataset)
        for file in os.listdir(path):
            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR if cfg_OnsuNet == False else cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (image_height, image_width)) if cfg_OnsuNet == False else cv2.resize(image, (image_height, image_width)).reshape(image_height, image_width, 1)
            images[index] = image
            labels[index] = class_num
            index += 1

    x_train, x_test, y_train, y_test = skl.train_test_split(images, labels, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = skl.train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    steps_per_epoch = len(x_train) // batch_size
    validation_steps = len(x_val) // batch_size
    test_steps = len(x_test) // batch_size

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
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