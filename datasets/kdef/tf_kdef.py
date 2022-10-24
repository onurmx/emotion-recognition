import cv2
import numpy as np
import os
import sklearn.model_selection as skl
import tensorflow as tf

def tf_load_kdef(filepath, image_height=224, image_width=224, batch_size=64):
    labels = {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}

    file_paths = []
    for folder, subfolders, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.lower().endswith(('.jpg')):
                file_paths.append(os.path.join(folder, filename))

    num_images = len(file_paths)
    images = np.zeros(shape=(num_images, image_height, image_width, 3))
    labels = np.zeros(shape=(num_images))

    for file_arg, file_path in enumerate(file_paths):
        image = cv2.imread(file_path)
        image = cv2.resize(image, (image_height, image_width))
        images[file_arg] = image
        file_basename = os.path.basename(file_path)
        file_emotion = file_basename[4:6]
        try:
            labels[file_arg] = labels[file_emotion]
        except:
            continue

    x_train, x_test, y_train, y_test = skl.train_test_split(images, labels, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = skl.train_test_split(x_train, y_train, test_size=0.25, random_state=1)

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