import os
import cv2
import numpy as np
import sklearn.model_selection as skl

def np_load_kdef(filepath, image_height=224, image_width=224):
    # define labels
    labels = {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}

    # traverse directory and save image paths
    file_paths = []
    for folder, subfolders, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.lower().endswith(('.jpg')):
                file_paths.append(os.path.join(folder, filename))

    # initialize array for images and labels
    num_faces = len(file_paths)
    faces = np.zeros(shape=(num_faces, image_height, image_width, 3))
    emotions = np.zeros(shape=(num_faces))

    # load images and labels
    for file_arg, file_path in enumerate(file_paths):
        image = cv2.imread(file_path) / 255
        image = cv2.resize(image, (image_height, image_width))
        faces[file_arg] = image
        file_basename = os.path.basename(file_path)
        file_emotion = file_basename[4:6]
        try:
            emotions[file_arg] = labels[file_emotion]
        except:
            continue

    # split data into training and test sets
    x_train, x_test, y_train, y_test = skl.train_test_split(faces, emotions, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test