import os
import math
import random
import time

import cv2
import numpy as np
from shapely import affinity
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from models import unet
from utils import read_train_wkt
from utils import read_grid_sizes
from utils import read_img
from utils import get_img_scalers
from utils import get_polygons
from utils import get_training_img_ids
from config import WEIGHTS_DIR
from config import INPUT_SIZE
from config import CLASSES


def generate_mask(img, img_id, df, gs):
    x_scaler, y_scaler = get_img_scalers(img, img_id, gs)

    mask = np.zeros((len(CLASSES), img.shape[0], img.shape[1]))

    for class_type in range(1, len(CLASSES) + 1):
        polygons = get_polygons(img_id, class_type, df)
        polygons = affinity.scale(polygons,
                                  xfact=x_scaler,
                                  yfact=y_scaler,
                                  origin=(0,0,0))

        polygon_list = []
        for polygon in polygons:
            polygon_list.append(np.array(polygon.exterior.coords)
                                .reshape((-1, 1, 2))
                                .astype(int))

        cv2.fillPoly(mask[class_type - 1], polygon_list, 1)

    mask = np.transpose(mask, (1, 2, 0))

    return mask


def get_train_data(train_img_ids, df, gs):
    x_train_list = []
    y_train_list = []

    for img_id in train_img_ids:
        x_train = read_img(img_id)
        y_train = generate_mask(x_train, img_id, df, gs)
        x_train_list.append(x_train)
        y_train_list.append(y_train)

    return (x_train_list, y_train_list)


def get_patches(x_train_list, y_train_list, num):
    x_train_patches = []
    y_train_patches = []

    for idx in range(num):
        train_idx = random.randint(0, len(x_train_list) - 1)
        x_train = x_train_list[train_idx]
        y_train = y_train_list[train_idx]

        pos = (random.randint(0, x_train.shape[0] - INPUT_SIZE - 1),
               random.randint(0, x_train.shape[1] - INPUT_SIZE - 1))
        x_patch = x_train[pos[0] : pos[0] + INPUT_SIZE,
                          pos[1] : pos[1] + INPUT_SIZE]
        y_patch = y_train[pos[0] : pos[0] + INPUT_SIZE,
                          pos[1] : pos[1] + INPUT_SIZE]

        x_train_patches.append(x_patch)
        y_train_patches.append(y_patch)

    x_train_patches = np.array(x_train_patches).astype(np.float32)
    y_train_patches = np.array(y_train_patches).astype(np.float32)

    return (x_train_patches, y_train_patches)


def eval_model(model, x, y, batch_size):
    total_size = x.shape[0]
    repeats = math.ceil(total_size / batch_size)
    avg_scores = np.array([])

    for idx in range(repeats):
        start_pos = idx * batch_size
        end_pos = (idx + 1) * batch_size
        x_input = x[start_pos:end_pos]
        y_input = y[start_pos:end_pos]

        scores = model.test_on_batch(x_input, y_input)
        weighted_scores = np.array(scores) * (x_input.shape[0] / total_size)

        if idx is 0:
            avg_scores = np.array(weighted_scores)
        else:
            avg_scores = np.add(avg_scores, weighted_scores)

    return avg_scores


def save_model(model):
    if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

    output_path = '{}/unet_{}.hdf5'.format(WEIGHTS_DIR, int(time.time()))
    model.save(output_path)


def train(df,
          gs,
          batch_size=128,
          epochs=1000,
          validation_size=1000,
          test_size=1000):
    train_img_ids = get_training_img_ids(df)

    (x_train_list, y_train_list) = get_train_data(train_img_ids, df, gs)

    model = unet((INPUT_SIZE, INPUT_SIZE, x_train_list[0].shape[2]),
                 len(CLASSES))
    model.compile(optimizer=Adam(),
                  loss=binary_crossentropy,
                  metrics=['accuracy'])

    (x_val_patches, y_val_patches) = get_patches(x_train_list,
                                                 y_train_list,
                                                 validation_size)

    for epoch in range(epochs):
        (x_train_patches, y_train_patches) = get_patches(x_train_list,
                                                         y_train_list,
                                                         batch_size)
        train_scores = model.train_on_batch(x_train_patches, y_train_patches)
        val_scores = eval_model(model, x_val_patches, y_val_patches, batch_size)

        print('[Epoch {}] Loss: {} - Accuracy: {} - Validation Accuracy: {}'
              .format(epoch, train_scores[0], train_scores[1], val_scores[1]))

    (x_test_patches, y_test_patches) = get_patches(x_train_list,
                                                   y_train_list,
                                                   test_size)
    test_scores = eval_model(model, x_test_patches, y_test_patches, batch_size)

    print('Test Accuracy: {}'.format(test_scores[1]))

    save_model(model)


def main():
    df = read_train_wkt()
    gs = read_grid_sizes()

    train(df, gs)


if __name__ == '__main__':
    main()
