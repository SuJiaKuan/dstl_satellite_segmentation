import os
import random
import time

import cv2
import numpy as np
from shapely import affinity
from keras.callbacks import ModelCheckpoint
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

    # for img_id in train_img_ids:
    for img_id in train_img_ids[0:2]:
        x_train = read_img(img_id)
        y_train = generate_mask(x_train, img_id, df, gs)
        x_train_list.append(x_train)
        y_train_list.append(y_train)

    return (x_train_list, y_train_list)


def get_patches(x_train_list, y_train_list, num=10000):
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

    x_train_patches = np.array(x_train_patches)
    y_train_patches = np.array(y_train_patches)

    return (x_train_patches, y_train_patches)


def get_model_checkpoint():
    if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

    path_to_ckpt = '{}/unet_{}.hdf5'.format(WEIGHTS_DIR, int(time.time()))
    model_checkpoint = ModelCheckpoint(path_to_ckpt,
                                       monitor='loss',
                                       save_best_only=True)

    return model_checkpoint


def train(df, gs):
    train_img_ids = get_training_img_ids(df)

    (x_train_list, y_train_list) = get_train_data(train_img_ids, df, gs)
    (x_train_patches, y_train_patches) = get_patches(x_train_list, y_train_list)

    model = unet((INPUT_SIZE, INPUT_SIZE, x_train_patches[0].shape[2]),
                 len(CLASSES))
    model.compile(optimizer=Adam(),
                  loss=binary_crossentropy,
                  metrics=['accuracy'])
    model.fit(x_train_patches,
              y_train_patches,
              batch_size=128,
              epochs=100,
              shuffle=True,
              callbacks=[get_model_checkpoint()])


def main():
    df = read_train_wkt()
    gs = read_grid_sizes()

    train(df, gs)


if __name__ == '__main__':
    main()
