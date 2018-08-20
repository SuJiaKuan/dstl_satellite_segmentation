import os

import pandas as pd
import tifffile as tiff
import numpy as np
from shapely.wkt import loads as wkt_loads

from config import INPUT_DIR
from config import TRAIN_CSV_PATH
from config import GRID_SIZES_CSV_PATH


def read_train_wkt():
    return pd.read_csv(TRAIN_CSV_PATH)


def read_grid_sizes():
    return pd.read_csv(GRID_SIZES_CSV_PATH,
                       names=['ImageId', 'Xmax', 'Ymin'],
                       skiprows=1)


def read_img(img_id):
    img_path = os.path.join(INPUT_DIR, 'three_band/{}.tif'.format(img_id))
    img = tiff.imread(img_path)
    img = np.rollaxis(img, 0, 3)

    return img


def normalize_img(img):
    result = np.zeros_like(img, dtype=np.float32)

    for idx in range(img.shape[2]):
        minimum = float(np.amin(img[:, :, idx]))
        maxinum = float(np.amax(img[:, :, idx]))
        normalized = (img[:, :, idx] - minimum) / (maxinum - minimum)
        result[:, :, idx] = normalized

    return result


def get_img_scalers(img, img_id, gs):
    grid_size = gs[gs.ImageId == img_id]
    x_max = grid_size.Xmax.values[0]
    y_min = grid_size.Ymin.values[0]

    width, height = img.shape[0:2]

    width = float(width) * width / (width + 1)
    height = float(height) * height / (height + 1)

    x_scaler = width / x_max
    y_scaler = height / y_min

    return (x_scaler, y_scaler)


def get_training_img_ids(df):
    return df.ImageId.unique()


def get_polygons(img_id, class_type, df):
    class_data = df[(df.ImageId == img_id) & (df.ClassType == class_type)]
    polygons = wkt_loads(class_data.MultipolygonWKT.values[0])

    return polygons
