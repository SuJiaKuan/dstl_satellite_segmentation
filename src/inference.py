import math
import sys
from collections import defaultdict

import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

from visualize import visualize_polygons_list
from utils import read_img
from utils import normalize_img
from config import CLASSES
from config import INFERENCE_THRESHOLD
from config import INPUT_SIZE


def show_raster(raster):
    import tifffile as tiff
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    tiff.imshow(raster, figure=fig, subplot=ax)
    plt.show()


def mask_to_polygons(mask, epsilon=10., min_area=10.):
    # Code from: https://michhar.github.io/masks_to_polygons_and_back/

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)

    return all_polygons


def masks_to_polygons_list(masks):
    polygons_list = []

    for idx in range(masks.shape[2]):
        polygons = mask_to_polygons(masks[:, :, idx].astype(np.uint8))
        polygons_list.append(polygons)

    return polygons_list


def inference(model_path, img_id):
    model = load_model(model_path)
    img = read_img(img_id)
    normalized = normalize_img(img)

    repeats_x = math.ceil(img.shape[0] / INPUT_SIZE)
    repeats_y = math.ceil(img.shape[1] / INPUT_SIZE)
    input_shape = (1, INPUT_SIZE, INPUT_SIZE, img.shape[2])
    result = np.zeros((img.shape[0], img.shape[1], len(CLASSES)))

    for idx_x in range(repeats_x):
        for idx_y in range(repeats_y):
            s_x = idx_x * INPUT_SIZE
            s_y = idx_y * INPUT_SIZE

            patch = normalized[s_x:s_x + INPUT_SIZE, s_y:s_y + INPUT_SIZE]
            input_img = np.zeros(input_shape)
            input_img[0, 0:patch.shape[0], 0:patch.shape[1]] = patch

            pred = model.predict(input_img)
            result[s_x:s_x + patch.shape[0], s_y:s_y + patch.shape[1]] \
                = pred[0, 0:patch.shape[0], 0:patch.shape[1]]

    result = np.where(result > INFERENCE_THRESHOLD, 1, 0)

    return result


def main(model_path, img_id):
    result = inference(model_path, img_id)
    polygons_list = masks_to_polygons_list(result)
    visualize_polygons_list(polygons_list)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} model_path img_id'.format(sys.argv[0]))
        sys.exit(-1)

    model_path = sys.argv[1]
    img_id = sys.argv[2]

    main(model_path, img_id)
