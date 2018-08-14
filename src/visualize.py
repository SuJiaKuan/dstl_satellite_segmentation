import sys

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
import numpy as np
from shapely import affinity

from utils import read_train_wkt
from utils import read_grid_sizes
from utils import read_img
from utils import get_img_scalers
from utils import get_training_img_ids
from utils import get_polygons
from config import CLASSES


def is_training_img(img_id, training_img_ids):
    return any(training_img_ids == img_id)


def visualize_polygons_list(polygons_list):
    fig, ax = plt.subplots(figsize=(8, 8))

    legend_patches = []

    for class_type in range(1, len(CLASSES) + 1):
        polygons = polygons_list[class_type - 1]
        color = plt.cm.Paired(class_type)

        for polygon in polygons:
            mpl_poly = Polygon(np.array(polygon.exterior),
                               color=color,
                               lw=0,
                               alpha=0.3)
            ax.add_patch(mpl_poly)

        legend_patch = Patch(color=color,
                             label=CLASSES[str(class_type)],
                             alpha=0.3)
        legend_patches.append(legend_patch)

    ax.legend(handles=legend_patches,
              loc='upper right',
              fontsize='x-small',
              framealpha=0.9)

    ax.autoscale_view()

    plt.show()


def get_polygons_list(img_id, df, gs):
    img = read_img(img_id)
    x_scaler, y_scaler = get_img_scalers(img, img_id, gs)

    polygons_list = []

    for class_type in range(1, len(CLASSES) + 1):
        polygons = get_polygons(img_id, class_type, df)
        polygons = affinity.scale(polygons,
                                  xfact=x_scaler,
                                  yfact=y_scaler,
                                  origin=(0,0,0))
        polygons_list.append(polygons)

    return polygons_list


def main(img_id):
    df = read_train_wkt()
    gs = read_grid_sizes()

    training_img_ids = get_training_img_ids(df)

    if not is_training_img(img_id, training_img_ids):
        print('The image ID "{}" is not in training set: {}'
              .format(img_id, training_img_ids))
        sys.exit(-1)
    else:
        polygons_list = get_polygons_list(img_id, df, gs)
        visualize_polygons_list(polygons_list)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python {} img_id'.format(sys.argv[0]))
        sys.exit(-1)

    img_id = sys.argv[1]

    main(img_id)
