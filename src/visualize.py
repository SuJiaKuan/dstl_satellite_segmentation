import sys

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
import numpy as np
from shapely import affinity
from shapely.wkt import loads as wkt_loads

from utils import read_train_wkt
from utils import read_grid_sizes
from utils import read_img
from utils import get_img_scalers
from utils import get_training_img_ids


CLASSES = {
    '1': 'Buildings',
    '2': 'Misc. Manmade structures ',
    '3': 'Road',
    '4': 'Track',
    '5': 'Trees',
    '6': 'Crops',
    '7': 'Waterway',
    '8': 'Standing Water',
    '9': 'Vehicle Large',
    '10': 'Vehicle Small',
}


def is_training_img(img_id, training_img_ids):
    return any(training_img_ids == img_id)


def visualize_img(img_id, df, gs):
    img = read_img(img_id)
    x_scaler, y_scaler = get_img_scalers(img, img_id, gs)

    fig, ax = plt.subplots(figsize=(8, 8))

    legend_patches = []
    for class_idx in range(1, len(CLASSES) + 1):
        class_data = df[(df.ImageId == img_id) & (df.ClassType == class_idx)]
        polygons = wkt_loads(class_data.MultipolygonWKT.values[0])
        polygons = affinity.scale(polygons,
                                  xfact=x_scaler,
                                  yfact=y_scaler,
                                  origin=(0,0,0))

        color = plt.cm.Paired(class_idx)

        for polygon in polygons:
            mpl_poly = Polygon(np.array(polygon.exterior),
                               color=color,
                               lw=0,
                               alpha=0.3)
            ax.add_patch(mpl_poly)

        legend_patch = Patch(color=color, label=CLASSES[str(class_idx)], alpha=0.3)
        legend_patches.append(legend_patch)

    ax.set_title(img_id)
    ax.legend(handles=legend_patches,
              loc='upper right',
              fontsize='x-small',
              framealpha=0.9)

    ax.autoscale_view()

    plt.show()


def main(img_id):
    df = read_train_wkt()
    gs = read_grid_sizes()

    training_img_ids = get_training_img_ids(df)

    if not is_training_img(img_id, training_img_ids):
        print('The image ID "{}" is not in training set: {}'
              .format(img_id, training_img_ids))
        sys.exit(-1)
    else:
        visualize_img(img_id, df, gs)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python {} img_id'.format(sys.argv[0]))
        sys.exit(-1)

    img_id = sys.argv[1]

    main(img_id)
