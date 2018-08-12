import os


INPUT_DIR = '../satellite-segmantation'
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, 'train_wkt_v4.csv')
GRID_SIZES_CSV_PATH = os.path.join(INPUT_DIR, 'grid_sizes.csv')

WEIGHTS_DIR = '../weights'

INPUT_SIZE = 112

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
