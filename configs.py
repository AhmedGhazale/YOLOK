
# dataset parameters

CLASSES = ['LT',  # 0
           'LB',  # 1
           'RT',  # 2
           'RB',  # 3
           ]

DATASET_PATH = 'goals749/'

IMAGE_SIZE = 448

# model parameters

GRID_SIZE = 14
BASE_MODEL = 'resnet50'
BOXES_PER_CELL = 1

# training parameters

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = .001
SAVE_DIR = 'resnet50_14'

# predict parameters

MODEL_PATH = 'resnet50_14/resnet50.pth'

