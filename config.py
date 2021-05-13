import os
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "D:\\Licenta\\licenta\\data"
VISUAL_ACUITY_DATA_PATH = "./data_layer/DMLVAVcuID.xls"

# TODO: save splitted data into folders or create a database?
# initialize the base path to the *new* directory that will contain
# the images after computing the training and testing split
BASE_PATH = "data_new"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.7
# the amount of validation data
VAL_SPLIT = 0.2

# augmented data paths
AUG_DATASET = "augmented_data"
