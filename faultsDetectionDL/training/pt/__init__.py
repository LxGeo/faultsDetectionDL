#

import os
import numpy as np
from faultsDetectionDL.utils.image_transformation import recurse_transform, images_transformations_list

DEVICE = 'cuda'
SIZE_X = 256
SIZE_Y = 256
IMG_CHANNELS=3
n_classes=3 #Number of classes for segmentation
class_weights= [0.6, 0.3, 0.1]
BATCH_SIZE=4
activation='softmax'
models_path="./models/buildings/trial1/"
model_name_template = "{model_name}_{loss_name}"


DATA_PATH_TMPL= os.path.abspath("./data/processed/b_n_c_256/{}")
TRAIN_PATH = DATA_PATH_TMPL.format("train")
VALID_PATH = DATA_PATH_TMPL.format("valid")