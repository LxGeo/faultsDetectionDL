#

import os
import numpy as np
from faultsDetectionDL.utils.image_transformation import recurse_transform, images_transformations_list

SIZE_X = 256
SIZE_Y = 256
IMG_CHANNELS=4
n_classes=1 #Number of classes for segmentation
class_weights= [0.95,0.5]
BATCH_SIZE=16
activation='sigmoid'
models_path="./models/ensemble/smp/"
model_name_template = "{model_name}_{loss_name}"


DATA_PATH_TMPL="./data/processed/refined_spline_partition_256_Site_A_C/{}"
TRAIN_PATH = DATA_PATH_TMPL.format("train")
VALID_PATH = DATA_PATH_TMPL.format("valid")