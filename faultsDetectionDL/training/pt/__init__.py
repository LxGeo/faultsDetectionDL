#

import os
import numpy as np
from faultsDetectionDL.utils.image_transformation import recurse_transform, images_transformations_list

SIZE_X = 128
SIZE_Y = 128
IMG_CHANNELS=3
n_classes=1 #Number of classes for segmentation
class_weights= 0.9
BATCH_SIZE=8
activation='sigmoid'
models_path="./models/ensemble/smp128_best_model2/"
model_name_template = "{model_name}_{loss_name}"


DATA_PATH_TMPL= os.path.abspath("./data/processed/refined_spline_partition_128_Site_A_C/{}")
TRAIN_PATH = DATA_PATH_TMPL.format("train")
VALID_PATH = DATA_PATH_TMPL.format("valid")