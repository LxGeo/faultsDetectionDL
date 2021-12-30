#

import os
import numpy as np
from faultsDetectionDL.utils.image_transformation import recurse_transform, images_transformations_list

DEVICE = 'cuda'
SIZE_X = 256
SIZE_Y = 256
IMG_CHANNELS=3
n_classes=3 #Number of classes for segmentation
class_weights= 1/np.array([0.86556702, 0.09528941, 0.0391435])
class_weights/=sum(class_weights)
BATCH_SIZE=8
activation=None#'softmax2d'
models_path="./models/buildings/trial1/"
model_name_template = "{model_name}_{loss_name}"


DATA_PATH_TMPL= os.path.abspath("../DATA_SANDBOX/processed/b_n_c_256/{}")
TRAIN_PATH = DATA_PATH_TMPL.format("train")
VALID_PATH = DATA_PATH_TMPL.format("valid")