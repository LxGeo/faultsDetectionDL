import os
import json
import yaml
from types import SimpleNamespace
from pathlib import Path
from faultsDetectionDL import _logger


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)

def load_config(config_file_path: str):
    
    extension = Path(config_file_path).suffix.lower()
    
    if extension == ".json":
        with open(config_file_path) as f:
            return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    if extension == ".yaml":
        with open(config_file_path) as f:
            dct = yaml.load(f, Loader)
            return json.loads(json.dumps(dct), object_hook=lambda d: SimpleNamespace(**d))
    
    _logger.error(f"Config file with format {extension} is not supported")
    raise Exception("Config file format is unknown!")




import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def load_cfg_trainer_params(cfg):
    """
    """

    callbacks = []
    # Checkpoint callback loading
    if cfg.MODEL_CHECKPOINT_CALLBACK.USE:
        if cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS is not None:
            os.makedirs(cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS.dirpath, exist_ok=True)
            save_params = vars(cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS)
            callbacks.append(
                ModelCheckpoint(**save_params)
            )
        else:
            raise Exception("ModelCheckpoint parameters missing!")   
    
    # EarlyStopping callback loading
    if cfg.EARLY_STOPPING.USE:
        if cfg.EARLY_STOPPING.PARAMS is not None:
            stopping_params = vars(cfg.EARLY_STOPPING.PARAMS)
            callbacks.append(
                EarlyStopping(**stopping_params)
            )
        else:
            raise Exception("EarlyStopping parameters missing!")   

    # Logger loading
    logger=None
    if cfg.TENSORBOARD_LOGGER.USE:
        os.makedirs(cfg.TENSORBOARD_LOGGER.LOG_PATH, exist_ok=True)
        logger = TensorBoardLogger(cfg.TENSORBOARD_LOGGER.LOG_PATH, cfg.TENSORBOARD_LOGGER.NAME)

    lightning_params = vars(cfg.LIGHTNING_PARAMS)

    lightning_params.update(
        {
            "callbacks":callbacks,
            "logger":logger
        }
    )

    return lightning_params

from LxGeoPyLibs.dataset import DATASETS_REGISTERY
from LxGeoPyLibs.dataset.multi_dataset import MultiDatasets

def load_datasets_from_cfg(dataset_cfg, **kwargs):    
    loaded_datasets = []
    for c_dataset in dataset_cfg.DATASETS:        
        fused_args = vars(c_dataset.ARGS); fused_args.update(kwargs)
        loaded_datasets.append(
            DATASETS_REGISTERY.get(c_dataset.DATASET_LOADER)( **fused_args)
        )
    return MultiDatasets(loaded_datasets)

