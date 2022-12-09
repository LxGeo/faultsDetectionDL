
import os
import click
import torch
from faultsDetectionDL.training.ptl.lightning_segmentation_model import lightningSegModel
from faultsDetectionDL.utils.config_loader import load_config, load_cfg_trainer_params, load_datasets_from_cfg
from LxGeoPyLibs.dataset.multi_dataset import MultiDatasets
from LxGeoPyLibs.dataset.specific_datasets.rasterized_vector_with_reference import VectorWithRefDataset
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import numpy as np

def worker_init_fn(worker_id):
    import rasterio, fiona
    from LxGeoPyLibs.dataset.raster_dataset import rasters_map
    from LxGeoPyLibs.dataset.vector_dataset import vectors_map
    w_multi_dst = torch.utils.data.get_worker_info()
    for c_dst in w_multi_dst.dataset.datasets:
        rasters_map.update({c_dst.image_dataset.image_path: rasterio.open(c_dst.image_dataset.image_path)})
        vectors_map.update({c_dst.vector_dataset.vector_path: fiona.open(c_dst.vector_dataset.vector_path)})
    

@click.command()
@click.argument('train_data_dir', type=click.Path(exists=True))
@click.argument('val_data_dir', type=click.Path(exists=True))
@click.argument('ckpt_dir', type=click.Path(exists=False))
@click.argument('log_dir', type=click.Path(exists=False))
@click.option('--custom_model_cfg', required=True, type=click.Path(exists=True))
@click.option('--custom_training_cfg', required=True, type=click.Path(exists=True))
@click.option('--resume_ckpt', required=False, type=click.Path(exists=True))
def main(train_data_dir, val_data_dir, ckpt_dir, log_dir, custom_model_cfg, custom_training_cfg, resume_ckpt):
    """_summary_

    Args:
        train_data_dir (_type_): _description_
        val_data_dir (_type_): _description_
        ckpt_dir (_type_): _description_
        log_dir (_type_): _description_
        custom_model_cfg (_type_): _description_
        custom_training_cfg (_type_): _description_
        resume_ckpt (_type_): _description_
    """
    
    model_cfg = load_config(custom_model_cfg)
    training_cfg = load_config(custom_training_cfg)
    train_data_cfg = load_config(train_data_dir)
    val_data_cfg = load_config(val_data_dir)
    
    light_model = lightningSegModel(**vars(model_cfg))
    #light_model = torch.compile(light_model)
    
    training_cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS.dirpath = ckpt_dir
    training_cfg.TENSORBOARD_LOGGER.LOG_PATH = log_dir
    
    training_params = load_cfg_trainer_params(training_cfg)
    
    preprocessing_fn = lambda x, forward=True: light_model.get_preprocessing_fn()( x[:3] )
    def preprocessing_fn(x, forward=True):
        if forward:
            return light_model.get_preprocessing_fn()( x[:3] )
        else:
            mean = np.array(light_model.get_preprocessing_fn().keywords["mean"])
            std = np.array(light_model.get_preprocessing_fn().keywords["std"])
            x = x * std[:,None, None]
            x = x + mean[:,None, None]
            return x
            

    from LxGeoPyLibs.vision.image_transformation import Trans_Identity, Trans_Rot90, Trans_Rot180, Trans_Rot270, Trans_fliplr, Trans_gaussian_noise, Trans_gamma_adjust, Trans_equal_hist
    augmentation_transforms = [Trans_Identity(), Trans_Rot90(), Trans_Rot180(), Trans_Rot270(), Trans_fliplr(), Trans_gaussian_noise(), Trans_gamma_adjust(), Trans_equal_hist()]
    
    train_dataset= load_datasets_from_cfg(train_data_cfg, preprocessing=preprocessing_fn, augmentation_transforms=augmentation_transforms)
    valid_dataset= load_datasets_from_cfg(val_data_cfg, preprocessing=preprocessing_fn, augmentation_transforms=augmentation_transforms)
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_cfg.DATA.BATCH_SIZE, num_workers=16, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=training_cfg.DATA.BATCH_SIZE, num_workers=16, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn)
    
    
    light_model = light_model.cuda()
    light_model.sample_val = next(iter(valid_dataloader))
    light_model.sample_train = next(iter(train_dataloader))

    # Check if resume training
    if resume_ckpt:
        if os.path.isfile(resume_ckpt):
            training_params["resume_from_checkpoint"] = resume_ckpt
        else:
            list_of_ckpt = [os.path.join(resume_ckpt, f) for f in os.listdir(resume_ckpt)]
            training_params["resume_from_checkpoint"] = max(list_of_ckpt, key=os.path.getctime)
    
    trainer = pl.Trainer(**training_params)

    if (training_params["auto_scale_batch_size"] or training_params["auto_lr_find"]):
        trainer.tune(light_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    else:
        trainer.fit(light_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()