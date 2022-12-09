#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:30:25 2022

@author: cherif
"""

#%% Cell 1
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import io
from torchvision import models



##### Inpaint dataset ####
#%% Cell dataset

def get_filtered_files(folder_to_search, include_extension=(".tif",) ):
    """
    """
    all_files = os.listdir(folder_to_search)
    filtered_files = filter( lambda x: any([c_ext.lower() in x.lower() for c_ext in include_extension]), all_files )
    filtered_files_full_path = map( lambda x: os.path.join(folder_to_search, x), filtered_files )
    return list(filtered_files_full_path)

import random 
from torch.utils.data import Dataset
from skimage.io import imread

class InpaintDataset(Dataset):
    
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    
    def __init__(self, data_dir, augmentation_transforms=None,preprocessing=None, img_bands=3, out_channels=3,
                 image_sub_dir="image", include_extension=(".tif",), shuffle=True, walker_length=40000):
        
        self.img_dir = data_dir
        assert os.path.isdir(data_dir), "Can't find path {}".format(data_dir)
        self.image_dir = os.path.join(data_dir, image_sub_dir)
        
        self.images_paths = get_filtered_files(self.image_dir, include_extension=include_extension)
        
        self.walker_length = walker_length
        
        if shuffle:
            random.shuffle(self.images_paths)
        
        self.non_augmented_images_count = len(self.images_paths)
        
        self.augmentation_transforms = augmentation_transforms
        self.augmented_count = len(augmentation_transforms) * self.non_augmented_images_count
        
        self.img_bands=img_bands
        self.out_channels = out_channels
        #ENcoder related preprocessing
        self.preprocessing=preprocessing
        

    def __len__(self):
        return self.augmented_count * 10
    
    def generate_mask(self, img):
        H = img.shape[0]
        W = img.shape[1]
        canvas = np.ones((H, W)).astype("i")
        x = random.randint(0, H - 1)
        y = random.randint(0, W - 1)
        x_list = []
        y_list = []
        for i in range(self.walker_length):
            r = random.randint(0, len(self.action_list) - 1)
            x = np.clip(x + self.action_list[r][0], a_min=0, a_max=H - 1)
            y = np.clip(y + self.action_list[r][1], a_min=0, a_max=W - 1)
            x_list.append(x)
            y_list.append(y)
        canvas[np.array(x_list), np.array(y_list)] = 0
        return canvas
        
    
    def __getitem__(self, idx):
        
        image_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        image_idx = image_idx % len(self.images_paths)
        
        img_path = self.images_paths[image_idx]
        
        img = imread(img_path)[:,:,0:self.img_bands]
        mask = self.generate_mask(img)
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, mask = c_trans.apply_transformation(img, mask)
        
        if self.preprocessing:
            #img[:,:,0:3] = self.preprocessing(img[:,:,0:3])
            img = self.preprocessing(img)[:,:,0:self.img_bands]
        
        mask = np.expand_dims(mask, -1)
        
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.copy()).permute(2, 0, 1)
        return (img * mask), mask, img

#### Inpainting datamodule ###
#%% Cell datamodule
import os
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity

class InpaintDataModule(pl.LightningDataModule):

    def setup(self, dataset_path, encoder_name, encoder_weights, in_channels, out_channels, batch_size):
                  
        self.train_dataset_path=os.path.join(dataset_path, 'train')
        self.valid_dataset_path=os.path.join(dataset_path, 'valid')
        self.batch_size=batch_size
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
        self.train_dataset = InpaintDataset(self.train_dataset_path, images_transformations_list,
                                           preprocessing=self.preprocessing_fn, img_bands=in_channels)
        self.valid_dataset = InpaintDataset(self.valid_dataset_path, images_transformations_list,
                                           preprocessing=self.preprocessing_fn, img_bands=in_channels, shuffle=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=72, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True)


#%% Cell feature extractor
### Feature Extractor ####
class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = torch.nn.Sequential(*vgg16.features[:5])
        self.enc_2 = torch.nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = torch.nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

#%% Cell loss
#### Losses ##

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss



class InpaintingLoss(torch.nn.Module):
    LAMBDA_DICT = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
    def __init__(self, extractor):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)
        
        total_loss = 0
        for key, coef in self.LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            total_loss += value
        return loss_dict, total_loss
#%% Cell model
class lightningInpaintModel(pl.LightningModule):
    
    def __init__(self, arch="Unet", encoder_name="resnext101_32x8d",
                 encoder_weights="imagenet", in_channels=3, out_channels=3, decoder_channels=[512,512,256,128,64], learning_rate=1e-5, **kwargs):
        super(lightningInpaintModel, self).__init__()
        self.save_hyperparameters()
        self.arch=arch
        self.encoder_name=encoder_name
        self.encoder_weights=encoder_weights                        
        self.out_channels=out_channels
        self.learning_rate=learning_rate
        
        # train params
        model_outputs_root_path= kwargs.get("model_outputs_root_path", "./models")
        tensorboard_logs_root_path= kwargs.get("tensorboard_logs_root_path", "./reports/tensorboard/")
        trial_name = "_".join([arch, encoder_name])
        self.output_path = os.path.join(model_outputs_root_path, trial_name)
        self.log_path = os.path.join(tensorboard_logs_root_path, trial_name)
        for dir_path in [self.output_path, self.log_path]:
            #if os.path.exists(dir_path):
            #    shutil.rmtree(dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        # Create model
        self.model = smp.create_model(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=out_channels, decoder_channels=decoder_channels)
        
        self.loss = InpaintingLoss(VGG16FeatureExtractor())
    
    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        scheduler = {
            "scheduler": scheduler, "interval": "epoch", "monitor": "val_total_loss",
        }  # logged value to monitor
        return [optimizer], [scheduler]
        
    def forward(self, image):
        # Forward pass
        return self.model(image)
    
    def training_step(self, train_batch, batch_idx):
        image, mask, gt = train_batch
        output = self.forward(image)
        loss_dict, total_loss = self.loss(image, mask, output, gt)
        
        for k,v in loss_dict.items():
            self.log('train_{}'.format(k), v, on_step=True, on_epoch=True)
        
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
        image, mask, gt = val_batch
        output = self.forward(image)
        loss_dict, total_loss = self.loss(image, mask, output, gt)
        
        for k,v in loss_dict.items():
            self.log('val_{}'.format(k), v, on_step=True, on_epoch=True)
        self.log("val_total_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss
    
    def image_grid(self, x, y, preds):  
        
        bs = preds.shape[0]
                
        for i in range(bs):
            fixed_rgb = (x[i]-x[i].min())/(x[i].max()-x[i].min())
            self.logger.experiment.add_image("rgb_{}".format(i),fixed_rgb,self.current_epoch)
            fixed_gt = (y[i]-y[i].min())/(y[i].max()-y[i].min())
            self.logger.experiment.add_image("gt_{}".format(i),fixed_gt,self.current_epoch)
            fixed_preds = (preds[i]-preds[i].min())/(preds[i].max()-preds[i].min())
            self.logger.experiment.add_image("preds_{}".format(i),fixed_preds,self.current_epoch)
    
    def validation_epoch_end(self, outputs):
                
        image, mask, gt = self.sample_val
        logits = self.forward(image.to(self.device))
        
        self.image_grid(image, gt, logits)
    
#%% Cell launch   
model_outputs_root_path = "models/inpaint"
tensorboard_logs_path = "reports/tensorboard/inpaint"
arch = "Unet"
decoder_channels = [256,256,128,64,32]

train_params = {
    "learning_rate":0.0005,
    "batch_size":8,
    "model_outputs_root_path":model_outputs_root_path,#"./models/bnc/",
    "tensorboard_logs_path":tensorboard_logs_path,#"./reports/tensorboard/",
    
    }

#TRAIN_DATASET_PATH="./data/processed/b_n_c_256_full/train"
#VALID_DATASET_PATH="./data/processed/b_n_c_256_full/valid"
in_channels=3
out_channels=3

c_light_model = lightningInpaintModel(arch=arch, in_channels=in_channels, out_channels=out_channels
                                  ,decoder_channels=decoder_channels, **train_params)


data_module = InpaintDataModule()
data_module.setup(dataset_path="./data/processed/inpaint_dataset/", encoder_name=c_light_model.encoder_name,
                  encoder_weights=c_light_model.encoder_weights,in_channels=in_channels,
                  out_channels=out_channels, batch_size=train_params["batch_size"])


#trainer = pl.Trainer( plugins=[RayPlugin(num_workers=2, use_gpu=True)] ,**c_light_model._get_trainer_params())
#trainer.fit(c_light_model, datamodule=data_module)

def _get_trainer_params(model):
    # Define callback behavior
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model.output_path,
        monitor="val_total_loss",
        mode="min",
        save_top_k=4,
        verbose=True,
    )
    # Specify where TensorBoard logs will be saved
    logger = pl.loggers.TensorBoardLogger(model.log_path, name="benchmark-model")
    trainer_params = {
        "callbacks": [checkpoint_callback],
        "max_epochs": 200,
        "min_epochs": 100,
        "default_root_dir": model.output_path,
        "logger": logger,
        #"accelerator":self.device,
        "gpus": 1 ,
    }
    return trainer_params


dataloader_iterator = iter(data_module.val_dataloader())
sample_val = next(dataloader_iterator)
c_light_model.sample_val = sample_val

c_light_model=c_light_model.cuda()
trainer = pl.Trainer(**_get_trainer_params(c_light_model))
trainer.fit(c_light_model, datamodule=data_module)    

preds = c_light_model.model.predict(sample_val[0].cuda())

mean=torch.Tensor(data_module.preprocessing_fn.keywords["mean"])
std=torch.Tensor(data_module.preprocessing_fn.keywords["std"])

X, mask, gt = sample_val
X=(X-X.min())/(X.max()-X.min())
mask=(mask-mask.min())/(mask.max()-mask.min())
gt=(gt-gt.min())/(gt.max()-gt.min())
fig, axes = plt.subplots(1,3)
s_index=5
axes[0].imshow(X[s_index].permute(1,2,0).cpu())
axes[1].imshow(preds[s_index].permute(1,2,0).cpu()*std+mean)
axes[2].imshow(gt[s_index].permute(1,2,0).cpu())
    
    
    
    
    
    
    
    
    
    
    
