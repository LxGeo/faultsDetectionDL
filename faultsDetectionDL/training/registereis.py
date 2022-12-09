import torch

from faultsDetectionDL.training.losses import PixelWiseWeightedDiceLoss, CrossEntropyLoss, DiceLoss, PixelWiseWeightedCrossEntropyLoss, PixelWiseWeightedBinaryCrossEntropyLoss
import segmentation_models_pytorch as smp

optimizers_registery = {    
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "LBFGS": torch.optim.LBFGS,    
}

loss_registery = {
    #"multilabel_bce_loss": multilabel_bce_loss,
    "smp_ce_loss": CrossEntropyLoss,
    "PixelWiseWeightedCrossEntropyLoss_ce_loss":PixelWiseWeightedCrossEntropyLoss,
    "PixelWiseWeightedBinaryCrossEntropyLoss_ce_loss":PixelWiseWeightedBinaryCrossEntropyLoss,
    "PixelWiseWeightedDiceLoss":PixelWiseWeightedDiceLoss,
    #"smp_jacc_loss": smp.losses.JaccardLoss,
    "smp_dicee_loss": DiceLoss,
    #"smp_focal_loss": smp.losses.FocalLoss,
}

schedulers_registry = {    
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR, 
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR, 
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR, 
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR, 
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau, 
    "StepLR": torch.optim.lr_scheduler.StepLR,     
}
