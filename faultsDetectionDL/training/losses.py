

import torch
import torch.nn as nn
from torch.nn import functional as F
from segmentation_models_pytorch.base.modules import Activation
import segmentation_models_pytorch as smp


class multilabel_bce_loss(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.class_weights = weight
        
    def forward(self, logits, true):
        o_true = F.one_hot(true, 3)
        o_true = o_true.permute(0,3,1,2)
        loss = F.binary_cross_entropy(logits, o_true, weight=self.class_weights, reduction="mean")
        return loss

class CrossEntropyLoss(nn.CrossEntropyLoss):
    
    def __init__(self, logits_activation=None, labels_activation=None, **kwargs):
        super().__init__(**kwargs)
        self.logits_activation = Activation(logits_activation)
        self.labels_activation = Activation(labels_activation)
    
    def forward(self, input, target):
        input = self.logits_activation(input)
        target = self.labels_activation(target)
        return super().forward(input, target)
    

class PixelWiseWeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    
    def __init__(self, logits_activation=None, labels_activation=None, **kwargs):
        kwargs.update({"reduction": "none"})
        super().__init__(**kwargs)
        self.logits_activation = Activation(logits_activation)
        self.labels_activation = Activation(labels_activation)
    
    def forward(self, input, bundle_target):
        target, target_weights= bundle_target
        target_weights = target_weights.squeeze(1)
        input = self.logits_activation(input)
        target = self.labels_activation(target)
        loss = super().forward(input, target)
        w_loss = loss * target_weights
        return torch.mean(w_loss)

class PixelWiseWeightedBinaryCrossEntropyLoss(nn.BCEWithLogitsLoss):
    
    def __init__(self, logits_activation=None, labels_activation=None, **kwargs):
        self.class_weight = kwargs.pop("weight")
        kwargs.update({"reduce": False})#, "pos_weight":pos_weight})
        super().__init__(**kwargs)
        self.logits_activation = Activation(logits_activation)
        self.labels_activation = Activation(labels_activation)
    
    def forward(self, input, bundle_target):
        target, target_weights= bundle_target
        input = self.logits_activation(input)
        target = self.labels_activation(target)
        loss = super().forward(input, target)
        w_loss = (loss * target_weights).mean(dim=(0,2,3)) * self.class_weight
        return torch.mean(w_loss)
    
class DiceLoss(smp.losses.DiceLoss):
    def __init__(self, logits_activation=None, labels_activation=None, **kwargs):
        super().__init__(**kwargs)
        self.logits_activation = Activation(logits_activation)
        self.labels_activation = Activation(labels_activation)
    
    def forward(self, input, target):
        input = self.logits_activation(input)
        target = self.labels_activation(target)
        return super().forward(input, target)


class PixelWiseWeightedDiceLoss(nn.Module):
    def __init__(self, logits_activation=None, labels_activation=None, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.logits_activation = Activation(logits_activation)
        self.labels_activation = Activation(labels_activation)
    
    def forward(self, logits, labels_bundle):
        """
        logits: a tensor of shape (bs, num_class, H, W)
        labels_bundle: a tuple of two tensors 
            - labels: a tensor of shape (bs, num_class, H, W)
            - weights: a tensor of shape (bs, 1, H, W)
        """
        num_classes = logits.shape[1]
        
        labels, weights = labels_bundle
        
        #weights = weights.permute(0,2,3,1)
        logits = self.logits_activation(logits)
        labels = self.labels_activation(labels)
        
        #logits = F.one_hot(logits, num_classes)
        #one_hot_labels = F.one_hot(labels, num_classes)
        
        tp = (logits * labels * weights).sum(dim=[0,2,3])
        fp = (logits * (1-labels) * weights).sum(dim=[0,2,3])
        fn = ((1-logits) * labels * weights).sum(dim=[0,2,3])
        
        class_losses = 1 - 2 * tp / (2*tp+fp+fn+self.eps)
        return class_losses.mean()
        
        
        