# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:15:33 2022

@author: cherif
"""

from __future__ import division
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from faultsDetectionDL.utils.roteqnet_utils import *

class RotConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, n_angles = 4, mode=1, old_transform=True):
        super(RotConv, self).__init__()

        kernel_size = ntuple(2)(kernel_size)
        stride = ntuple(2)(stride)
        padding = ntuple(2)(padding)
        dilation = ntuple(2)(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.mode = mode

        #Angles
        self.n_angles = n_angles
        self.angles = np.linspace(0,360,n_angles, endpoint=False)
        self.angle_tensors = []

        #Get interpolation variables
        self.interp_vars = []
        for angle in self.angles:
            out = get_filter_rotation_transforms(list(self.kernel_size), angle)
            self.interp_vars.append(out[:-1])
            self.mask = out[-1]

            self.angle_tensors.append( Variable(torch.FloatTensor( np.array([angle/ 180. * np.pi]) )) )

        self.weight1 = Parameter(torch.Tensor( out_channels, in_channels , *kernel_size))
        #If input is vector field, we have two filters (one for each component)
        self.weight2=None
        self.bias = None
        self.old_transform=old_transform
        if self.mode == 2:
            self.weight2 = Parameter(torch.Tensor( out_channels, in_channels, *kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.mode == 2:
            self.weight2.data.uniform_(-stdv, stdv)

    def mask_filters(self):
        self.weight1.data[self.mask.expand_as(self.weight1) == 0] = 1e-8
        if self.mode == 2:
            self.weight2.data[self.mask.expand_as(self.weight1) == 0] = 1e-8

    def _apply(self, func):
        # This is called whenever user calls model.cuda()
        # We intersect to replace tensors and variables with cuda-versions
        self.mask = func(self.mask)
        self.interp_vars = [[[func(el2) for el2 in el1] for el1 in el0] for el0 in self.interp_vars]
        self.angle_tensors = [func(el) for el in self.angle_tensors]

        return super(RotConv, self)._apply(func)


    def forward(self,input):
        #Uncomment this to turn on filter-masking
        #Todo: fix broken convergence when filter-masking is on
        #self.mask_filters()

        if self.mode == 1:
            outputs = []

            #Loop through the different filter-transformations
            for ind, interp_vars in enumerate(self.interp_vars):
                #Apply rotation
                weight = apply_transform(self.weight1, interp_vars, self.kernel_size, self.old_transform)

                #Do convolution
                out = F.conv2d(input, weight, None, self.stride, self.padding, self.dilation)
                outputs.append(out.unsqueeze(-1))

        if self.mode == 2:
            u = input[0]
            v = input[1]

            outputs = []
            # Loop through the different filter-transformations
            for ind, interp_vars in enumerate(self.interp_vars):
                angle = self.angle_tensors[ind]
                # Apply rotation
                wu = apply_transform(self.weight1, interp_vars, self.kernel_size, self.old_transform)
                wv = apply_transform(self.weight2, interp_vars, self.kernel_size, self.old_transform)

                # Do convolution for u
                wru = torch.cos(angle) * wu - torch.sin(angle ) * wv
                u_out = F.conv2d(u, wru, None, self.stride, self.padding, self.dilation)

                # Do convolution for v
                wrv = torch.sin(angle) * wu + torch.cos(angle) * wv
                v_out = F.conv2d(v, wrv, None, self.stride, self.padding, self.dilation)

                #Compute magnitude (p)
                outputs.append(  (u_out + v_out).unsqueeze(-1) )
                

        # Get the maximum direction (Orientation Pooling)
        strength, max_ind = torch.max(torch.cat(outputs, -1), -1)

        # Convert from polar representation q
        angle_map = max_ind.float() * (360. / len(self.angles) / 180. * np.pi)
        u = F.relu(strength) * torch.cos(angle_map)
        v = F.relu(strength) * torch.sin(angle_map)


        return u, v
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, mode={mode}, n_angles={n_angles}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        
        return s.format(**self.__dict__)

class VectorMaxPool(nn.Module):
    def __init__(self,  kernel_size, stride=None, padding=0, dilation=1,
                  ceil_mode=False):
            super(VectorMaxPool, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride or kernel_size
            self.padding = padding
            self.dilation = dilation
            self.ceil_mode = ceil_mode

    def forward(self,input):
        #Assuming input is vector field
        u = input[0]
        v = input[1]

        #Magnitude
        p = torch.sqrt( v**2 + u**2)
        #Max pool
        _, max_inds = F.max_pool2d(p, self.kernel_size, self.stride,
                     self.padding, self.dilation, self.ceil_mode,
                     return_indices=True)
        #Reshape to please pytorch
        s1 = u.size()
        s2 = max_inds.size()
        
        max_inds = max_inds.view(s1[0], s1[1], s2[2] * s2[3])
        
        u = u.view(s1[0], s1[1], s1[2] * s1[3])
        v = v.view(s1[0], s1[1], s1[2] * s1[3])
        
        #Select u/v components according to max pool on magnitude
        u = torch.gather(u, 2, max_inds)
        v = torch.gather(v, 2, max_inds)

        #Reshape back
        u = u.view(s1[0], s1[1], s2[2], s2[3])
        v = v.view(s1[0], s1[1], s2[2], s2[3])
        
        return u,v

class Vector2Magnitude(nn.Module):
    def __init__(self):
        super(Vector2Magnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        p = torch.sqrt(v ** 2 + u ** 2)
        return p

class Vector2Angle(nn.Module):
    def __init__(self):
        super(Vector2Angle, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        angle = torch.atan2(u, v)

        return angle

class VectorBatchNorm(nn.Module):
    def     __init__(self, num_features, eps=1e-5, momentum=0.5, affine=True):

        super(VectorBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

        if self.affine:
            self.weight = Parameter(torch.Tensor(1,num_features,1,1))
        else:
            self.register_parameter('weight', None)
        self.register_buffer('running_var', torch.ones(1,num_features,1,1))
        self.reset_parameters()


    def reset_parameters(self):
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()

    def forward(self, input):
        """
        Based on https://github.com/lberrada/bn.pytorch
        """
        if self.training:
            #Compute std
            std = self.std(input)

            alpha = self.weight / (std + self.eps)

            # update running variance
            self.running_var *= (1. - self.momentum)
            self.running_var += self.momentum * std.data ** 2
            # compute output
            u = input[0] * alpha
            v = input[1] * alpha

        else:
            alpha = self.weight.data / torch.sqrt(self.running_var + self.eps)

            # compute output
            u = input[0] * Variable(alpha)
            v = input[1] * Variable(alpha)
        return u,v

    def std(self, input):
        u = input[0]
        v = input[1]

        #Vector to magnitude
        p = torch.sqrt(u ** 2 + v ** 2)

        #We want to normalize the vector magnitudes,
        #therefore we ommit the mean (var = (p-p.mean())**2) 
        #since we do not want to move the center of the vectors.
        
        var = (p)**2 
        var = torch.mean(var, 0, keepdim=True) 
        var = torch.mean(var, 2, keepdim=True)
        var = torch.mean(var, 3, keepdim=True)
        std = torch.sqrt(var)

        return std

class VectorUpsampling(nn.Module):
    def __init__(self,  size=None, scale_factor=None, mode = 'bilinear'):
        super(VectorUpsampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        u = F.upsample(u, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        v = F.upsample(v, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


        return u, v
    


class RotConv2dReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        n_angles=5,
    ):
        super(RotConv2dReLU, self).__init__()

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        self.r_conv_1 = RotConv(in_channels, in_channels//4, [kernel_size, kernel_size], stride, kernel_size//2, n_angles=n_angles, mode=1) #The first RotConv must have mode=1 
        
        self.vec_to_mag = Vector2Magnitude() #This call converts the vector field to a conventional multichannel image/feature image
        
        self.conv_1 = nn.Conv2d(in_channels//4, out_channels, 1, stride, 0)
        
        if use_batchnorm == "inplace":
            self.batch_norm_1 = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            self.relu_1 = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            self.batch_norm_1 = nn.BatchNorm2d(out_channels)
            self.relu_1 = nn.ReLU(inplace=True)

        else:
            self.batch_norm_1 = nn.Identity()
            self.relu_1 = nn.ReLU(inplace=True)
        
        
        #nn.Dropout2d(0.7),
        #conv_2 = nn.Conv2d(128, out_channels, 1)
        
        """super(RotConv2dReLU, self).__init__(r_conv_1, v_maxpool_1, v_batchnorm_1,
                                            r_conv_2, v_maxpool_2, v_batchnorm_2,
                                            r_conv_3, vec_to_mag,
                                            conv_1, batch_norm_1, relu_1)"""
        
    def forward(self,x):
        x = self.r_conv_1(x)
        x = self.vec_to_mag(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        return x
        