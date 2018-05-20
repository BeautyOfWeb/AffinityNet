__all__ = ['Factor1d', 'Block1d', 'TinyConv', 'FactorConv', 'FactorBlock', 'DenseFactorBlock']

import random
import sys
import os
import re
import copy
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from ..utils.gen_conv_params import *


class Factor1d(nn.Module):
    """Compute factors with 1d scope
    Args:
        projection: python dictionary; keys are index of output, values are list of indexes of input, 
                    eg, {1: [0, 29]}
        out_features: int; size of output torch.autograd.Variable; default None; 
                      If None, inferred from projection
        hidden_dim: int, default 10 (can be changed); number of hidden units feeded to the output;
                    hidden_dim = out_channels of the first conv layer
        in_channels: int, default 1; eg, if for RGB images, in_channels = 3
        out_channles: int, default 1; for the second conv layer
        bias: bool, default False; if True, use bias in conv
        nonlinearity: default: nn.ReLU(); this is followed after conv or batchnorm (specified by use_batchnorm)
    """
    def __init__(self, projection, out_features=None, hidden_dim=10, in_channels=1, out_channels=1, bias=False, 
                 nonlinearity=nn.ReLU(), use_batchnorm=True):
        super(Factor1d, self).__init__()
        self.projection = projection
        self.out_features = out_features
        if self.out_features is None:
            projections, idx_to_vars = reduce_projections([projection])
            self.projection = projections[0]
            self.idx_to_var = idx_to_vars[0]
            self.out_features = len(self.projection)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.params_dict = {}
        # It is better to use nn.ModuleList instead. Here for code variety, keep this implementation
        for k, v in self.projection.items():
            # Use conv to implement part-based fully connected layers (can use nn.Linear instead)
            # Possible extension: automatically "grow" layers and adjust kernel sizes based on input size
            self.params_dict[k] = nn.Sequential()
            self.params_dict[k].add_module('unit%d_conv0' % k, nn.Conv1d(in_channels, hidden_dim, 
                                                                         kernel_size=len(v), bias=bias))
            if use_batchnorm:
                self.params_dict[k].add_module('unit%d_bn' % k, nn.BatchNorm1d(hidden_dim))
            self.params_dict[k].add_module('unit%d_activation' % k, nonlinearity)
            self.params_dict[k].add_module('unit%d_conv1' % k, nn.Conv1d(hidden_dim, out_channels, 
                                                                         kernel_size=1, bias=bias))
            # <Question> not sure if I should register parameters and buffers this way
            for name, param in self.params_dict[k].named_parameters():
                self.register_parameter(str(k) + '.' + name, param)
            for name, m in self.params_dict[k].named_children():
                if isinstance(m, nn.BatchNorm1d):
                    self.register_buffer(str(k) + '.' + name + '.running_mean', 
                                         m._buffers['running_mean'])
                    self.register_buffer(str(k) + '.' + name + '.running_var', 
                                         m._buffers['running_var'])
                    
    def forward(self, x):
        if len(x.size()) == 2 and self.in_channels == 1:
            x = x.unsqueeze(1)
        output = Variable(torch.zeros(x.size(0), self.out_channels, self.out_features))
        for k, v in self.projection.items():
            output[:,:,k] = self.params_dict[k](x[:,:,torch.LongTensor(v)])
        return output

    
class Block1d(nn.Module):
    """Multiple feed forward Factor1d layers
    """
    def __init__(self, projections, out_dims=None, hidden_dims=10, in_channels=1, out_channels=1, 
                 bias=False, nonlinearity=nn.ReLU(), use_batchnorm=True):
        super(Block1d, self).__init__()
        self.projections = projections
        if out_dims is None:
            self.projections, self.idx_to_vars = reduce_projections(self.projections)
            out_dims = [len(proj) for proj in projections]
        num_layers = len(self.projections)
        assert len(out_dims) == num_layers
        hidden_dims = list(get_iter(hidden_dims, num_layers))
        out_channels = list(get_iter(out_channels, num_layers))
        self.layers = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.layers.add_module("Factor1d%d_{0}".format(i), 
                                   Factor1d(self.projections[i], out_dim, hidden_dims[i],
                                            in_channels=in_channels if i == 0 else out_channels[i - 1],
                                            out_channels=out_channels[i], bias=bias,
                                            nonlinearity=nonlinearity, use_batchnorm=use_batchnorm))

    def forward(self, x):
        return self.layers.forward(x)


class TinyConv(nn.Module):
    """Replace wide network with deep network; Each TinyConv for a single output Variable;
       i.e., "single-cell" output from multiple inputs (in its domain)
    Args:
        in_features: int or list/tuple of two ints; the dimentionality of 1d or 2d features
        in_channels: int; number of input planes used for nn.Conv2d
        out_channels: int, or list/tuple; number of output channels; 
            Can also seen as the dimension of hidden variables 
            used for representing one single output variable
        Other parameters are passed to gen_conv_params, except bias, nonlinearity and use_batchnorm
    """
    def __init__(self, in_features, in_channels=1, out_channels=1, kernel_size=3, 
                 stride=2, threshold=40, force_square=False, bias=False, 
                 nonlinearity=nn.ReLU(), use_batchnorm=False):
        super(TinyConv, self).__init__()
        if isinstance(in_features, int):
            self.feature_dim = 1
            if force_square:
                in_features = squaredims(in_features)
            else:
                in_features = (in_features, 1) # Use Conv2d as Conv1d (waste?)
        else:
            # Only handle 1d or 2d features; To do: 3d features with Conv3d
            self.feature_dim = 2
        assert isinstance(in_features, (tuple, list)) and len(in_features)==2
        conv_params = gen_conv_params(in_features, kernel_size=kernel_size, 
                                      stride=stride, threshold=threshold)
        self.in_features = in_features
        self.in_channels = in_channels
        self.layers = nn.Sequential()
        out_channels = list(get_iter(out_channels, len(conv_params)))
        for i, conv_param in enumerate(conv_params):
            # To do: add more sub-layers and refine
            in_channels = in_channels if i == 0 else out_channels[i-1]
            self.layers.add_module('conv%d' % i, 
                                   nn.Conv2d(in_channels, out_channels[i], kernel_size=conv_param[0], 
                                             stride=conv_param[1], padding=conv_param[2], bias=bias))
            if use_batchnorm:
                self.layers.add_module('batchnorm%d' % i, nn.BatchNorm2d(out_channels[i]))
            self.layers.add_module('activation%d' % i, nonlinearity)
            
    def forward(self, x):
        a = self.in_features[0]
        b = self.in_features[1]
        d = a * b
        if self.feature_dim == 1:
            d = x.size(-1) # d <= a*b
            # sanity check
            assert (x.dim()==2 and self.in_channels==1 and d==x.size(1)) or (
                x.dim()==3 and self.in_channels==x.size(1) and d==x.size(2))
        else:
            assert (x.dim()==3 and self.in_channels==1 and a==x.size(1) and b==x.size(2)) or (
                x.dim()==4 and self.in_channels==x.size(1) and a==x.size(2) and b==x.size(3))
        x_padded = Variable(torch.zeros(x.size(0), self.in_channels, a, b))
        x_padded.data.view(x.size(0), self.in_channels, -1)[:, :, :d] = x.data.view(
            x.size(0), self.in_channels, -1)
        output = self.layers.forward(x_padded)
        return output

    
class FactorConv(nn.Module):
    """Use TinyConv as units to do multiple output specified by argument projection
    Args:
        projection: python dictionary: keys are int, values are list of ints
        out_features: int or None (default); if None, infer from projection
        All other args are for TinyConv
    
    """
    def __init__(self, projection, out_features=None, in_channels=1, out_channels=1, kernel_size=3, 
                 stride=2, threshold=40, force_square=False, bias=False, 
                 nonlinearity=nn.ReLU(), use_batchnorm=False):
        super(FactorConv, self).__init__()
        self.projection = projection
        if out_features is None:
            projections, idx_to_vars = reduce_projections([self.projection])
            # be careful here; easily ignore this
            self.projection = projections[0]
            self.idx_to_var = idx_to_vars[0]
            out_features = len(self.projection)
        assert out_features > max(self.projection.keys()) # assume keys are int
        self.out_features = out_features
        self.in_channels = in_channels
        self.out_channels = list(get_iter(out_channels))[-1]
        self.units = nn.ModuleList()
        for k, v in self.projection.items():
            self.units.append(TinyConv(len(v), in_channels, out_channels, kernel_size, stride, 
                                       threshold, force_square, bias, nonlinearity, use_batchnorm))
            
    def forward(self, x):
        # for lazy input of size (N, d)
        if x.dim()==2 and self.in_channels==1:
            x = x.unsqueeze(1)
        assert x.size(1) == self.in_channels
        output = Variable(torch.zeros(x.size(0), self.out_channels, self.out_features))
        for i, (k, v) in enumerate(self.projection.items()):
            output[:, :, k] = self.units[i].forward(x[:, :, v])
        return output

    
class FactorBlock(nn.Module):
    def __init__(self, projections, out_features=None, in_channels=1, out_channels=1, kernel_size=3, 
                 stride=2, threshold=40, force_square=False, bias=False, 
                 nonlinearity=nn.ReLU(), use_batchnorm=False):
        super(FactorBlock, self).__init__()
        self.projections = projections
        if out_features is None:
            self.projections, self.idx_to_vars = reduce_projections(projections)
            out_features = [len(proj) for proj in self.projections]
        num_layers = len(self.projections)
        assert len(out_features) == num_layers
        out_channels = list(get_iter(out_channels, num_layers)) # For each layer, out_channels is an int
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module('FactorConv_{0}'.format(i), FactorConv(
                self.projections[i], out_features=out_features[i],
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i], kernel_size=kernel_size, stride=stride, 
                threshold=threshold, force_square=force_square, bias=bias,
                nonlinearity=nonlinearity, use_batchnorm=use_batchnorm))
            
    def forward(self, x):
        return self.layers.forward(x)   

    
class DenseFactorBlock(nn.Module):
    def __init__(self, projections, out_features=None, in_channels=1, out_channels=1, kernel_size=3, 
                 stride=2, threshold=40, force_square=False, bias=False, 
                 nonlinearity=nn.ReLU(), use_batchnorm=False):
        super(DenseFactorBlock, self).__init__()
        self.projections = projections
        if out_features is None:
            self.projections, self.idx_to_vars = reduce_projections(self.projections)
            out_features = [len(proj) for proj in self.projections]
        num_layers = len(self.projections)
        assert len(out_features) == num_layers
        out_channels = list(get_iter(out_channels, num_layers))
        self.in_channels = in_channels
        self.factor_layers = nn.ModuleList()
        self.bottlenecks = nn.Sequential() # Using Sequential here just for coding variety 
        for i in range(1, 1 + num_layers):
            for j in range(i):
                in_channels = self.in_channels if j == 0 else out_channels[j - 1]
                self.factor_layers.append(FactorConv(functools.reduce(join_dict, self.projections[j:i]),
                                                     out_features[i-1], in_channels, 
                                                     out_channels[i-1], kernel_size, 
                                                     stride, threshold, force_square, bias, 
                                                     nonlinearity, use_batchnorm))
            self.bottlenecks.add_module('bottleneck_{0}'.format(i-1), nn.Conv1d(i * out_channels[i-1],
                                                                       out_channels[i-1], 1))
    def forward(self, x):
        output = [x]
        for i in range(1, 1 + len(self.projections)):
            m = []
            for j in range(i):
                m.append(self.factor_layers[sum(range(i)) + j].forward(output[j]))
            output.append(self.bottlenecks[i-1].forward(torch.cat(m, 1)))
        return output[len(output)-1]  