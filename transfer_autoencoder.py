#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:20:32 2017

@author: shenzy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class T2SNet(nn.Module):
    def __init__(self):
        super(T2SNet, self).__init__()
    
    def forward(self, input_s, input_t, label_s, label_t):
        output = 0
        return output

if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 100
    BATCH_SIZE = 10
    LR = 0.005
    
    # data
    dataset_size = 100
    n_data = torch.ones(dataset_size, 2)
    x0 = torch.normal(2*n_data, 1)
    y0 = torch.zeros(dataset_size)
    x1 = torch.normal(-2*n_data, 1)
    y1 = torch.ones(dataset_size)
    
    xs = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    ys = torch.cat((y0, y1), ).type(torch.LongTensor)
    
    xs = Variable(xs)
    ys = Variable(ys)
#    plt.scatter(xs.data.numpy()[:, 0], xs.data.numpy()[:, 1], c=ys.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#    plt.xlim([-10,10])
#    plt.ylim([-10,10])
#    plt.show()
    
    x0 = torch.normal(3*n_data, 1)
    y0 = torch.zeros(dataset_size)
    x1 = torch.normal(-4*n_data, 1)
    y1 = torch.ones(dataset_size)
    
    xt = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    yt = torch.cat((y0, y1), ).type(torch.LongTensor)
    
    xt = Variable(xt)
    yt = Variable(yt)
#    plt.scatter(xt.data.numpy()[:, 0], xt.data.numpy()[:, 1], c=yt.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#    plt.xlim([-10,10])
#    plt.ylim([-10,10])
#    plt.show()
    
    xs_1 = torch.cat((xs.data, torch.ones(xs.data.shape[0], 1)), 1).type(torch.FloatTensor)
    xs_1 = Variable(xs_1)

    #
    T_pos = torch.ones(xs.data.shape[1]+1, xt.data.shape[1])
    T_neg = torch.ones(xs.data.shape[1]+1, xt.data.shape[1])
    W = torch.ones(xt.data.shape[0])/2
    T_pos = Variable(T_pos)
    T_neg = Variable(T_neg)
    W = Variable(W)
    
    proj_pos = torch.mm(xs_1, T_pos)
    proj_neg = torch.mm(xs_1, T_neg)
    
    












































