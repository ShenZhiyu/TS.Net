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
import time


class T2SNet(nn.Module):
    def __init__(self, n_feature, n_w):
        super(T2SNet, self).__init__()
        self.T_pos = nn.Parameter(torch.rand(n_feature, n_feature))
        self.bias_pos = nn.Parameter(torch.rand(1, n_feature))
        self.T_neg = nn.Parameter(torch.rand(n_feature, n_feature))
        self.bias_neg = nn.Parameter(torch.rand(1, n_feature))
        self.w = nn.Parameter(torch.rand(n_w, 1))
    
    def forward(self, Xtu, Xtn, Xtp):
        proj_u_pos = torch.add(torch.matmul(Xtu, self.T_pos), self.bias_pos)
        proj_p_pos = torch.add(torch.matmul(Xtp, self.T_pos), self.bias_pos)
        proj_u_neg = torch.add(torch.matmul(Xtu, self.T_neg), self.bias_neg)
        proj_n_neg = torch.add(torch.matmul(Xtn, self.T_neg), self.bias_neg)
        
        proj_u = torch.add(torch.mul(proj_u_pos, torch.sigmoid(self.w)), torch.mul(proj_u_neg, 1. - torch.sigmoid(self.w)))
        return proj_u, proj_n_neg, proj_p_pos

D = nn.Sequential(                      # Discriminator
    nn.Linear(2, 2),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)
Dp = nn.Sequential(                      # Discriminator
    nn.Linear(2, 2),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)
Dn = nn.Sequential(                      # Discriminator
    nn.Linear(2, 2),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)

if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 100
    BATCH_SIZE = 10
    LR_D = 0.0005
    LR_G = 0.0005
    
    # data
    dataset_size = 100
    n_data = torch.ones(dataset_size, 2)
    x0 = torch.normal(4*n_data, 1)
    y0 = torch.zeros(dataset_size)
    x1 = torch.normal(-2*n_data, 1)
    y1 = torch.ones(dataset_size)
    
    xsn, xsp = x0, x1
    xs = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    ys = torch.cat((y0, y1), ).type(torch.LongTensor)
    
    xs, xsn, xsp = Variable(xs), Variable(xsn), Variable(xsp)
    ys = Variable(ys)
#    plt.scatter(xs.data.numpy()[:, 0], xs.data.numpy()[:, 1], c=ys.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#    plt.xlim([-10,10])
#    plt.ylim([-10,10])
#    plt.show()
    
    x0 = torch.normal(12*n_data, 1)
    y0 = torch.zeros(dataset_size)
    x1 = torch.normal(-20.5*n_data, 1)
    y1 = torch.ones(dataset_size)
    
    xtn, xtp = x0[0:5,:], x1[0:5,:]
    xt = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    yt = torch.cat((y0, y1), ).type(torch.LongTensor)
    
    xt, xtn, xtp = Variable(xt), Variable(xtn), Variable(xtp)
    yt = Variable(yt)
#    plt.scatter(xt.data.numpy()[:, 0], xt.data.numpy()[:, 1], c=yt.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#    plt.xlim([-10,10])
#    plt.ylim([-10,10])
#    plt.show()
    
#    xs_1 = torch.cat((xs.data, torch.ones(xs.data.shape[0], 1)), 1).type(torch.FloatTensor)
#    xs_1 = Variable(xs_1)

    # Network
    tsNet = T2SNet(xt.data.shape[1], xt.data.shape[0])
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    opt_Dp = torch.optim.Adam(Dp.parameters(), lr=LR_D)
    opt_Dn = torch.optim.Adam(Dn.parameters(), lr=LR_D)
    opt_G = torch.optim.Adam(tsNet.parameters(), lr=LR_G)
    
    for step in range(50000):
        p_u, p_0, p_1 = tsNet(xt, xtn, xtp)
        
        prob_s0 = D(xs)
        prob_t0 = D(p_u)
        D_loss = -1.* (torch.mean(torch.log(prob_s0)) + torch.mean(torch.log(1. - prob_t0)))

        prob_s1 = Dn(xsn)
        prob_t1 = Dn(p_0)
        Dn_loss = -1.* (torch.mean(torch.log(prob_s1)) + torch.mean(torch.log(1. - prob_t1)))
        
        prob_s2 = Dp(xsp)
        prob_t2 = Dp(p_1)
        Dp_loss = -1.* (torch.mean(torch.log(prob_s2)) + torch.mean(torch.log(1. - prob_t2)))
        
        G_loss = 1.*(torch.mean(torch.log(1. - prob_t0))+torch.mean(torch.log(1. - prob_t1))+torch.mean(torch.log(1. - prob_t2)))
        
        opt_D.zero_grad()
        D_loss.backward(retain_variables=True)      # retain_variables for reusing computational graph
        opt_D.step()
        
        opt_Dn.zero_grad()
        Dn_loss.backward(retain_variables=True)      # retain_variables for reusing computational graph
        opt_Dn.step()
        
        opt_Dp.zero_grad()
        Dp_loss.backward(retain_variables=True)      # retain_variables for reusing computational graph
        opt_Dp.step()
        
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()
        
        if step % 100 == 0:
            print('step:%d Total_loss:%1.2f'%(step,D_loss.data[0]+Dn_loss.data[0]+Dp_loss.data[0]+G_loss.data[0]))
            print('D_loss:%1.2f  Dn_loss:%1.2f  Dp_loss:%1.2f  G_loss:%1.2f'%(D_loss.data[0],Dn_loss.data[0],Dp_loss.data[0],G_loss.data[0]))
            
            plt.scatter(xs.data.numpy()[:, 0], xs.data.numpy()[:, 1], c=ys.data.numpy(), s=100, lw=0)
            plt.show()
            
            plt.scatter(xt.data.numpy()[:, 0], xt.data.numpy()[:, 1], c=yt.data.numpy(), s=100, lw=0)
            plt.show()
    
#            plt.scatter(xs.data.numpy()[:, 0], xs.data.numpy()[:, 1], c=ys.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#            plt.scatter(p_u.data.numpy()[:, 0], p_u.data.numpy()[:, 1], c=yt.data.numpy(), s=100, lw=0, cmap='RdYlGn')
            plt.scatter(np.concatenate((xs.data.numpy()[:, 0], p_u.data.numpy()[:, 0]), 0), 
                        np.concatenate((xs.data.numpy()[:, 1], p_u.data.numpy()[:, 1]), 0), 
                        c=np.concatenate((ys.data.numpy(), yt.data.numpy()+2), 0), 
                        s=100, lw=0)
            plt.show()
#            time.sleep(1)
#    for name, param in tsNet.state_dict().items():
#        if name == 'w':
#            print(name, param)
