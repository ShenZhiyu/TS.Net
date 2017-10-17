#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:16:29 2017

@author: zhiyu
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
from svmutil import *


class T2SNet(nn.Module):
    def __init__(self, n_feature, n_w):
        super(T2SNet, self).__init__()
        self.T_pos = nn.Parameter(torch.rand(n_feature, n_feature)-0.5)
        self.bias_pos = nn.Parameter(torch.rand(1, n_feature)-0.5)
        self.T_neg = nn.Parameter(torch.rand(n_feature, n_feature)-0.5)
        self.bias_neg = nn.Parameter(torch.rand(1, n_feature)-0.5)
        self.w = nn.Parameter(torch.rand(n_w, 1))
    
    def forward(self, Xtu, Xtn, Xtp):
        proj_u_pos = torch.add(torch.matmul(Xtu, self.T_pos), self.bias_pos)
        proj_p_pos = torch.add(torch.matmul(Xtp, self.T_pos), self.bias_pos)
        proj_n_pos = torch.add(torch.matmul(Xtn, self.T_pos), self.bias_pos)
        proj_u_neg = torch.add(torch.matmul(Xtu, self.T_neg), self.bias_neg)
        proj_n_neg = torch.add(torch.matmul(Xtn, self.T_neg), self.bias_neg)
        proj_p_neg = torch.add(torch.matmul(Xtp, self.T_neg), self.bias_neg)
        
        proj_u = torch.add(torch.mul(proj_u_pos, torch.sigmoid(self.w)), torch.mul(proj_u_neg, 1. - torch.sigmoid(self.w)))
#        w_loss = - torch.matmul(torch.t(self.w), self.w)
        return torch.sigmoid(proj_u), torch.sigmoid(proj_n_neg), torch.sigmoid(proj_p_pos), torch.sigmoid(proj_p_neg), torch.sigmoid(proj_n_pos)

D = nn.Sequential(                      # Discriminator
    nn.Linear(28*28, 256),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)
Dp = nn.Sequential(                      # Discriminator
    nn.Linear(28*28, 256),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)
Dn = nn.Sequential(                      # Discriminator
    nn.Linear(28*28, 256),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)
Dpf = nn.Sequential(                      # Discriminator
    nn.Linear(28*28, 256),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)
Dnf = nn.Sequential(                      # Discriminator
    nn.Linear(28*28, 256),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)

if __name__ == '__main__':
    # Hyper Parameters
#    EPOCH = 100
#    BATCH_SIZE = 10
    LR_D = 0.0005
    LR_G = 0.5
    DOWNLOAD_MNIST = False
    N_TEST_IMG = 10
    LABELED_TARGET = 3
    INPUT0_LABEL = 1
    INPUT1_LABEL = 2
    OUTPUT0_LABEL = 3
    OUTPUT1_LABEL = 4
    
    # Mnist digits dataset
    train_data = torchvision.datasets.MNIST(
        root='./dataset/mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,                        # download it if you don't have it
    )
    
    labels = train_data.train_labels.numpy()
    
    s0_np = train_data.train_data.numpy()[labels==INPUT0_LABEL,:,:].astype(np.float)/255.
    s1_np = train_data.train_data.numpy()[labels==INPUT1_LABEL,:,:].astype(np.float)/255.
    t0_np = train_data.train_data.numpy()[labels==OUTPUT0_LABEL,:,:].astype(np.float)/255.
    t1_np = train_data.train_data.numpy()[labels==OUTPUT1_LABEL,:,:].astype(np.float)/255.
    
    # Variable
    xs, xsn, xsp = Variable(torch.FloatTensor(np.concatenate((s0_np[0:1000],s1_np[0:1000]), 0)).view(-1,28*28)), Variable(torch.FloatTensor(s0_np[0:1000]).view(-1,28*28)), Variable(torch.FloatTensor(s1_np[0:1000]).view(-1,28*28))
    xt, xtn, xtp = Variable(torch.FloatTensor(np.concatenate((t0_np[0:1000],t1_np[0:1000]), 0)).view(-1,28*28)), Variable(torch.FloatTensor(t0_np[0:LABELED_TARGET]).view(-1,28*28)), Variable(torch.FloatTensor(t1_np[0:LABELED_TARGET]).view(-1,28*28))
    
    # cuda()
    xs, xsn, xsp = xs.cuda(), xsn.cuda(), xsp.cuda()
    xt, xtn, xtp = xt.cuda(), xtn.cuda(), xtp.cuda()
    
    # Network
    tsNet = T2SNet(28*28, xt.data.shape[0])
    tsNet = tsNet.cuda()
    D = D.cuda()
    Dp = Dp.cuda()
    Dn = Dn.cuda()
    Dpf = Dpf.cuda()
    Dnf = Dnf.cuda()
    
    # Optim
    opt_D = torch.optim.SGD(D.parameters(), lr=LR_D)
    opt_Dp = torch.optim.SGD(Dp.parameters(), lr=LR_D)
    opt_Dn = torch.optim.SGD(Dn.parameters(), lr=LR_D)
    opt_Dpf = torch.optim.SGD(Dpf.parameters(), lr=LR_D)
    opt_Dnf = torch.optim.SGD(Dnf.parameters(), lr=LR_D)
    opt_G = torch.optim.SGD(tsNet.parameters(), lr=LR_G)
    
    for step in range(50000):
        p_u, p_0, p_1, p_0f, p_1f = tsNet(xt, xtn, xtp)
        
        prob_s0 = D(xs)
        prob_t0 = D(p_u)
        D_loss = -10.**0 * (torch.mean(torch.log(prob_s0)) + torch.mean(torch.log(1. - prob_t0)))

        prob_s1 = Dn(xsn)
        prob_t1 = Dn(p_0)
        Dn_loss = -10.**0 * (torch.mean(torch.log(prob_s1)) + torch.mean(torch.log(1. - prob_t1)))
        
        prob_s2 = Dp(xsp)
        prob_t2 = Dp(p_1)
        Dp_loss = -10.**0 * (torch.mean(torch.log(prob_s2)) + torch.mean(torch.log(1. - prob_t2)))
        
#        G_loss = 10.**0 * (torch.mean(torch.log(1. - prob_t1))+torch.mean(torch.log(1. - prob_t2)))
        G_loss = 10.**0 * (torch.mean(torch.log(1. - prob_t0))+torch.mean(torch.log(1. - prob_t1))+torch.mean(torch.log(1. - prob_t2)))
        
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

        prob_s3 = Dnf(xsn)
        prob_t3 = Dnf(p_0f)
        Dnf_loss = -10.**0 * (torch.mean(torch.log(prob_s3)) + torch.mean(torch.log(1. - prob_t3)))

        prob_s4 = Dpf(xsp)
        prob_t4 = Dpf(p_1f)
        Dpf_loss = -10.**0 * (torch.mean(torch.log(prob_s4)) + torch.mean(torch.log(1. - prob_t4)))

        opt_Dnf.zero_grad()
        opt_G.zero_grad()
        Dnf_loss.backward(retain_variables=True)      # retain_variables for reusing computational graph
        opt_Dnf.step()
        opt_G.step()

        opt_Dpf.zero_grad()
        opt_G.zero_grad()
        Dpf_loss.backward(retain_variables=True)      # retain_variables for reusing computational graph
        opt_Dpf.step()
        opt_G.step()
        
        if step % 100 == 0:
            print('step: %d'%(step))
            print('D_loss:%.5f\nDn_loss:%.5f\nDp_loss:%.5f\nG_loss:%.5f\nDnf_loss:%.5f\nDpf_loss:%.5f'\
                  %(D_loss.data[0],Dn_loss.data[0],Dp_loss.data[0],G_loss.data[0],Dnf_loss.data[0],Dpf_loss.data[0]))
            
            plt.figure()
            
            plt.subplot(431)
            plt.imshow(np.reshape(xtn.data[0].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(432)
            plt.imshow(np.reshape(p_0.data[0].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(433)
            plt.imshow(np.reshape(p_1f.data[0].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(434)
            plt.imshow(np.reshape(xtn.data[1].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(435)
            plt.imshow(np.reshape(p_0.data[1].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(436)
            plt.imshow(np.reshape(p_1f.data[1].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(437)
            plt.imshow(np.reshape(xtp.data[0].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(438)
            plt.imshow(np.reshape(p_0f.data[0].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(439)
            plt.imshow(np.reshape(p_1.data[0].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(4,3,10)
            plt.imshow(np.reshape(xtp.data[1].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(4,3,11)
            plt.imshow(np.reshape(p_0f.data[1].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.subplot(4,3,12)
            plt.imshow(np.reshape(p_1.data[1].cpu().numpy(), (28,28)), cmap='gray')
            
            plt.show()
            plt.savefig('fig/0_'+str(int(step/100))+'.jpg', dpi=75)
            
#            plt.scatter(xs.data.numpy()[:, 0], xs.data.numpy()[:, 1], c=ys.data.numpy(), s=100, lw=0)
#            plt.show()
            
#            plt.scatter(xt.data.numpy()[:, 0], xt.data.numpy()[:, 1], c=yt.data.numpy(), s=100, lw=0)
#            plt.show()
    
#            plt.scatter(xs.data.numpy()[:, 0], xs.data.numpy()[:, 1], c=ys.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#            plt.scatter(p_u.data.numpy()[:, 0], p_u.data.numpy()[:, 1], c=yt.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#            plt.scatter(np.concatenate((xs.data.numpy()[:, 0], p_u.data.numpy()[:, 0]), 0), 
#                        np.concatenate((xs.data.numpy()[:, 1], p_u.data.numpy()[:, 1]), 0), 
#                        c=np.concatenate((ys.data.numpy(), yt.data.numpy()+2), 0), 
#                        s=100, lw=0)
#            plt.show()
    for name, param in tsNet.state_dict().items():
        if name == 'w':
            w = param.cpu().numpy()
     
    # source model
    for t in range(10):
        y = np.concatenate((np.zeros((1,1000)),np.ones((1,1000))),1).tolist()[0]
        x = np.reshape(np.concatenate((s0_np[0:1000],s1_np[0:1000]),0),(2000,28*28)).tolist()
        
        svm_train_opt = '-t 0 -c 0.001 -b 1 -q'
        model_2000 = svm_train(y, x, svm_train_opt)
        
        p_label, p_acc, p_val = svm_predict(y, np.reshape(np.concatenate((s0_np[1000:2000],s1_np[1000:2000]),0),(2000,28*28)).tolist(), model_2000, '-b 1 -q')
        smodel_to_s_acc = p_acc[0]
        p_label, p_acc, p_val = svm_predict(np.concatenate((np.zeros((1,1000)),np.ones((1,1000))),1).tolist()[0], np.reshape(np.concatenate((t0_np[0:1000],t1_np[0:1000]),0),(2000,28*28)).tolist(), model_2000, '-b 1 -q')
        smodel_to_t_acc = p_acc[0]
        p_u, p_0, p_1, p_0f, p_1f = tsNet(xt, xt, xt)
        p_label0, p_acc0, p_val0 = svm_predict(np.zeros([1,2000]).tolist()[0], p_0.data.cpu().numpy().tolist(), model_2000, '-b 1 -q')
        p_label1, p_acc1, p_val1 = svm_predict(np.ones([1,2000]).tolist()[0], p_1.data.cpu().numpy().tolist(), model_2000, '-b 1 -q')
        pv0 = np.array(p_val0)[:,0]
        pv1 = np.array(p_val1)[:,1]
        p_labelt = np.sign(pv1-pv0).astype(int)
        smodel_to_p_acc = (np.sum(p_labelt[0:1000]==-1)+np.sum(p_labelt[1000:2000]==1))/2000
        print('smodel_to_s_acc: %f' % smodel_to_s_acc)
        print('smodel_to_t_acc: %f' % smodel_to_t_acc)
        print('smodel_to_p_acc: %f' % (smodel_to_p_acc*100))