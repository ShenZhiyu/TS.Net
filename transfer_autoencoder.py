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


#class AutoEncoder(nn.Module):
#    def __init__(self):
#        super(AutoEncoder, self).__init__()
#
#        self.encoder = nn.Sequential(
##            nn.Linear(28*28, 128),
##            nn.Tanh(),
##            nn.Linear(128, 64),
##            nn.Tanh(),
##            nn.Linear(64, 12),
##            nn.Tanh(),
##            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
#        )
#        self.decoder = nn.Sequential(
##            nn.Linear(3, 12),
##            nn.Tanh(),
##            nn.Linear(12, 64),
##            nn.Tanh(),
##            nn.Linear(64, 128),
##            nn.Tanh(),
##            nn.Linear(128, 28*28),
#            nn.Linear(28*28, 28*28),
##            nn.Sigmoid(),       # compress to a range (0, 1)
#        )

    def forward(self, x):
#        encoded = self.encoder(x)
        decoded = self.decoder(x)
        return decoded, decoded

class T2SNet(nn.Module):
    def __init__(self):
        super(T2SNet, self).__init__()
    
    def forward(self, input):
        output = 0
        return output

if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 1000
    BATCH_SIZE = 1
    LR = 0.005
    
    
#if __name__ == '__main1__':
#    # Hyper Parameters
#    EPOCH = 100
#    BATCH_SIZE = 5000
#    LR = 0.005         # learning rate
#    DOWNLOAD_MNIST = False
#    N_TEST_IMG = 10
#    INPUT_LABEL = 9
#    OUTPUT_LABEL = 2
#
#    # Mnist digits dataset
#    train_data = torchvision.datasets.MNIST(
#        root='./dataset/mnist/',
#        train=True,                                     # this is training data
#        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
#                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#        download=DOWNLOAD_MNIST,                        # download it if you don't have it
#    )
#
#    labels = train_data.train_labels.numpy()
#    input_data_index = labels==INPUT_LABEL
#    output_data_index = labels==OUTPUT_LABEL
#    
#    input_np = train_data.train_data.numpy()[input_data_index,:,:].astype(np.float)/255.
#    output_np = train_data.train_data.numpy()[output_data_index,:,:].astype(np.float)/255.
#    input_tensor = torch.FloatTensor(input_np[:5000,:,:])
#    output_tensor = torch.FloatTensor(output_np[:5000,:,:])
#    
#    input_loader = Data.DataLoader(dataset=input_tensor, batch_size=BATCH_SIZE, shuffle=True)
#    output_loader = Data.DataLoader(dataset=output_tensor, batch_size=BATCH_SIZE, shuffle=True)
#    
#    autoencoder = AutoEncoder()
#    
#    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1)
#    loss_func = nn.MSELoss()
#    
#    # initialize figure
#    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
#    plt.ion()   # continuously plot
#    
#    # original data (first row) for viewing
##    view_data = Variable(input_tensor[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor))
#    view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)
#    for i in range(N_TEST_IMG):
#        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())
#        
#    for epoch in range(EPOCH):
#        for step, x in enumerate(input_loader):
#            for step2, x2 in enumerate(output_loader):
#                b_x = Variable(x.view(-1, 28*28))
#                b_y = Variable(x2.view(-1, 28*28))   # batch y, shape (batch, 28*28)
#        
#                encoded, decoded = autoencoder(b_x)
#        
#                loss = loss_func(decoded, b_y)      # mean square error
##                loss = nn.MSEloss(decoded, b_y) + nn.L1??
#                optimizer.zero_grad()               # clear gradients for this training step
#                loss.backward()                     # backpropagation, compute gradients
#                optimizer.step()                    # apply gradients
#                
#                if step2 % 10 == 0:
#                    print('epoch:%d step1:%d step2:%d' % (epoch, step, step2), '| train loss: %.4f' % loss.data[0])
#    
#                    # plotting decoded image (second row)
#                    _, decoded_data = autoencoder(view_data)
#                    for i in range(N_TEST_IMG):
#                        a[1][i].clear()
#                        a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
#                        a[1][i].set_xticks(()); a[1][i].set_yticks(())
#                    plt.draw(); plt.pause(0.5)
#                    
#    # visualize in 3D plot
#    view_data = Variable(train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.)
#    encoded_data, _ = autoencoder(view_data)
#    fig = plt.figure(2); ax = Axes3D(fig)
#    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
#    values = train_data.train_labels[:200].numpy()
#    for x, y, z, s in zip(X, Y, Z, values):
#        c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
#    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
#    plt.show()