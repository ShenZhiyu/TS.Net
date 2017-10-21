#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:55:26 2017

@author: zhiyu
"""
import numpy as np
import pickle

domains=['books', 'dvd', 'electronics', 'kitchen']


for d in range(4):
    f_n = open(domains[d]+'/negative.review', 'r')
    f_p = open(domains[d]+'/positive.review', 'r')
    f_u = open(domains[d]+'/unlabeled.review', 'r')
    
    dict1 = []
    
    ###
    y_n = []
    lines_n = f_n.readlines()
    for ln in lines_n:
        tolist = ln.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                y_n.append(-1)
            else:
                if i.split(':')[0] in tolist:
                    pass
                else:
                    dict1.append(i.split(':')[0])
    
    y_p = []
    lines_p = f_p.readlines()
    for lp in lines_p:
        tolist = lp.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                y_p.append(1)
            else:
                if i.split(':')[0] in tolist:
                    pass
                else:
                    dict1.append(i.split(':')[0])
                    
    y_u = []
    lines_u = f_u.readlines()
    for lu in lines_u:
        tolist = lu.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                if i.split(':')[1] == 'positive\n':
                    y_u.append(1)
                elif i.split(':')[1] == 'negative\n':
                    y_u.append(-1)
    #        else:
    #            if i.split(':')[0] in tolist:
    #                pass
    #            else:
    #                dict1.append(i.split(':')[0])
    ###
    y_n = np.array(y_n)
    x_n = np.zeros([y_n.shape[0], len(dict1)])
    index = 0
    for ln in lines_n:
        print('0:'+str(index))
        tolist = ln.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                pass
            else:
                x_n[index, dict1.index(i.split(':')[0])] = int(i.split(':')[1])
        index += 1
    
    y_p = np.array(y_p)
    x_p = np.zeros([y_p.shape[0], len(dict1)])
    index = 0
    for lp in lines_p:
        print('1:'+str(index))
        tolist = lp.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                pass
            else:
                x_p[index, dict1.index(i.split(':')[0])] = int(i.split(':')[1])
        index += 1
    
    temp = np.sum(x_n, 0)+np.sum(x_p, 0)
    dict2 = []
    index = 0
    for i in temp>100:
        if i:
            dict2.append(dict1[index])
        index += 1
       
    y_n = np.array(y_n)
    x_n = np.zeros([y_n.shape[0], len(dict2)])
    index = 0
    for ln in lines_n:
        print('0: '+str(index))
        tolist = ln.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                pass
            elif i.split(':')[0] in dict2:
                x_n[index, dict2.index(i.split(':')[0])] = int(i.split(':')[1])
        index += 1
    
    y_p = np.array(y_p)
    x_p = np.zeros([y_p.shape[0], len(dict2)])
    index = 0
    for lp in lines_p:
        print('1: '+str(index))
        tolist = lp.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                pass
            elif i.split(':')[0] in dict2:
                x_p[index, dict2.index(i.split(':')[0])] = int(i.split(':')[1])
        index += 1
    
    y_u = np.array(y_u)
    x_u = np.zeros([y_u.shape[0], len(dict2)])
    index = 0
    for lu in lines_u:
        print('2: '+str(index))
        tolist = lu.split(' ')
        for i in tolist:
            if i.split(':')[0] == '#label#':
                pass
            elif i.split(':')[0] in dict2:
                x_u[index, dict2.index(i.split(':')[0])] = int(i.split(':')[1])
        index += 1
        
    pickle.dump(y_n, open(domains[d]+'_y_n', 'wb'))
    pickle.dump(x_n, open(domains[d]+'_x_n', 'wb'))
    pickle.dump(y_p, open(domains[d]+'_y_p', 'wb'))
    pickle.dump(x_p, open(domains[d]+'_x_p', 'wb'))
    pickle.dump(y_u, open(domains[d]+'_y_u', 'wb'))
    pickle.dump(x_u, open(domains[d]+'_x_u', 'wb'))