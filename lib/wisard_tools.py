#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 08:45:23 2021

@author: igor
"""
import numpy as np
from encoders import ThermometerEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import ctypes as c
from scipy.stats import norm
import cv2 

def wisard_data_encode(X, classes, resolution=1, minimum=0, maximum=1):
    n_dim = len(X.shape)
    if n_dim==2:
        o,f = X.shape
    elif (n_dim==3):
        o,m,n = X.shape
        f = m*n
    else:
        return 0, 0
    
    X_lst = np.zeros((o,f*resolution))
    
    if resolution>1:
        thermometer = ThermometerEncoder(minimum=minimum, maximum=maximum, resolution=resolution)
    for i in range(o):
        if n_dim==2:
            x_lst_t = X[i,:].reshape(-1).tolist()
        else:
            x_lst_t = X[i,:,:].reshape(-1).tolist()
        if resolution>1:
            x_lst_t = thermometer.encode(x_lst_t)
        else:
            x_lst_t = np.array(x_lst_t).reshape(1,-1)
            
        flat_x_lst = [item for sublist in x_lst_t for item in sublist]
        X_lst[i,:] = np.array(flat_x_lst)
    
    return X_lst

def mnist_data_encode_b(X):

    o,m,n = X.shape
    f = m*n
    
    X_lst = np.zeros((o,f))

    for i in range(o):

        # x_lst_t = X[i,:,:].reshape(-1).tolist()
        X_lst[i,:] = X[i,:,:].reshape(1,-1)
        xi_mean = np.mean(X_lst[i,:])
        
        X_lst[i,:] = np.where(X_lst[i,:] > xi_mean, 1, 0)
    
    return X_lst

def mnist_data_encode_t(X, minimum, maximum, resolution):
    # minimum = 0
    # maximum = 255
    # resolution = 8
    o,m,n = X.shape
    # m = 28
    # n = 28
    f = m*n
    
    X_lst = np.zeros((o,resolution*f))

    thermometer = ThermometerEncoder(minimum=minimum, maximum=maximum, resolution=resolution)
    
    for i in range(o):
        img = X[i,:,:]
        # img = cv2.resize(X[i,:,:], (m,n), interpolation = cv2.INTER_CUBIC)
        x_lst_t = img.reshape(-1).tolist()
        X_lst[i,:] = np.array(thermometer.encode(x_lst_t)).reshape(1,-1)

    
    return X_lst

def mnist_data_encode_z(X,x_mean_a, x_std_a):
    bits_per_input = 8
    o,m,n = X.shape
    f = m*n
    
    X_lst = np.zeros((o,bits_per_input*f))
    X_flat = X.reshape(o,-1)
    if len(x_mean_a)==0:
        x_mean = np.mean(X_flat, axis=0,keepdims=True)
        x_std = np.std(X_flat, axis=0,keepdims=True)
    else:
        x_mean = x_mean_a
        x_std = x_std_a
    std_skews = [norm.ppf((i+1)/(bits_per_input+1)) for i in range(bits_per_input)]
    
    x_tmp = np.zeros((f,bits_per_input))
    for i in range(o):
        for j in range(bits_per_input):
           x_tmp[:,j] = (X_flat[i,:] >= x_mean + (std_skews[j]*x_std)).astype(c.c_ubyte)
        X_lst[i,:] = x_tmp.reshape(1,-1)
    
    return X_lst, x_mean, x_std

def separate_classes (X, Y, classes, address_size):
    n_rams = int(X.shape[1]/address_size)
    
    X_class = {}
    
    for c in range (len(classes)):
        X_class_t = np.empty((0,n_rams),dtype=int)
        for i in range (X.shape[0]):
            if int(Y[i])==c:
                xt = X[i,:].reshape(-1, address_size)
                xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
                X_class_t = np.vstack([X_class_t, xti.reshape(1,-1)])
    
        X_class[classes[c]] = X_class_t
    return X_class


def eval_predictions(y_true, y_pred, classes, do_plot):

    test_acc = sum(y_pred == y_true) / len(y_true)

    
    # Display a confusion matrix
    # A confusion matrix is helpful to see how well the model did on each of 
    # the classes in the test set.
    if do_plot==True:
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
    return test_acc


# This function writes content to a file. 
def write2file(to_write, file_name='./log.txt'):
    # Append-adds at last 
    print(to_write)
    file = open(file_name,"a")#append mode 
    file.write(to_write) 
    file.close() 