#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 06:30:12 2021

@author: igor
"""
import numpy as np
from discriminator import discriminator_train, discriminator_eval
from wisard_tools import separate_classes, eval_predictions
from hamming import hamming_correction

def wisard_train (X, Y, classes, address_size):
    mapping = np.arange(X.shape[1])
    np.random.shuffle(mapping)
    X_mapped = X[:,mapping]
    # X_mapped = hamming_correction(X_mapped, address_size)
    
    X_class = separate_classes (X_mapped, Y, classes, address_size)
    
    model = {}
    
    for c in range (len(classes)):
        model[classes[c]] = discriminator_train(X_class[classes[c]])
    
    return model, mapping

def wisard_eval_bin (X, model, mapping, classes, address_size, bleaching=1, hamming=False):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    
    if hamming:
        X_mapped = hamming_correction(X_mapped, address_size)
    
    Y_pred = []
    
    # Eval for each sample
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))
        
        ####### Binarized model ####################
            
        for c in range (len(classes)):
            scores[c] = discriminator_eval(xti.astype(int), model[classes[c]], bleaching)
                
        ############################################        
        
        best_class = np.argmax(scores)    
        Y_pred.append(best_class)
    
    
    return np.array(Y_pred)


def wisard_eval (X, model, mapping, classes, address_size, min_bleaching=1, max_bleaching=100, hamming=False):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    
    if hamming:
        X_mapped = hamming_correction(X_mapped, address_size)
    
    Y_pred = []
    
    # Eval for each sample
    blc_cnt = 0
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))    
        
        # Increase bleaching until the max or there is no tie 
        for bleaching in range(min_bleaching,max_bleaching+1):
            if(bleaching==max_bleaching):
                # print('# WARNING max bleaching reached: '+str(bleaching))
                blc_cnt+=1
            tie = False
            for c in range (len(classes)):
                scores[c] = discriminator_eval(xti.astype(int), model[classes[c]], bleaching)
                
                # Compare with the previous                
                for c2 in range (c):
                    if scores[c]==scores[c2]:
                        tie = True
                        break
                if tie==True and bleaching<max_bleaching:
                    break                    
            # Stops if there is no more ties
            if tie==False:
                break
      
        
        best_class = np.argmax(scores)    
        Y_pred.append(best_class)
    
    # print('%d samples of %d reached the max bleaching' % (blc_cnt,n_samples))
    
    return np.array(Y_pred)

def get_above_bleaching_count (model, classes, bleaching):
    cnt = 0 
    for c in range (len(classes)):  
        for r in range(len(model[classes[c]])):                
            dict_tmp = model[classes[c]][r]
            for a in dict_tmp:
                if dict_tmp[a]>=bleaching:
                    cnt += 1
    return cnt

def wisard_find_threshold (X, Y, model, mapping, classes, address_size, min_bleaching=1, max_bleaching=100, hamming=False):
    delta = 0.001
    max_acc = 0
    max_threshold = 0
    max_cnt = 0
    best_acc = 0
    best_threshold = 0
    best_cnt = 0
    fall_cnt = 0
    
    
    for bleaching in range(min_bleaching,max_bleaching+1):
        
        Y_pred = wisard_eval_bin (X, model, mapping, classes, address_size, bleaching=bleaching, hamming=hamming)        
        acc_t = eval_predictions(Y, Y_pred, classes, do_plot=False)   
        cnt_t = get_above_bleaching_count (model, classes, bleaching)
        
        if acc_t>=max_acc:
            max_acc = acc_t
            max_threshold = bleaching
            max_cnt = cnt_t
            fall_cnt = 0
        else:
            fall_cnt += 1
        # print('<b: %d, acc: %.04f> '%(bleaching, acc_t))
        if fall_cnt>=5:
            break
        if acc_t>max_acc-delta:
            best_acc = acc_t
            best_threshold = bleaching                
            best_cnt = cnt_t
            
    return best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt
            